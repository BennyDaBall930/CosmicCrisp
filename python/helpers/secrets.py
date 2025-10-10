import re
import threading
from dataclasses import dataclass
from io import StringIO
from typing import Callable, Dict, List, Literal, Optional, Set

from dotenv.parser import parse_stream

from python.helpers import files
from python.helpers.errors import RepairableException

ALIAS_PATTERN = r"§§secret\(([A-Za-z_][A-Za-z0-9_]*)\)"


def alias_for_key(key: str, placeholder: str = "§§secret({key})") -> str:
    key = key.upper()
    return placeholder.format(key=key)


@dataclass
class EnvLine:
    raw: str
    type: Literal["pair", "comment", "blank", "other"]
    key: Optional[str] = None
    value: Optional[str] = None
    key_part: Optional[str] = None
    inline_comment: Optional[str] = None


class StreamingSecretsFilter:
    def __init__(self, key_to_value: Dict[str, str], min_trigger: int = 3):
        self.min_trigger = max(1, int(min_trigger))
        self.value_to_key: Dict[str, str] = {
            v: k for k, v in key_to_value.items() if isinstance(v, str) and v
        }
        self.secret_values: List[str] = [v for v in self.value_to_key.keys() if v]
        self.prefixes: Set[str] = set()
        for value in self.secret_values:
            for i in range(self.min_trigger, len(value) + 1):
                self.prefixes.add(value[:i])
        self.max_len: int = max((len(v) for v in self.secret_values), default=0)
        self.pending: str = ""

    def _replace_full_values(self, text: str) -> str:
        for val in sorted(self.secret_values, key=len, reverse=True):
            if not val:
                continue
            key = self.value_to_key.get(val, "")
            if key:
                text = text.replace(val, alias_for_key(key))
        return text

    def _longest_suffix_prefix(self, text: str) -> int:
        max_check = min(len(text), self.max_len)
        for length in range(max_check, self.min_trigger - 1, -1):
            suffix = text[-length:]
            if suffix in self.prefixes:
                return length
        return 0

    def process_chunk(self, chunk: str) -> str:
        if not chunk:
            return ""
        self.pending += chunk
        self.pending = self._replace_full_values(self.pending)

        hold_len = self._longest_suffix_prefix(self.pending)
        if hold_len > 0:
            emit = self.pending[:-hold_len]
            self.pending = self.pending[-hold_len:]
        else:
            emit = self.pending
            self.pending = ""
        return emit

    def finalize(self) -> str:
        if not self.pending:
            return ""

        hold_len = self._longest_suffix_prefix(self.pending)
        if hold_len > 0:
            safe = self.pending[:-hold_len]
            result = safe + "***"
        else:
            result = self.pending
        self.pending = ""
        return result


class SecretsManager:
    SECRETS_FILE = "tmp/secrets.env"
    PLACEHOLDER_PATTERN = ALIAS_PATTERN
    MASK_VALUE = "***"

    _instance: Optional["SecretsManager"] = None
    _secrets_cache: Optional[Dict[str, str]] = None
    _last_raw_text: Optional[str] = None

    @classmethod
    def get_instance(cls) -> "SecretsManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._lock = threading.RLock()
        self._secrets_file_rel = self.SECRETS_FILE

    def set_secrets_file(self, relative_path: str):
        with self._lock:
            self._secrets_file_rel = relative_path
            self.clear_cache()

    def read_secrets_raw(self) -> str:
        try:
            content = files.read_file(self._secrets_file_rel)
            self._last_raw_text = content
            return content
        except Exception:
            self._last_raw_text = ""
            return ""

    def _write_secrets_raw(self, content: str):
        files.write_file(self._secrets_file_rel, content)

    def load_secrets(self) -> Dict[str, str]:
        with self._lock:
            if self._secrets_cache is not None:
                return self._secrets_cache
            secrets: Dict[str, str] = {}
            try:
                content = self.read_secrets_raw()
                self._last_raw_text = content
                if content:
                    secrets = self.parse_env_content(content)
            except Exception:
                secrets = {}
            self._secrets_cache = secrets
            return secrets

    def save_secrets(self, secrets_content: str):
        with self._lock:
            self._write_secrets_raw(secrets_content)
            self._secrets_cache = self.parse_env_content(secrets_content)
            self._last_raw_text = secrets_content

    def save_secrets_with_merge(self, submitted_content: str):
        with self._lock:
            if self._last_raw_text is not None:
                existing_text = self._last_raw_text
            else:
                try:
                    existing_text = self.read_secrets_raw()
                except Exception as exc:
                    if self.MASK_VALUE in submitted_content:
                        raise RepairableException(
                            "Saving secrets failed because existing secrets could not be read to preserve masked values and comments. Please retry."
                        ) from exc
                    existing_text = ""
            merged_lines = self._merge_env(existing_text, submitted_content)
            merged_text = self._serialize_env_lines(merged_lines)
            self.save_secrets(merged_text)

    def get_keys(self) -> List[str]:
        secrets = self.load_secrets()
        return list(secrets.keys())

    def get_secrets_for_prompt(self) -> str:
        content = self._last_raw_text or self.read_secrets_raw()
        if not content:
            return ""
        env_lines = self.parse_env_lines(content)
        return self._serialize_env_lines(
            env_lines,
            with_values=False,
            with_comments=True,
            with_blank=True,
            with_other=True,
            key_formatter=alias_for_key,
        )

    def create_streaming_filter(self) -> "StreamingSecretsFilter":
        return StreamingSecretsFilter(self.load_secrets())

    def replace_placeholders(self, text: str) -> str:
        if not text:
            return text

        secrets = self.load_secrets()

        def replacer(match):
            key = match.group(1).upper()
            if key in secrets:
                return secrets[key]
            available_keys = ", ".join(secrets.keys())
            raise RepairableException(
                f"Secret placeholder '{alias_for_key(key)}' not found in secrets store.\nAvailable secrets: {available_keys}"
            )

        return re.sub(self.PLACEHOLDER_PATTERN, replacer, text)

    def change_placeholders(self, text: str, new_format: str) -> str:
        if not text:
            return text

        secrets = self.load_secrets()
        result = text
        for key, _value in sorted(
            secrets.items(), key=lambda item: len(item[1]), reverse=True
        ):
            result = result.replace(alias_for_key(key), new_format.format(key=key))
        return result

    def mask_values(
        self, text: str, min_length: int = 4, placeholder: str = "§§secret({key})"
    ) -> str:
        if not text:
            return text

        secrets = self.load_secrets()
        result = text
        for key, value in sorted(
            secrets.items(), key=lambda item: len(item[1]), reverse=True
        ):
            if value and len(value.strip()) >= min_length:
                result = result.replace(value, alias_for_key(key, placeholder))
        return result

    def get_masked_secrets(self) -> str:
        content = self.read_secrets_raw()
        if not content:
            return ""

        secrets_map = self.parse_env_content(content)
        env_lines = self.parse_env_lines(content)
        for line in env_lines:
            if line.type == "pair" and line.key is not None:
                line.key = line.key.upper()
                if line.key in secrets_map and secrets_map[line.key] != "":
                    line.value = self.MASK_VALUE
        return self._serialize_env_lines(env_lines)

    def parse_env_content(self, content: str) -> Dict[str, str]:
        env: Dict[str, str] = {}
        for binding in parse_stream(StringIO(content)):
            if binding.key and not binding.error:
                env[binding.key.upper()] = binding.value or ""
        return env

    def _parse_env_content(self, content: str) -> Dict[str, str]:
        return self.parse_env_content(content)

    def clear_cache(self):
        with self._lock:
            self._secrets_cache = None

    def parse_env_lines(self, content: str) -> List[EnvLine]:
        lines: List[EnvLine] = []
        for binding in parse_stream(StringIO(content)):
            orig = getattr(binding, "original", None)
            raw = getattr(orig, "string", "") if orig is not None else ""
            if binding.key and not binding.error:
                line_text = raw.rstrip("\n")
                if "=" in line_text:
                    left, right = line_text.split("=", 1)
                    key_part = left
                else:
                    key_part = binding.key
                    right = ""
                in_single = False
                in_double = False
                esc = False
                comment_index = None
                for idx, ch in enumerate(right):
                    if esc:
                        esc = False
                        continue
                    if ch == "\\":
                        esc = True
                        continue
                    if ch == "'" and not in_double:
                        in_single = not in_single
                        continue
                    if ch == '"' and not in_single:
                        in_double = not in_double
                        continue
                    if ch == "#" and not in_single and not in_double:
                        comment_index = idx
                        break
                inline_comment = None
                if comment_index is not None:
                    inline_comment = right[comment_index:]
                lines.append(
                    EnvLine(
                        raw=line_text,
                        type="pair",
                        key=binding.key,
                        value=binding.value or "",
                        key_part=key_part,
                        inline_comment=inline_comment,
                    )
                )
            else:
                raw_line = raw.rstrip("\n")
                if raw_line.strip() == "":
                    lines.append(EnvLine(raw=raw_line, type="blank"))
                elif raw_line.lstrip().startswith("#"):
                    lines.append(EnvLine(raw=raw_line, type="comment"))
                else:
                    lines.append(EnvLine(raw=raw_line, type="other"))
        return lines

    def _serialize_env_lines(
        self,
        lines: List[EnvLine],
        with_values=True,
        with_comments=True,
        with_blank=True,
        with_other=True,
        key_delimiter="",
        key_formatter: Optional[Callable[[str], str]] = None,
    ) -> str:
        output: List[str] = []
        for line in lines:
            if line.type == "pair" and line.key is not None:
                left_raw = line.key_part if line.key_part is not None else line.key
                left = left_raw.upper()
                value = line.value if line.value is not None else ""
                comment = line.inline_comment or ""
                formatted_key = (
                    key_formatter(left) if key_formatter else f"{key_delimiter}{left}{key_delimiter}"
                )
                value_part = f'="{value}"' if with_values else ""
                comment_part = f" {comment}" if with_comments and comment else ""
                output.append(f"{formatted_key}{value_part}{comment_part}")
            elif line.type == "blank" and with_blank:
                output.append(line.raw)
            elif line.type == "comment" and with_comments:
                output.append(line.raw)
            elif line.type == "other" and with_other:
                output.append(line.raw)
        return "\n".join(output)

    def _merge_env(self, existing_text: str, submitted_text: str) -> List[EnvLine]:
        existing_lines = self.parse_env_lines(existing_text)
        submitted_lines = self.parse_env_lines(submitted_text)

        existing_pairs: Dict[str, EnvLine] = {
            line.key: line
            for line in existing_lines
            if line.type == "pair" and line.key is not None
        }

        merged: List[EnvLine] = []
        for sub in submitted_lines:
            if sub.type != "pair" or sub.key is None:
                merged.append(sub)
                continue

            key = sub.key
            submitted_val = sub.value or ""

            if key in existing_pairs and submitted_val == self.MASK_VALUE:
                existing_val = existing_pairs[key].value or ""
                merged.append(
                    EnvLine(
                        raw=f"{(sub.key_part or key)}={existing_val}",
                        type="pair",
                        key=key,
                        value=existing_val,
                        key_part=sub.key_part or key,
                        inline_comment=sub.inline_comment,
                    )
                )
            elif key not in existing_pairs and submitted_val == self.MASK_VALUE:
                continue
            else:
                merged.append(sub)

        return merged
