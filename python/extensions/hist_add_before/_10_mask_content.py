from python.helpers.extension import Extension


class MaskHistoryContent(Extension):
    async def execute(self, **kwargs):
        content_data = kwargs.get("content_data")
        if not content_data:
            return

        try:
            from python.helpers.secrets import SecretsManager

            secrets_mgr = SecretsManager.get_instance()
            content_data["content"] = self._mask_content(
                content_data["content"], secrets_mgr
            )
        except Exception:
            pass

    def _mask_content(self, content, secrets_mgr):
        if isinstance(content, str):
            return secrets_mgr.mask_values(content)
        if isinstance(content, list):
            return [self._mask_content(item, secrets_mgr) for item in content]
        if isinstance(content, dict):
            return {k: self._mask_content(v, secrets_mgr) for k, v in content.items()}
        return content
