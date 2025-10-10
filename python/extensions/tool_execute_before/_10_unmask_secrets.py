from python.helpers.extension import Extension
from python.helpers.secrets import SecretsManager


class UnmaskToolSecrets(Extension):
    async def execute(self, **kwargs):
        tool_args = kwargs.get("tool_args")
        if not tool_args:
            return

        secrets_mgr = SecretsManager.get_instance()
        for key, value in tool_args.items():
            if isinstance(value, str):
                tool_args[key] = secrets_mgr.replace_placeholders(value)
