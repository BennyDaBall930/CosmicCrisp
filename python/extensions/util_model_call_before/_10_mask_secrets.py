from python.helpers.extension import Extension
from python.helpers.secrets import SecretsManager


class MaskToolSecrets(Extension):
    async def execute(self, **kwargs):
        call_data: dict = kwargs.get("call_data", {})  # type: ignore[assignment]
        secrets_mgr = SecretsManager.get_instance()

        system = call_data.get("system")
        if system:
            call_data["system"] = secrets_mgr.mask_values(system)

        message = call_data.get("message")
        if message:
            call_data["message"] = secrets_mgr.mask_values(message)
