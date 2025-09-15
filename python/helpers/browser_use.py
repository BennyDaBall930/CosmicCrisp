import os
from python.helpers import dotenv, files

# Ensure telemetry is disabled both in .env and process env
dotenv.save_dotenv_value("ANONYMIZED_TELEMETRY", "false")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")

# Force browser-use config into project tmp to avoid home perms issues
_cfg_dir = files.get_abs_path("tmp/browseruse")
os.makedirs(os.path.join(_cfg_dir, "profiles"), exist_ok=True)
os.environ.setdefault("BROWSER_USE_CONFIG_DIR", _cfg_dir)
# Also set XDG fallback root for completeness
os.environ.setdefault("XDG_CONFIG_HOME", files.get_abs_path("tmp/xdg"))

import browser_use
import browser_use.utils
