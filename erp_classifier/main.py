# The main event loop
from pathlib import Path

import yaml

CONFIG = yaml.safe_load(open(Path("./configs/general_config.yaml").resolve()))
