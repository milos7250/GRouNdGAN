import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))
__name__ = "groundgan"

os.environ["TORCHDYNAMO_VERBOSE"] = "0"
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_REPORT_CHOICES_STATS"] = "0"