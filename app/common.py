from enum import Enum
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"


class OriginKey(str, Enum):
    arxiv = "arxiv"
    github = "github"
