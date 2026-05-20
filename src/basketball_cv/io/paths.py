from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_output_path(*subpaths: str | Path) -> Path:
    new_path = OUTPUTS_DIR.joinpath(*[str(part) for part in subpaths])
    return ensure_directory(new_path)
