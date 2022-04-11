from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent

with open(PACKAGE_DIR / "VERSION") as version_file:
    __version__ = version_file.read().strip()
