from fer_pytorch.config import CFG

with open(CFG.PACKAGE_DIR / "VERSION") as version_file:
    __version__ = version_file.read().strip()
