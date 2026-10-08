"""Copy the license files of the bundled third-party packages into a directory.

Usage: python collect_licenses.py OUTPUT_DIR
Reads them from the installed distributions, so they match the exact versions
that went into the build.
"""
import shutil
import sys
from importlib.metadata import distribution
from pathlib import Path

# distribution name -> license files/dirs inside its .dist-info directory
SOURCES = {
    "opencv-python-headless": ["LICENSE.txt", "LICENSE-3RD-PARTY.txt"],
    "numpy": ["licenses"],
    "pyinstaller": ["licenses"],
}

out = Path(sys.argv[1])
for name, entries in SOURCES.items():
    dist = distribution(name)
    info = Path(str(dist._path))  # the .dist-info directory
    target = out / name
    for entry in entries:
        src = info / entry
        if not src.exists():
            sys.exit(f"{name}: {entry} not found in {info}")
        if src.is_dir():
            shutil.copytree(src, target / entry, dirs_exist_ok=True)
        else:
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target / entry)
    print(f"{name} {dist.version}: licenses copied to {target}")
