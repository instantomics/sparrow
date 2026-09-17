from __future__ import annotations

import argparse
import hashlib
import os
import re
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from wheel.wheelfile import WheelFile

_UPSTREAM_URL = (
    "https://files.pythonhosted.org/packages/55/24/"
    "4c327f57a6dc3f672b956cc47a46a966475b9ddbf8b675bc0b1603e0f327/"
    "cellpose-3.1.1.3-py3-none-any.whl"
)
_UPSTREAM_SHA256 = "0c309f306a8356ba2779054dfd8a1b4a88ea7d6c3388a6fdac99b91b21ce7ff2"
_UPSTREAM_SIZE = 226_352
_UPSTREAM_VERSION = "3.1.1.3"
_PATCHED_VERSION = "3.1.1.3.post1"


def _download(path: Path) -> None:
    request = urllib.request.Request(_UPSTREAM_URL, headers={"User-Agent": "iomix-sparrow-build"})
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(request, timeout=60) as response, path.open("wb") as stream:
        while chunk := response.read(1024 * 1024):
            size += len(chunk)
            if size > _UPSTREAM_SIZE:
                raise ValueError("Cellpose wheel download exceeds its declared size")
            digest.update(chunk)
            stream.write(chunk)
    if size != _UPSTREAM_SIZE or digest.hexdigest() != _UPSTREAM_SHA256:
        raise ValueError("Cellpose wheel download does not match its declared identity")


def build(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / f"cellpose-{_PATCHED_VERSION}-py3-none-any.whl"
    with tempfile.TemporaryDirectory(prefix="iomix-cellpose-build-") as temporary:
        root = Path(temporary)
        upstream = root / f"cellpose-{_UPSTREAM_VERSION}-py3-none-any.whl"
        unpacked = root / "unpacked"
        _download(upstream)
        with zipfile.ZipFile(upstream) as archive:
            archive.extractall(unpacked)

        old_info = unpacked / f"cellpose-{_UPSTREAM_VERSION}.dist-info"
        new_info = unpacked / f"cellpose-{_PATCHED_VERSION}.dist-info"
        old_info.rename(new_info)
        metadata_path = new_info / "METADATA"
        metadata = metadata_path.read_text(encoding="utf-8")
        metadata, version_count = re.subn(
            rf"(?m)^Version: {re.escape(_UPSTREAM_VERSION)}$",
            f"Version: {_PATCHED_VERSION}",
            metadata,
        )
        metadata, numpy_count = re.subn(
            r"(?m)^Requires-Dist: numpy<2\.1,>=1\.20\.0$",
            "Requires-Dist: numpy>=2.2.0,<2.3",
            metadata,
        )
        if version_count != 1 or numpy_count != 1:
            raise ValueError("upstream Cellpose metadata does not match the reviewed patch")
        metadata_path.write_text(metadata, encoding="utf-8")
        (new_info / "RECORD").unlink()

        temporary_output = root / output.name
        with WheelFile(temporary_output, "w") as wheel:
            for path in sorted(unpacked.rglob("*")):
                if path.is_file():
                    wheel.write(path, path.relative_to(unpacked).as_posix())
        os.replace(temporary_output, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(build(args.destination))


if __name__ == "__main__":
    main()
