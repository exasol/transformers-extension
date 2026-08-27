"""Benchmark model archive creation and extraction with tarfile and fastar.

Example::

    poetry run python benchmark/tarfile_fastar.py \
        --model bert-tiny=/path/to/bert-tiny \
        --model qwen3-4b=/path/to/qwen3-4b \
        --output benchmark-results.json \
        --work-dir /path/to/persistent/benchmark-work

The source directories are never modified. The work directory contains the
archives and temporary extraction directories and should be on persistent
storage when benchmarking large models.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
)

import fastar

Backend = str
CreateArchive = Callable[[Path, Path, bool], None]


@dataclass(frozen=True)
class BenchmarkCase:
    """Configuration for one backend and archive-format benchmark."""

    model: str
    source: Path
    backend: Backend
    create: CreateArchive
    extract: Callable[[Path, Path], None]
    work_dir: Path
    repetitions: int
    compressed: bool


def _manifest(directory: Path) -> dict[str, int]:
    return {
        str(path.relative_to(directory)): path.stat().st_size
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _create_with_tarfile(source: Path, archive: Path, compressed: bool) -> None:
    mode = "w:gz" if compressed else "w"
    with tarfile.open(archive, mode) as output:
        for path in sorted(source.iterdir()):
            output.add(path, arcname=path.name)


def _create_with_fastar(source: Path, archive: Path, compressed: bool) -> None:
    mode = "w:gz" if compressed else "w"
    with fastar.open(archive, mode) as output:  # pylint: disable=no-member
        for path in sorted(source.iterdir()):
            output.append(path=path, arcname=path.name)


def _extract_with_tarfile(archive: Path, destination: Path) -> None:
    mode = "r:gz" if archive.name.endswith(".gz") else "r"
    with tarfile.open(archive, mode) as source:
        source.extractall(destination)


def _extract_with_fastar(archive: Path, destination: Path) -> None:
    mode = "r:gz" if archive.name.endswith(".gz") else "r"
    with fastar.open(archive, mode) as source:  # pylint: disable=no-member
        source.unpack(to=destination)


def _measure(function: Callable[[], None]) -> float:
    start = time.perf_counter()
    function()
    return (time.perf_counter() - start) * 1000


def _benchmark(case: BenchmarkCase) -> dict[str, Any]:
    suffix = ".tar.gz" if case.compressed else ".tar"
    archive = case.work_dir / f"{case.model}-{case.backend}{suffix}"
    manifest = _manifest(case.source)
    creation_samples: list[float] = []
    extraction_samples: list[float] = []

    for repetition in range(case.repetitions):
        archive.unlink(missing_ok=True)
        creation_samples.append(
            _measure(lambda: case.create(case.source, archive, case.compressed))
        )
        destination = (
            case.work_dir / f"{case.model}-{case.backend}-extract-{repetition}"
        )
        shutil.rmtree(destination, ignore_errors=True)
        destination.mkdir()
        extraction_samples.append(
            _measure(lambda destination=destination: case.extract(archive, destination))
        )
        if _manifest(destination) != manifest:
            raise RuntimeError(f"Extraction validation failed for {archive}")
        shutil.rmtree(destination)

    creation_samples.sort()
    extraction_samples.sort()
    return {
        "model": case.model,
        "backend": case.backend,
        "compressed": case.compressed,
        "files": len(manifest),
        "source_gib": sum(manifest.values()) / 2**30,
        "archive_gib": archive.stat().st_size / 2**30,
        "repetitions": case.repetitions,
        "creation_median_ms": creation_samples[len(creation_samples) // 2],
        "extraction_median_ms": extraction_samples[len(extraction_samples) // 2],
        "creation_samples_ms": creation_samples,
        "extraction_samples_ms": extraction_samples,
    }


def _parse_model(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("model must have the form NAME=PATH")
    source = Path(path).expanduser()
    if not source.is_dir():
        raise argparse.ArgumentTypeError(f"model directory does not exist: {source}")
    return name, source


def main() -> None:
    """Run the configured archive benchmarks and write JSON results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        type=_parse_model,
        required=True,
        help="model source in the form NAME=PATH; may be repeated",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--uncompressed-only",
        action="store_true",
        help="benchmark only uncompressed tar archives",
    )
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    formats = [False] if args.uncompressed_only else [True, False]
    results = []
    for model, source in args.model:
        for compressed in formats:
            for backend, create, extract in (
                ("tarfile", _create_with_tarfile, _extract_with_tarfile),
                ("fastar", _create_with_fastar, _extract_with_fastar),
            ):
                results.append(
                    _benchmark(
                        BenchmarkCase(
                            model=model,
                            source=source,
                            backend=backend,
                            create=create,
                            extract=extract,
                            work_dir=args.work_dir,
                            repetitions=args.repetitions,
                            compressed=compressed,
                        )
                    )
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
