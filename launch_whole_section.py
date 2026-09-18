from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tomllib
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("section_id")
    parser.add_argument("--yes", action="store_true")
    return parser.parse_args()


def _reference_provenance(receipt: dict) -> tuple[str, bool]:
    matches = [item for item in receipt["packages"] if item["project_id"] == "reference:sparrow"]
    if len(matches) != 1:
        raise ValueError("active environment receipt does not identify the SPArrOW reference")
    return matches[0]["commit"], matches[0]["dirty"]


def main() -> None:
    args = _parse_args()
    source_root = Path(__file__).parent.resolve()
    config_source = source_root / "whole_section.toml"
    entrypoint_source = source_root / "whole_section.py"
    config = tomllib.loads(config_source.read_text(encoding="utf-8"))
    sections = {section["section_id"]: section for section in config["sections"]}
    if args.section_id not in sections:
        raise ValueError(f"unknown section identity: {args.section_id}")
    workspace_root_value = os.environ.get("IOM_WORKSPACE_ROOT")
    active_receipt_value = os.environ.get("IOM_ACTIVE_SHELL_RECEIPT")
    if not workspace_root_value or not active_receipt_value:
        raise RuntimeError("launch must run in an active managed Iom author environment")
    workspace_root = Path(workspace_root_value).resolve()
    active_receipt = Path(active_receipt_value).resolve()
    receipt_document = json.loads(active_receipt.read_text(encoding="utf-8"))
    reference_commit, dirty = _reference_provenance(receipt_document)
    if dirty:
        raise RuntimeError("commit the reference integration before launching exact execution")
    python_executable = Path(receipt_document["python"]["executable"])
    python_root = os.environ.get("EBROOTPYTHON")
    library_path = os.environ.get("LD_LIBRARY_PATH")
    loaded_modules = os.environ.get("LOADEDMODULES", "").split(":")
    if (
        not python_root
        or Path(python_root) != python_executable.parents[1]
        or config["execution"]["python_module"] not in loaded_modules
        or not library_path
    ):
        raise RuntimeError("launch requires the Python module recorded by the active receipt")

    section = sections[args.section_id]
    output = workspace_root / section["output"]
    output_receipt = workspace_root / section["receipt"]
    if output.exists() or output_receipt.exists():
        raise FileExistsError("whole-section result already exists")
    control = output.parent / "execution" / args.section_id
    if control.exists():
        raise FileExistsError(f"run-control directory already exists: {control}")
    if not args.yes:
        print(json.dumps({"section": args.section_id, "resources": config["resources"]}, indent=2))
        print("Re-run with --yes to stage and submit the exact managed execution.")
        return

    control.mkdir(parents=True)
    entrypoint = control / "whole_section.py"
    config_path = control / "whole_section.toml"
    bootstrap_receipt = control / "environment_receipt.json"
    shutil.copyfile(entrypoint_source, entrypoint)
    shutil.copyfile(config_source, config_path)
    shutil.copyfile(active_receipt, bootstrap_receipt)
    stage_receipt = {
        "reference_commit": reference_commit,
        "entrypoint_sha256": _sha256(entrypoint),
        "config_sha256": _sha256(config_path),
        "environment_receipt_sha256": _sha256(bootstrap_receipt),
        "python_runtime": {
            "executable": str(python_executable),
            "module": config["execution"]["python_module"],
            "ld_library_path_sha256": hashlib.sha256(library_path.encode()).hexdigest(),
        },
    }
    (control / "stage_receipt.json").write_text(
        json.dumps(stage_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    iom = shutil.which("iom")
    uv = shutil.which("uv")
    if iom is None or uv is None:
        raise RuntimeError("iom or uv executable is unavailable")
    resources = config["resources"]
    slurm = config["slurm"]
    command = [
        "env",
        f"HOME={Path.home()}",
        f"USER={os.environ['USER']}",
        f"PATH={Path(uv).parent}:/usr/local/bin:/usr/bin:/bin",
        f"IOM_PYTHON={python_executable}",
        f"LD_LIBRARY_PATH={library_path}",
        f"IOM_WORKSPACE_ROOT={workspace_root}",
        f"IOM_AUTHOR_WORKTREE_ID={os.environ['IOM_AUTHOR_WORKTREE_ID']}",
        iom,
        "env",
        "exec-receipt",
        str(bootstrap_receipt),
        "python",
        str(entrypoint),
        "--config",
        str(config_path),
        "--section-id",
        args.section_id,
        "--workspace-root",
        str(workspace_root),
        "--reference-commit",
        reference_commit,
        "--environment-receipt",
        str(bootstrap_receipt),
    ]
    hours, remainder = divmod(resources["walltime_seconds"], 3600)
    minutes, seconds = divmod(remainder, 60)
    submit = [
        "sbatch",
        "--parsable",
        "--export=NONE",
        f"--account={slurm['account']}",
        f"--partition={slurm['partition']}",
        f"--cpus-per-task={resources['cpus']}",
        f"--mem={resources['memory_gib']}G",
        f"--time={hours:02d}:{minutes:02d}:{seconds:02d}",
        f"--gres=gpu:{resources['gpus']}",
        f"--job-name=sparrow-{args.section_id.split('-')[2]}",
        f"--output={control}/slurm-%j.log",
        f"--chdir={control}",
        f"--wrap={shlex.join(command)}",
    ]
    try:
        completed = subprocess.run(submit, check=True, capture_output=True, text=True, timeout=60)
    except Exception:
        shutil.rmtree(control)
        raise
    job_id = completed.stdout.strip().split(";", 1)[0]
    (control / "job_id.txt").write_text(job_id + "\n", encoding="utf-8")
    print(json.dumps({"job_id": job_id, "section": args.section_id}, sort_keys=True))


if __name__ == "__main__":
    main()
