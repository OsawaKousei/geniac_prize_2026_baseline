"""Configuration and command construction; importing this module performs no I/O."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from pathlib import Path, PurePosixPath

MODELS = {
    "qwen": ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"),
    "molmo": ("allenai/Molmo2-8B", "e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"),
    "sam3": ("facebook/sam3", "3c879f39826c281e95690f02c7821c4de09afae7"),
}
DEFAULTS = {
    "host": "<ssh-alias>",
    "account": "<slurm-account>",
    "partition": "<compute-partition>",
    "shared_root": "/<shared-storage>",
    "remote_user": "<cluster-user>",
    "project": "baseline-tutorial",
    "experiment": "smoke",
    "models_root": None,
    "capx_source": None,
    "capx_url": "https://github.com/matsuolab/geniac_prize_2026_code.git",
    "port_offset": None,
    "task": "cube_restack",
    "collect_seeds": ["301-302", "391-392"],
    "eval_seeds": ["401-402"],
    "sft_max_steps": 2,
    "setup_cpus": 4,
    "setup_mem": "16G",
    "setup_time": "01:00:00",
    "gpu_cpus": 8,
    "gpu_mem": "128G",
    "sft_mem": "256G",
    "gpu_time": "04:00:00",
    "gpu_gres": "gpu:1",
}
STAGES = ("setup", "check", "collect", "convert", "eval-base", "sft", "eval-sft")


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_config(raw: dict) -> dict:
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown config fields: {sorted(unknown)}")
    c = {**DEFAULTS, **raw}
    for name in ("host", "account", "partition", "remote_user", "project", "experiment", "task", "gpu_gres"):
        value = c[name]
        pattern = r"[A-Za-z0-9][A-Za-z0-9_.@:-]*" if name in {"host", "gpu_gres"} else r"[A-Za-z0-9][A-Za-z0-9_.-]*"
        if not isinstance(value, str) or not re.fullmatch(pattern, value):
            raise ValueError(f"configure {name}: expected a name without shell syntax or placeholders")
    if c["remote_user"] == "models" or c["project"] in {".", ".."}:
        raise ValueError("invalid personal workspace name")
    if not re.fullmatch(r"gpu(?::[A-Za-z0-9_.-]+)?:1", c["gpu_gres"]):
        raise ValueError("the initial tutorial supports exactly one GPU")
    for name in ("shared_root", "models_root"):
        value = c[name]
        if name == "models_root" and value is None:
            value = str(PurePosixPath(c["shared_root"]) / "models")
        if not isinstance(value, str) or any(x in value for x in ("<", ">", "\n", "\r", "\0")):
            raise ValueError(f"configure {name}: expected an absolute POSIX path")
        path = PurePosixPath(value)
        if not path.is_absolute() or ".." in path.parts or str(path) == "/":
            raise ValueError(f"unsafe {name}")
        c[name] = str(path)
    root = work_root(c)
    models = PurePosixPath(c["models_root"])
    if root == models or root in models.parents or models in root.parents:
        raise ValueError("personal workspace and shared models must not overlap")
    for name in ("setup_cpus", "gpu_cpus", "sft_max_steps"):
        if type(c[name]) is not int or c[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    off = c["port_offset"]
    if type(off) is not int or not 1 <= off <= 65535 - 8122:
        raise ValueError("set port_offset to a team-coordinated integer between 1 and 57413")
    for name in ("setup_mem", "gpu_mem", "sft_mem"):
        if not isinstance(c[name], str) or not re.fullmatch(r"[1-9][0-9]*[MG]", c[name]):
            raise ValueError(f"invalid {name}")
    for name in ("setup_time", "gpu_time"):
        if not isinstance(c[name], str) or not re.fullmatch(r"(?:[0-9]+-)?[0-9]{1,3}:[0-5][0-9]:[0-5][0-9]", c[name]):
            raise ValueError(f"invalid {name}")
    for name in ("collect_seeds", "eval_seeds"):
        if not isinstance(c[name], list) or not c[name]:
            raise ValueError(f"{name} must be a nonempty list of seed ranges")
        seen = set()
        for segment in c[name]:
            if not isinstance(segment, str) or not re.fullmatch(r"[0-9]+(?:-[0-9]+)?", segment):
                raise ValueError(f"invalid range in {name}")
            a, _, b = segment.partition("-")
            start, end = int(a), int(b or a)
            band = range(301, 401) if name == "collect_seeds" else range(401, 451)
            if start > end or start not in band or end not in band:
                raise ValueError(f"{name} is outside the Robosuite protocol")
            values = set(range(start, end + 1))
            if seen & values:
                raise ValueError(f"overlapping ranges in {name}")
            seen |= values
        if name == "collect_seeds" and not (any(s < 391 for s in seen) and any(s >= 391 for s in seen)):
            raise ValueError("collect_seeds must include both train and validation seeds")
    url = c["capx_url"]
    if not isinstance(url, str) or not (
        re.fullmatch(r"https://[A-Za-z0-9.-]+/[A-Za-z0-9_./-]+", url)
        or re.fullmatch(r"git@[A-Za-z0-9_.-]+:[A-Za-z0-9_./-]+", url)
    ):
        raise ValueError("capx_url must be an HTTPS or SSH Git URL without embedded credentials")
    if c["capx_source"] is not None and not isinstance(c["capx_source"], str):
        raise ValueError("capx_source must be a local checkout path or null")
    return c


def work_root(config: dict) -> PurePosixPath:
    return PurePosixPath(config["shared_root"]) / config["remote_user"] / config["project"]


def recipe(repo: Path) -> dict:
    """Read the pin and selected submodules from the existing setup script, without executing it."""
    text = (repo / "patches/geniac_prize/setup.sh").read_text()
    commit = re.search(r"^commit=([a-f0-9]{40})$", text, re.M)
    modules = re.search(r"^submodules=\(\n(.*?)\n\)", text, re.M | re.S)
    if not commit or not modules:
        raise ValueError("cannot read pinned CaP-X recipe from patches/geniac_prize/setup.sh")
    paths = modules[1].split()
    if not paths or any(not re.fullmatch(r"capx/third_party/[A-Za-z0-9_/-]+", p) for p in paths):
        raise ValueError("invalid submodule recipe")
    return {"commit": commit[1], "submodules": paths, "patches": ["geniac-prize.patch", "skill-library.patch"]}


def resources(c: dict, stage: str) -> list[str]:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage}")
    cpu = stage in {"setup", "convert"}
    args = [f"--partition={c['partition']}", f"--account={c['account']}", "--nodes=1", "--ntasks=1"]
    args += [
        f"--cpus-per-task={c['setup_cpus'] if cpu else c['gpu_cpus']}",
        f"--mem={c['setup_mem'] if cpu else (c['sft_mem'] if stage == 'sft' else c['gpu_mem'])}",
        f"--time={c['setup_time'] if cpu else c['gpu_time']}",
    ]
    if not cpu:
        args += [f"--gres={c['gpu_gres']}"]
    return args


def job_shell(root: str, release: str, stage: str) -> str:
    if not re.fullmatch(r"[a-f0-9]{24}", release) or stage not in STAGES:
        raise ValueError("invalid job identity")
    directory = str(PurePosixPath(root) / "releases" / release)
    # All generated commands use explicit interpreters and the selected source snapshot.
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "umask 077",
            ': "${SLURM_JOB_ID:?Submit through Slurm}"',
            f"cd {shlex.quote(directory)}",
            'export PYTHONPATH="$PWD"',
            f"exec {'python3' if stage == 'setup' else '.venv/bin/python'} -m remote_tutorial.runner {stage} .",
            "",
        ]
    )
