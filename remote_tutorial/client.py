"""Local-only planning/packaging and explicit SSH/Slurm commands."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

from .common import DEFAULTS, STAGES, canonical, recipe, resources, sha256, validate_config, work_root

REPO = Path(__file__).resolve().parents[1]
DIRECTORIES = {
    "configs",
    "docs",
    "examples",
    "geniac_baseline",
    "licenses",
    "patches",
    "remote_tutorial",
    "scripts",
    "tests",
}
FILES = {
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
    "requirements-train.txt",
}
PRIVATE_PARTS = {".git", ".aws", ".codex", ".agents", ".venv", ".venv-serve", ".venv-libero", ".cache", "__pycache__"}


def run(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=True, **kwargs)


def git(repo: Path, *args: str) -> str:
    return run(["git", "-C", str(repo), *args], stdout=subprocess.PIPE, text=True).stdout.strip()


def included(name: str) -> bool:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        return False
    if any(p in PRIVATE_PARTS or p.startswith(".env") for p in path.parts):
        return False
    if path.parts[0] == "configs" and (path.name.startswith(("my_", "train_my_"))):
        return False
    return name in FILES or path.parts[0] in DIRECTORIES


def source_files(repo: Path) -> list[str]:
    # Read current files, including uncommitted edits and new implementation files; do not archive HEAD alone.
    names = (
        run(
            ["git", "-C", str(repo), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            stdout=subprocess.PIPE,
        )
        .stdout.decode()
        .split("\0")
    )
    result = []
    for name in sorted(set(filter(included, names))):
        path = repo / name
        if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != repo.parent):
            raise ValueError(f"refusing source symlink: {name}")
        if path.is_file():  # deleted tracked files are intentionally absent
            result.append(name)
    return result


def prepare_source(repo: Path, config: dict, source: str | None = None) -> Path:
    cache = repo / ".remote-build/source"
    cache.mkdir(parents=True, exist_ok=True)
    pin = recipe(repo)["commit"]
    checkout = cache / "capx"
    supplied = source or config["capx_source"]
    if not checkout.exists():
        origin = str(Path(supplied).expanduser().resolve()) if supplied else config["capx_url"]
        run(
            ["git", "clone", "--no-hardlinks", "--no-checkout", origin, str(checkout)],
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
        run(["git", "-C", str(checkout), "checkout", "--detach", pin])
    if git(checkout, "rev-parse", "HEAD") != pin:
        raise ValueError("managed source cache has a different HEAD; select a fresh source cache")
    if git(checkout, "status", "--porcelain"):
        raise ValueError("managed source cache was edited; refusing to bundle it")
    bundle = cache / f"capx-{pin}.bundle"
    if not bundle.exists():
        run(["git", "-C", str(checkout), "bundle", "create", str(bundle), "HEAD"])
    refs = run(["git", "bundle", "list-heads", str(bundle)], stdout=subprocess.PIPE, text=True).stdout
    if f"{pin} HEAD" not in refs.splitlines():
        raise ValueError("source bundle does not contain the expected pin")
    return bundle


def build_payload(repo: Path, config: dict, bundle: Path) -> dict:
    required = {
        f"remote_tutorial/envs/{name}/{filename}"
        for name in ("base", "serve")
        for filename in ("pyproject.toml", "uv.lock")
    }
    for name in sorted(required):
        if not (repo / name).is_file():
            raise ValueError(f"missing uv project file: {name}")
    names = source_files(repo)
    if missing := required - set(names):
        raise ValueError(f"uv project files are excluded from the payload; check Git ignore rules: {sorted(missing)}")
    entries = {
        name: {"sha256": sha256(repo / name), "mode": 0o750 if os.access(repo / name, os.X_OK) else 0o640}
        for name in names
    }
    entries["_tutorial/capx.bundle"] = {"sha256": sha256(bundle), "mode": 0o640}
    remote_config = {**config, "capx_source": None}
    identity = {
        "schema": 1,
        "config": remote_config,
        "recipe": recipe(repo),
        "files": entries,
        "git_head": git(repo, "rev-parse", "HEAD"),
    }
    release = hashlib.sha256(canonical(identity)).hexdigest()[:24]
    manifest = {**identity, "release": release}
    directory = repo / ".remote-build" / release
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "payload.tar"
    with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT) as tar:
        for name, info in entries.items():
            path = bundle if name == "_tutorial/capx.bundle" else repo / name
            member = tarfile.TarInfo(name)
            member.size, member.mode = path.stat().st_size, info["mode"]
            with path.open("rb") as stream:
                tar.addfile(member, stream)
        data = canonical(manifest)
        member = tarfile.TarInfo("_tutorial/manifest.json")
        member.size, member.mode = len(data), 0o600
        tar.addfile(member, io.BytesIO(data))
    state = {"release": release, "archive": str(archive), "sha256": sha256(archive), "jobs": {}, "last_job": None}
    previous = directory / "state.json"
    if previous.exists():
        old = json.loads(previous.read_text())
        if old["sha256"] != state["sha256"]:
            raise ValueError("same release identity produced a different archive")
        state["jobs"] = old["jobs"]
        state["last_job"] = old.get("last_job")
    previous.write_bytes(canonical(state))
    (repo / ".remote-build/latest.json").write_bytes(canonical(state))
    (directory / "manifest.json").write_bytes(canonical(manifest))
    return state


def ssh(config: dict, command: list[str], **kwargs) -> str:
    args = [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "UpdateHostKeys=no",
        config["host"],
        shlex.join(command),
    ]
    return run(args, stdout=subprocess.PIPE, **kwargs).stdout.decode().strip()


def transfer_command(c: dict, state: dict) -> list[str]:
    # Python receives the archive on a compute node; the login node only relays SSH/Slurm streams.
    with tarfile.open(state["archive"]) as archive, archive.extractfile("remote_tutorial/receive.py") as source:
        receiver = source.read().decode()
    return [
        "srun",
        "--quiet",
        *resources(c, "setup"),
        "python3",
        "-c",
        receiver,
        canonical(c).decode().strip(),
        state["release"],
        state["sha256"],
    ]


def batch_command(c: dict, state: dict, stage: str) -> list[str]:
    release = state["release"]
    root = work_root(c)
    command = [
        "sbatch",
        "--parsable",
        *resources(c, stage),
        f"--job-name=tutorial-{stage}",
        f"--chdir={root / 'releases' / release}",
        f"--output={root / 'logs' / (stage + '-%j.out')}",
        f"--error={root / 'logs' / (stage + '-%j.err')}",
    ]
    if state.get("last_job"):
        # Serialize this deployment's stages, including resubmissions, to protect its data and ports.
        command += [f"--dependency=afterok:{state['last_job']['id']}", "--kill-on-invalid-dep=yes"]
    command += [str(root / "releases" / release / "_tutorial/jobs" / f"{stage}.sh")]
    return command


def describe(c: dict, state: dict) -> dict:
    return {
        "network_operations_performed": False,
        "release": state["release"],
        "remote_workspace": str(work_root(c)),
        "archive": state["archive"],
        "archive_bytes": Path(state["archive"]).stat().st_size,
        "archive_sha256": state["sha256"],
        "transfer_on_compute": shlex.join(transfer_command(c, state)),
        "stages": {
            stage: shlex.join(batch_command(c, {**state, "jobs": {}, "last_job": None}, stage)) for stage in STAGES
        },
        "note": "plan only; setup/check/submit perform remote operations only when explicitly invoked",
    }


def save_state(repo: Path, state: dict) -> None:
    for path in (repo / ".remote-build/latest.json", Path(state["archive"]).parent / "state.json"):
        path.write_bytes(canonical(state))


def main(argv: list[str] | None = None, repo: Path = REPO) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=repo / ".remote.local.json")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="create an ignored local configuration; no SSH")
    for name in ("host", "account", "partition", "shared_root", "remote_user", "models_root", "capx_source"):
        init.add_argument("--" + name.replace("_", "-"))
    init.add_argument("--port-offset", type=int)
    prepare = sub.add_parser("prepare", help="bundle the current workspace and pinned CaP-X source locally")
    prepare.add_argument("--source", help="existing local CaP-X checkout; otherwise clone using local Git credentials")
    sub.add_parser("plan", help="write a local deployment plan; never connects to the cluster")
    setup = sub.add_parser("setup", help="transfer on a CPU compute node, then submit environment construction")
    setup.add_argument("--retry", action="store_true", help="retry only after the last setup job failed")
    check = sub.add_parser("check", help="submit a GPU/import/render validation job; starts no tutorial trial")
    check.add_argument("--retry", action="store_true")
    submit = sub.add_parser("submit", help="explicitly start a tutorial stage")
    submit.add_argument("stage", choices=STAGES[2:])
    submit.add_argument("--retry", action="store_true")
    sub.add_parser("status", help="read accounting for the submitted jobs")
    logs = sub.add_parser("logs", help="read the latest job's bounded log tail")
    logs.add_argument("--lines", type=int, default=80)
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            if args.config.exists():
                raise ValueError(f"refusing to overwrite {args.config}")
            cfg = {**DEFAULTS, **{k: getattr(args, k) for k in DEFAULTS if getattr(args, k, None) is not None}}
            args.config.parent.mkdir(parents=True, exist_ok=True)
            args.config.write_bytes(canonical(cfg))
            args.config.chmod(0o600)
            print(f"created {args.config}; fill placeholders and select a team-coordinated port_offset")
            return 0
        c = validate_config(json.loads(args.config.read_text()))
        if args.command == "prepare":
            state = build_payload(repo, c, prepare_source(repo, c, args.source))
            print(f"prepared {state['release']}: {state['archive']}")
            return 0
        state = json.loads((repo / ".remote-build/latest.json").read_text())
        archive_manifest = json.loads((Path(state["archive"]).parent / "manifest.json").read_text())
        if {**c, "capx_source": None} != archive_manifest["config"]:
            raise ValueError("configuration changed; run prepare again")
        if args.command == "plan":
            plan = describe(c, state)
            path = Path(state["archive"]).parent / "plan.json"
            path.write_bytes(canonical(plan))
            print(json.dumps({k: v for k, v in plan.items() if k != "transfer_on_compute"}, indent=2))
            print(f"full transfer command and stage commands: {path}")
            return 0
        if args.command in {"status", "logs"}:
            if not state["jobs"]:
                raise ValueError("no jobs submitted for this deployment")
            if args.command == "status":
                print(
                    ssh(
                        c,
                        [
                            "sacct",
                            "-j",
                            ",".join(state["jobs"].values()),
                            "--format=JobID,State,Elapsed,ExitCode",
                            "-P",
                        ],
                    )
                )
            else:
                if not 1 <= args.lines <= 1000:
                    raise ValueError("--lines must be between 1 and 1000")
                stage, job = state["last_job"]["stage"], state["last_job"]["id"]
                paths = [str(work_root(c) / "logs" / f"{stage}-{job}.{suffix}") for suffix in ("out", "err")]
                print(ssh(c, ["tail", "-n", str(args.lines), "--", *paths]))
            return 0
        stage = args.stage if args.command == "submit" else args.command
        if stage != "setup" and "setup" not in state["jobs"]:
            raise ValueError("submit setup for this deployment first")
        if stage in state["jobs"]:
            if not args.retry:
                print(f"{stage} already submitted: {state['jobs'][stage]}; use status/logs or --retry after failure")
                return 0
            if state["last_job"] != {"stage": stage, "id": state["jobs"][stage]}:
                raise ValueError("retry the last submitted stage before queuing dependent stages")
            status = ssh(c, ["sacct", "-X", "-n", "-j", state["jobs"][stage], "--format=State", "-P"])
            if not re.fullmatch(
                r"(?:FAILED|CANCELLED(?: by [0-9]+)?|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED)\|?", status
            ):
                raise ValueError(f"job is not a failed terminal job: {status!r}")
            state["last_job"] = None
        if stage == "setup":
            if sha256(Path(state["archive"])) != state["sha256"]:
                raise ValueError("payload was modified after preparation")
            with Path(state["archive"]).open("rb") as stream:
                print(ssh(c, transfer_command(c, state), stdin=stream))
        output = ssh(c, batch_command(c, state, stage))
        job = output.split(";")[0].strip()
        if not re.fullmatch(r"[0-9]+", job):
            raise ValueError(f"unexpected sbatch response: {output!r}")
        state["jobs"][stage] = job
        state["last_job"] = {"stage": stage, "id": job}
        save_state(repo, state)
        print(f"submitted {stage}: {job}; use status or logs")
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
