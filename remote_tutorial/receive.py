"""Standalone archive receiver, executed by srun via python3 -c (stdin is the archive)."""

from __future__ import annotations

import hashlib
import json
import os
import pwd
import re
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


def receive(config: dict, release: str, digest: str, stream) -> Path:
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("archive reception must run in a Slurm compute job")
    if config["remote_user"] != pwd.getpwuid(os.getuid()).pw_name:
        raise ValueError("remote SSH user does not match configured workspace owner")
    if not re.fullmatch(r"[a-f0-9]{24}", release) or not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("invalid deployment identity")
    shared = Path(config["shared_root"])
    if not shared.is_absolute() or shared == Path("/") or ".." in shared.parts:
        raise ValueError("unsafe shared storage path")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", config["project"]):
        raise ValueError("unsafe project name")
    root = shared / config["remote_user"] / config["project"]
    models = Path(config["models_root"])
    if root == models or root in models.parents or models in root.parents:
        raise ValueError("workspace overlaps shared models")
    if root == shared or ".." in root.parts or "/" in config["project"] or config["remote_user"] == "models":
        raise ValueError("unsafe personal workspace")
    for path in (root, *root.parents):
        if path.is_symlink():
            raise ValueError(f"refusing symlink in workspace path: {path}")
    personal = root.parent
    personal.mkdir(mode=0o700, exist_ok=True)
    if personal.stat().st_uid != os.getuid():
        raise ValueError("personal workspace parent belongs to another user")
    root.mkdir(mode=0o700, exist_ok=True)
    if root.stat().st_uid != os.getuid():
        raise ValueError("workspace belongs to another user")
    for name in ("releases", "logs", "runs", "cache", "tools"):
        path = root / name
        if path.is_symlink():
            raise ValueError(f"refusing managed directory symlink: {path}")
        path.mkdir(mode=0o700, exist_ok=True)
        if path.stat().st_uid != os.getuid():
            raise ValueError("managed directory belongs to another user")
    temp = Path(tempfile.mkdtemp(prefix=".receive-", dir=root))
    try:
        archive = temp / "payload.tar"
        h = hashlib.sha256()
        with archive.open("xb") as output:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
                output.write(block)
        if h.hexdigest() != digest:
            raise ValueError("archive SHA256 mismatch")
        unpacked = temp / "release"
        unpacked.mkdir()
        seen = set()
        with tarfile.open(archive) as tar:
            for member in tar:
                name = PurePosixPath(member.name)
                if name.is_absolute() or ".." in name.parts or not member.isfile() or member.name in seen:
                    raise ValueError(f"unsafe archive member: {member.name}")
                seen.add(member.name)
                destination = unpacked / member.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, destination.open("xb") as target:
                    shutil.copyfileobj(source, target)
                destination.chmod(0o750 if member.mode & 0o100 else 0o640)
        manifest = json.loads((unpacked / "_tutorial/manifest.json").read_text())
        if manifest["release"] != release or manifest["config"] != {**config, "capx_source": None}:
            raise ValueError("archive configuration mismatch")
        if seen != set(manifest["files"]) | {"_tutorial/manifest.json"}:
            raise ValueError("archive file list mismatch")
        for name, expected in manifest["files"].items():
            file_hash = hashlib.sha256()
            with (unpacked / name).open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    file_hash.update(block)
            if file_hash.hexdigest() != expected["sha256"]:
                raise ValueError(f"file checksum mismatch: {name}")
        sys.path.insert(0, str(unpacked))
        from remote_tutorial.common import STAGES, job_shell

        jobs = unpacked / "_tutorial/jobs"
        jobs.mkdir()
        for stage in STAGES:
            (jobs / f"{stage}.sh").write_text(job_shell(str(root), release, stage))
        (unpacked / "_tutorial/received.sha256").write_text(digest + "\n")
        final = root / "releases" / release
        if final.exists():
            if final.is_symlink() or (final / "_tutorial/received.sha256").read_text().strip() != digest:
                raise ValueError("refusing to replace a different existing release")
        else:
            unpacked.rename(final)
        print(f"received verified deployment {release}", flush=True)
        return final
    finally:
        shutil.rmtree(temp)


if __name__ == "__main__":
    os.umask(0o077)
    receive(json.loads(sys.argv[1]), sys.argv[2], sys.argv[3], sys.stdin.buffer)
