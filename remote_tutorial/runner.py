"""Compute-node tutorial stages. Local tooling never calls these stages directly."""

from __future__ import annotations

import fcntl
import json
import os
import platform
import pwd
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from .common import MODELS, canonical, sha256, validate_config, work_root

KNOWN_SIM_WARNINGS = {
    "The package `pyrender` requires `pyopengl==3.1.0`, but `3.1.6` is installed",
    "The package `opencv-python-headless` requires `numpy>=2,<2.3.0 ; "
    "python_full_version >= '3.9'`, but `1.26.4` is installed",
    "The package `robosuite` requires `opencv-python`, but it's not installed",
    "The package `decord` was built for a different platform",
}


def command(args: list, *, cwd: Path, env: dict | None = None, capture: bool = False) -> str:
    print("+", shlex.join(map(str, args)), flush=True)
    result = subprocess.run(
        list(map(str, args)),
        cwd=cwd,
        env=env,
        check=True,
        stdout=subprocess.PIPE if capture else None,
        text=True,
    )
    return result.stdout or ""


def paths(release: Path, manifest: dict) -> tuple[dict, Path, Path]:
    c = validate_config(manifest["config"])
    root = Path(str(work_root(c)))
    if release != root / "releases" / manifest["release"]:
        raise ValueError("release does not match its configured personal workspace")
    return c, root, root / "runs" / manifest["release"]


def model_paths(c: dict) -> dict[str, Path]:
    return {name: Path(c["models_root"]) / model / rev for name, (model, rev) in MODELS.items()}


def validate_models(c: dict) -> dict[str, Path]:
    models = model_paths(c)
    for name, path in models.items():
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"shared {name} model is missing: {path}")
        if name == "sam3":
            if not (path / "sam3.pt").is_file():
                raise FileNotFoundError(f"shared SAM3 checkpoint is missing: {path}")
        else:
            index = json.loads((path / "model.safetensors.index.json").read_text())
            for shard in set(index["weight_map"].values()):
                shard_path = path / shard
                if Path(shard).name != shard or not shard_path.is_file():
                    raise FileNotFoundError(f"missing or unsafe model shard: {shard}")
    return models


def environment(release: Path, root: Path, models: dict[str, Path], versions: dict) -> dict:
    env = {**os.environ}
    # Avoid inheriting local/interactive venvs, user-site packages, or a globally selected CUDA toolkit.
    for name in ("VIRTUAL_ENV", "PYTHONHOME", "CUDA_HOME", "CUDA_PATH"):
        env.pop(name, None)
    env.update(
        {
            "UV_CACHE_DIR": str(root / "cache/uv"),
            "UV_PYTHON_INSTALL_DIR": str(root / "tools/python"),
            "UV_PYTHON_BIN_DIR": str(root / "tools/bin"),
            "UV_LINK_MODE": "hardlink",
            "UV_HTTP_TIMEOUT": "300",
            "HF_HOME": str(root / "cache/huggingface"),
            "HF_HUB_CACHE": str(root / "cache/huggingface/hub"),
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "WANDB_MODE": "offline",
            "TORCH_HOME": str(root / "cache/torch"),
            "XDG_CACHE_HOME": str(root / "cache"),
            "TORCH_EXTENSIONS_DIR": str(root / "cache/torch_extensions"),
            "VLLM_CACHE_ROOT": str(root / "cache/vllm"),
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "VLLM_NO_USAGE_STATS": "1",
            "NUMBA_CACHE_DIR": str(root / "cache/numba"),
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "TORCH_CUDA_ARCH_LIST": "10.0",
            "PYTHONNOUSERSITE": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "TMPDIR": "/tmp",
            "CAPX_SAM3_CHECKPOINT": str(models["sam3"] / "sam3.pt"),
            "PYTHONPATH": os.pathsep.join([str(release), str(release / "geniac_prize_2026_code")]),
        }
    )
    nvrtc = root / f"tools/nvrtc-{versions['nvrtc']}/nvidia/cuda_nvrtc/lib/libnvrtc.so.12"
    if nvrtc.exists():
        env["LD_PRELOAD"] = str(nvrtc)
    return env


def install_uv(root: Path, version: str, expected: str) -> Path:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("the compute environment must be Linux x86_64")
    binary = root / f"tools/uv-{version}/uv"
    if not binary.exists():
        target = binary.parent
        target.mkdir(parents=True, exist_ok=True)
        name = "uv-x86_64-unknown-linux-gnu.tar.gz"
        url = f"https://github.com/astral-sh/uv/releases/download/{version}/{name}"
        with tempfile.TemporaryDirectory(prefix="uv-", dir=root / "tools") as folder:
            archive = Path(folder) / name
            urllib.request.urlretrieve(url, archive)
            if not re.fullmatch(r"[a-f0-9]{64}", expected) or sha256(archive) != expected:
                raise ValueError("uv distribution checksum mismatch")
            with tarfile.open(archive) as tar:
                matches = [m for m in tar if m.isfile() and Path(m.name).name == "uv"]
                if len(matches) != 1:
                    raise ValueError("unexpected uv distribution contents")
                candidate = Path(folder) / "uv"
                with tar.extractfile(matches[0]) as source, candidate.open("xb") as destination:
                    destination.write(source.read())
            candidate.chmod(0o750)
            candidate.replace(binary)
    actual = command([binary, "--version"], cwd=root, capture=True)
    if actual.split()[:2] != ["uv", version]:
        raise ValueError(f"unexpected uv version: {actual}")
    return binary


def dependency_check(uv: Path, python: Path, release: Path, env: dict, *, simulator: bool = False) -> dict:
    result = subprocess.run(
        [str(uv), "--no-config", "pip", "check", "--python", str(python)],
        cwd=release,
        env=env,
        text=True,
        capture_output=True,
    )
    text = result.stdout + result.stderr
    print(text, flush=True)
    diagnostics = {line.strip() for line in text.splitlines() if line.startswith("The package ")}
    allowed = simulator and result.returncode == 1 and diagnostics and diagnostics <= KNOWN_SIM_WARNINGS
    if result.returncode and not allowed:
        raise RuntimeError(f"unexpected dependency incompatibility in {python}:\n{text}")
    return {"exit_code": result.returncode, "known_warnings": sorted(diagnostics), "output": text}


def sync_project(uv: Path, project: Path, target: Path, python: str, env: dict) -> None:
    """Sync the selected project into its final venv; fail on stale project metadata."""
    command(
        [uv, "sync", "--project", project, "--locked", "--python", python],
        cwd=project,
        env={**env, "UV_PROJECT_ENVIRONMENT": str(target)},
    )


def setup(release: Path, manifest: dict) -> None:
    c, root, _ = paths(release, manifest)
    models = validate_models(c)
    versions = json.loads((release / "remote_tutorial/versions.json").read_text())
    env = environment(release, root, models, versions)
    uv = install_uv(root, versions["uv"], versions["uv_sha256"])
    command(
        [uv, "--no-config", "python", "install", versions["python_base"], versions["python_sim"]], cwd=release, env=env
    )
    code = release / "geniac_prize_2026_code"
    recipe = manifest["recipe"]
    marker = release / "_tutorial/source-ready"
    if not marker.exists():
        if not (code / ".git").exists():
            command(["git", "clone", "--no-checkout", release / "_tutorial/capx.bundle", code], cwd=release, env=env)
        command(["git", "checkout", "--detach", recipe["commit"]], cwd=code, env=env)
        command(["git", "remote", "set-url", "origin", c["capx_url"]], cwd=code, env=env)
        for patch in recipe["patches"]:
            path = release / "patches/geniac_prize" / patch
            applied = subprocess.run(["git", "apply", "--reverse", "--check", str(path)], cwd=code, capture_output=True)
            if applied.returncode:
                command(["git", "apply", "--check", path], cwd=code, env=env)
                command(["git", "apply", path], cwd=code, env=env)
        command(["git", "submodule", "update", "--init", "--recursive", "--", *recipe["submodules"]], cwd=code, env=env)
        status = command(["git", "submodule", "status", "--", *recipe["submodules"]], cwd=code, env=env, capture=True)
        if any(not line.startswith(" ") for line in status.splitlines()):
            raise ValueError("submodule revisions do not match the pinned parent")
        marker.touch()
    for label, target, python in (
        ("base", release / ".venv", versions["python_base"]),
        ("sim", code / ".venv", versions["python_sim"]),
        ("serve", code / ".venv-serve", versions["python_base"]),
    ):
        done = release / "_tutorial" / f"{label}-ready"
        if done.exists():
            continue
        if not (target / "bin/python").exists():
            command([uv, "--no-config", "venv", "--python", python, target], cwd=release, env=env)
        if label == "sim":
            command(
                [uv, "sync", "--active", "--frozen", "--extra", "robosuite"],
                cwd=code,
                env={**env, "VIRTUAL_ENV": str(target)},
            )
            command(
                [target / "bin/python", "capx/third_party/robosuite/robosuite/scripts/setup_macros.py"],
                cwd=code,
                env=env,
            )
        else:
            sync_project(uv, release / "remote_tutorial/envs" / label, target, python, env)
        done.touch()
    nvrtc = root / f"tools/nvrtc-{versions['nvrtc']}"
    if not (nvrtc / "nvidia/cuda_nvrtc/lib/libnvrtc.so.12").exists():
        command(
            [
                uv,
                "--no-config",
                "pip",
                "install",
                "--python",
                release / ".venv/bin/python",
                "--target",
                nvrtc,
                "--no-deps",
                f"nvidia-cuda-nvrtc-cu12=={versions['nvrtc']}",
            ],
            cwd=release,
            env=env,
        )
    env = environment(release, root, models, versions)
    audit = {}
    for label, python in (
        ("base", release / ".venv/bin/python"),
        ("sim", code / ".venv/bin/python"),
        ("serve", code / ".venv-serve/bin/python"),
    ):
        audit[label] = dependency_check(uv, python, release, env, simulator=label == "sim")
        frozen = command([uv, "--no-config", "pip", "freeze", "--python", python], cwd=release, env=env, capture=True)
        (release / "_tutorial" / f"{label}-freeze.txt").write_text(frozen)
    (release / "_tutorial/dependency-audit.json").write_bytes(canonical(audit))
    base = release / ".venv/bin/python"
    command([base, "-m", "remote_tutorial.runner", "configure", release], cwd=release, env=env)
    command([base, "-m", "pytest", "-q"], cwd=release, env=env)
    command(
        [
            base,
            "-m",
            "geniac_baseline.cli",
            "template",
            "--model-dir",
            models["qwen"],
            "--out",
            release / "_tutorial/qwen.jinja",
        ],
        cwd=release,
        env=env,
    )
    command([base, "-m", "remote_tutorial.runner", "preflight", release], cwd=release, env=env)
    (release / "_tutorial/setup-complete").write_text(manifest["release"] + "\n")


def configure(release: Path, manifest: dict) -> dict[str, Path]:
    import yaml

    c, _, out = paths(release, manifest)
    models = model_paths(c)
    folder = release / "_tutorial/config"
    folder.mkdir(parents=True, exist_ok=True)
    task_file = release / "configs/capx/geniac_prize/robosuite" / f"{c['task']}.yaml"
    if not task_file.is_file():
        raise ValueError(f"unsupported Robosuite task: {c['task']}")
    collect = yaml.safe_load((release / "configs/collect.yaml").read_text())
    collect.update(
        {
            "capx_root": str(release / "geniac_prize_2026_code"),
            "model_path": str(models["qwen"]),
            "molmo_path": str(models["molmo"]),
            "chat_template": str(release / "_tutorial/qwen.jinja"),
            "output_dir": str(out / "collect"),
            "b300_workarounds": True,
            "num_workers": 1,
            "port_offset": c["port_offset"],
            "vllm_port_offset": c["port_offset"],
            "tasks": [{"task": c["task"], "config": str(task_file), "seeds": c["collect_seeds"]}],
            "extra_env": {"CAPX_SAM3_CHECKPOINT": str(models["sam3"] / "sam3.pt"), "VLLM_USE_FLASHINFER_SAMPLER": "0"},
        }
    )
    evaluate = {
        **collect,
        "mode": "eval",
        "output_dir": str(out / "eval-base"),
        "adapter": None,
        "code_thinking": False,
    }
    evaluate["tasks"] = [{"task": c["task"], "config": str(task_file), "seeds": c["eval_seeds"]}]
    train = yaml.safe_load((release / "configs/train_sft.yaml").read_text())
    train.update(
        {
            "data": str(out / "data/split"),
            "model": str(models["qwen"]),
            "output_dir": str(out / "sft"),
            "epochs": 1,
            "max_steps": c["sft_max_steps"],
            "grad_accum": 1,
            "warmup_steps": 1,
            "val_max_rows": 2,
            "wandb": {"enabled": False, "mode": "offline"},
        }
    )
    configs = {"collect": collect, "eval-base": evaluate, "sft": train}
    result = {}
    for name, value in configs.items():
        path = folder / f"{name}.yaml"
        data = yaml.safe_dump(value, sort_keys=False)
        if path.exists() and path.read_text() != data:
            raise ValueError(f"refusing to overwrite edited generated configuration: {path}")
        path.write_text(data)
        result[name] = path
    return result


def preflight(release: Path) -> None:
    from geniac_baseline.collect.plan import load_spec
    from geniac_baseline.collect.plan import preflight as check

    for name in ("collect", "eval-base"):
        check(load_spec(release / "_tutorial/config" / f"{name}.yaml"))


def gpu_check(release: Path, manifest: dict, env: dict) -> None:
    base = release / ".venv/bin/python"
    code = release / "geniac_prize_2026_code"
    check = (
        "import torch,transformers,peft,accelerate; assert torch.cuda.is_available(); "
        "assert torch.ones(4,device='cuda').sum().item()==4; "
        "print(torch.__version__,transformers.__version__,peft.__version__,"
        "accelerate.__version__,torch.cuda.get_device_name())"
    )
    command([base, "-c", check], cwd=release, env=env)
    task = release / "configs/capx/geniac_prize/robosuite" / f"{manifest['config']['task']}.yaml"
    command(
        [code / ".venv/bin/python", "-m", "remote_tutorial.probe", task, release / "_tutorial/task-reset.png"],
        cwd=code,
        env=env,
    )
    registry = (
        "from vllm.model_executor.models.registry import ModelRegistry; import torch,vllm; "
        "assert torch.cuda.is_available(); assert vllm.__version__=='0.26.0'; "
        "assert {'Qwen3_5ForConditionalGeneration','Molmo2ForConditionalGeneration'} "
        "<= set(ModelRegistry.get_supported_archs()); print('SERVE_OK',torch.__version__,vllm.__version__)"
    )
    command([code / ".venv-serve/bin/python", "-c", registry], cwd=release, env=env)
    (release / "_tutorial/check-complete").write_text(manifest["release"] + "\n")


def run_plan(release: Path, spec_path: Path, env: dict) -> Path:
    from geniac_baseline.collect.plan import load_spec, write_plan

    spec = load_spec(spec_path)
    if spec.output_dir.exists():
        raise FileExistsError(f"experiment output exists: {spec.output_dir}; choose a new experiment in local config")
    occupied = subprocess.run(["ss", "-H", "-ltn"], capture_output=True, text=True, check=True).stdout
    ports = {8110, 8111, 8113, 8114, 8115, 8116, 8117, 8122}
    off = spec.port_offset
    if any(line.split()[3].rsplit(":", 1)[-1] in {str(p + off) for p in ports} for line in occupied.splitlines()):
        raise RuntimeError("tutorial ports are already in use; choose another port_offset")
    script = write_plan(spec)
    command(["bash", script], cwd=release, env=env)
    # CaP-X may fall back after a perception service error: report those as a failed smoke run.
    errors = []
    for path in spec.output_dir.rglob("*.log"):
        with path.open(errors="replace") as stream:
            for line in stream:
                if re.search(
                    r"invalid value for --gpu-architecture|HTTP/1\.[01].* 500|"
                    r"CUDA error|Traceback \(most recent call last\)",
                    line,
                ):
                    errors.append(f"{path.name}: {line.strip()}")
                    if len(errors) >= 20:
                        break
    if errors:
        raise RuntimeError("runtime service errors:\n" + "\n".join(errors))
    return spec.output_dir


def convert(release: Path, manifest: dict, env: dict) -> None:
    _, _, out = paths(release, manifest)
    gb = [release / ".venv/bin/python", "-m", "geniac_baseline.cli"]
    data = out / "data"
    steps = [
        ("rows", ["convert", "rows", "--sources", out / "collect/sources.yaml", "--out", data / "rows"]),
        (
            "tagged",
            [
                "convert",
                "tag",
                "--src",
                data / "rows",
                "--out",
                data / "tagged",
                "--tokenizer",
                model_paths(manifest["config"])["qwen"],
            ],
        ),
        ("split", ["convert", "split", "--src", data / "tagged", "--out", data / "split"]),
    ]
    for name, args in steps:
        target = data / name
        if (target / "manifest.json").is_file():
            continue
        command([*gb, *args], cwd=release, env=env)
    require_training_rows(data / "split")
    command([*gb, "convert", "validate", data / "split/sft.jsonl", data / "split/val.jsonl"], cwd=release, env=env)
    command([*gb, "train", "sft", "--config", release / "_tutorial/config/sft.yaml", "--dry-run"], cwd=release, env=env)


def require_training_rows(directory: Path) -> dict[str, int]:
    counts = {}
    for filename in ("sft.jsonl", "val.jsonl"):
        path = directory / filename
        counts[filename] = sum(bool(line.strip()) for line in path.read_text().splitlines()) if path.exists() else 0
    if not all(counts.values()):
        raise ValueError(
            f"successful training/validation examples are insufficient: {counts}. "
            "Four trials do not guarantee successes; collect additional allowed seeds in a new experiment."
        )
    print("DATA_READY", counts, flush=True)
    return counts


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    stage, directory = args
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("this entry point must run in a Slurm compute job")
    release = Path(directory).resolve()
    manifest = json.loads((release / "_tutorial/manifest.json").read_text())
    c, root, out = paths(release, manifest)
    if pwd.getpwuid(os.getuid()).pw_name != c["remote_user"] or release.stat().st_uid != os.getuid():
        raise ValueError("compute user does not own the deployment")
    versions = json.loads((release / "remote_tutorial/versions.json").read_text())
    env = environment(release, root, model_paths(c), versions)
    os.umask(0o077)
    print("TUTORIAL_STAGE", stage, manifest["release"], "JOB", os.environ["SLURM_JOB_ID"], flush=True)
    if stage in {"configure", "preflight"}:
        configure(release, manifest) if stage == "configure" else preflight(release)
        return
    lock_path = release / "_tutorial/stage.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if stage == "setup":
            with (root / "tools.lock").open("a") as tools:
                fcntl.flock(tools, fcntl.LOCK_EX | fcntl.LOCK_NB)
                setup(release, manifest)
            return
        if not (release / "_tutorial/setup-complete").is_file():
            raise ValueError("environment setup has not completed")
        validate_models(c)
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        if stage == "check":
            gpu_check(release, manifest, env)
            return
        if not (release / "_tutorial/check-complete").is_file():
            raise ValueError("run the GPU/task check stage first")
        if stage == "convert":
            convert(release, manifest, env)
        elif stage == "sft":
            require_training_rows(out / "data/split")
            if (out / "sft").exists():
                raise FileExistsError("SFT output exists; resume manually or select a new experiment")
            command(
                [
                    release / ".venv/bin/python",
                    "-m",
                    "geniac_baseline.cli",
                    "train",
                    "sft",
                    "--config",
                    release / "_tutorial/config/sft.yaml",
                ],
                cwd=release,
                env=env,
            )
        elif stage in {"collect", "eval-base", "eval-sft"}:
            spec_path = release / "_tutorial/config" / f"{stage}.yaml"
            if stage == "eval-sft":
                import yaml

                from geniac_baseline.trainer.dmpo_trainer import resolve_init_adapter

                cfg = yaml.safe_load((release / "_tutorial/config/eval-base.yaml").read_text())
                # Short smoke runs may stop before a full epoch; the trainer still writes its final adapter.
                selected = (
                    resolve_init_adapter(out / "sft") if (out / "sft/best_epoch.json").exists() else out / "sft/adapter"
                )
                cfg.update(output_dir=str(out / "eval-sft"), adapter=str(selected), lora_name="sft-smoke")
                spec_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            # A per-user lock also prevents port collisions between different source deployments.
            with (root / "ports.lock").open("a") as ports:
                fcntl.flock(ports, fcntl.LOCK_EX | fcntl.LOCK_NB)
                output = run_plan(release, spec_path, env)
            if stage.startswith("eval"):
                command(
                    [
                        release / ".venv/bin/python",
                        "-m",
                        "geniac_baseline.cli",
                        "eval",
                        "score",
                        output,
                        "--json",
                        output / "score.json",
                    ],
                    cwd=release,
                    env=env,
                )
        else:
            raise ValueError(f"unknown stage: {stage}")
        (release / "_tutorial" / f"{stage}-complete").write_text(manifest["release"] + "\n")


if __name__ == "__main__":
    main()
