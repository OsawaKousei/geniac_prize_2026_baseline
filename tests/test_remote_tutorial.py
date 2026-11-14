"""Exercise deployment boundaries locally: no SSH, downloads, GPU, or model files."""

from __future__ import annotations

import hashlib
import io
import json
import os
import pwd
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import yaml

from geniac_baseline.collect.plan import load_spec, preflight
from geniac_baseline.trainer.config import load_config
from remote_tutorial import client, receive, runner
from remote_tutorial.common import canonical, job_shell, resources, sha256, validate_config, work_root

from .test_collect_eval import fake_checkout

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def config(tmp_path: Path) -> dict:
    return validate_config(
        {
            "host": "cluster-alias",
            "account": "team-account",
            "partition": "team-partition",
            "remote_user": pwd.getpwuid(os.getuid()).pw_name,
            "shared_root": str(tmp_path / "shared"),
            "port_offset": 2000,
        }
    )


def git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    repo = tmp_path / "repository"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Test")
    git(repo, "config", "user.email", "test@example.invalid")
    (repo / "README.md").write_text("initial source\n")
    (repo / ".gitignore").write_text(
        "/.remote-build/\n/.remote.local.json\n/tmp/\nuv.lock\n"
        "!/remote_tutorial/envs/base/uv.lock\n!/remote_tutorial/envs/serve/uv.lock\n"
    )
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "initial")
    for name in ("remote_tutorial", "patches/geniac_prize", "configs", "tests"):
        (repo / name).mkdir(parents=True)
    for name in ("base", "serve"):
        project = repo / "remote_tutorial/envs" / name
        project.mkdir(parents=True)
        for filename in ("pyproject.toml", "uv.lock"):
            (project / filename).write_text("# fixture project metadata\n")
    shutil.copyfile(REPO / "remote_tutorial/receive.py", repo / "remote_tutorial/receive.py")
    shutil.copyfile(REPO / "patches/geniac_prize/setup.sh", repo / "patches/geniac_prize/setup.sh")
    return repo


def payload(repo: Path, config: dict) -> dict:
    bundle = repo.parent / "source.bundle"
    bundle.write_bytes(b"test source bundle\n")
    return client.build_payload(repo, config, bundle)


@pytest.mark.parametrize(
    "override",
    [
        {"host": "$(touch invalid)"},
        {"shared_root": "/"},
        {"shared_root": "/shared/../outside"},
        {"project": "../models"},
        {"port_offset": None},
        {"port_offset": 60000},
        {"collect_seeds": ["301-302"]},
        {"collect_seeds": ["301-302", "391-401"]},
        {"collect_seeds": ["301-302", "302", "391"]},
        {"eval_seeds": ["391-392"]},
        {"gpu_gres": "gpu:2"},
        {"capx_url": "https://token@github.com/example/code.git"},
        {"unknown_key": True},
    ],
)
def test_reject_unsafe_or_incomplete_config(config: dict, override: dict) -> None:
    with pytest.raises(ValueError):
        validate_config({**config, **override})


def test_models_cannot_be_a_personal_workspace(config: dict) -> None:
    with pytest.raises(ValueError, match="overlap"):
        validate_config({**config, "models_root": str(work_root(config))})


def test_payload_uses_current_files_and_excludes_private_material(repository: Path, config: dict) -> None:
    (repository / "README.md").write_text("uncommitted edit\n")
    (repository / "tests/new_test.py").write_text("# untracked source\n")
    for name in ("tmp/private.md", ".remote.local.json", "configs/my_collect.yaml", "docs/.env"):
        path = repository / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("secret fixture\n")
    first = payload(repository, config)
    with tarfile.open(first["archive"]) as archive:
        assert archive.extractfile("README.md").read() == b"uncommitted edit\n"
        assert "tests/new_test.py" in archive.getnames()
        for name in ("base", "serve"):
            assert f"remote_tutorial/envs/{name}/uv.lock" in archive.getnames()
            assert f"remote_tutorial/envs/{name}/pyproject.toml" in archive.getnames()
        assert not any("private" in name or ".env" in name or "my_collect" in name for name in archive.getnames())
    first["jobs"] = {"setup": "123"}
    first["last_job"] = {"stage": "setup", "id": "123"}
    client.save_state(repository, first)
    second = payload(repository, config)
    assert first == second
    (repository / "README.md").write_text("next edit\n")
    assert payload(repository, config)["release"] != first["release"]


def test_source_symlink_is_rejected(repository: Path) -> None:
    (repository / "docs").mkdir()
    (repository / "docs/linked.md").symlink_to(repository / "README.md")
    with pytest.raises(ValueError, match="symlink"):
        client.source_files(repository)


@pytest.mark.parametrize("filename", ["pyproject.toml", "uv.lock"])
def test_payload_requires_complete_uv_projects(repository: Path, config: dict, filename: str) -> None:
    (repository / "remote_tutorial/envs/base" / filename).unlink()
    with pytest.raises(ValueError, match="missing uv project file"):
        payload(repository, config)


def test_payload_refuses_an_ignored_lock(repository: Path, config: dict) -> None:
    with (repository / ".gitignore").open("a") as stream:
        stream.write("/remote_tutorial/envs/serve/uv.lock\n")
    with pytest.raises(ValueError, match="Git ignore rules"):
        payload(repository, config)


def test_base_project_preserves_the_canonical_training_pins() -> None:
    tomllib = pytest.importorskip("tomllib")
    project = tomllib.loads((REPO / "remote_tutorial/envs/base/pyproject.toml").read_text())
    dependencies = set(project["project"]["dependencies"])
    for line in (REPO / "requirements-train.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        if " @ git+" in line:
            name, source = line.split(" @ git+")
            url, revision = source.rsplit("@", 1)
            assert project["tool"]["uv"]["sources"][name] == {"git": url, "rev": revision}
            assert name in dependencies
        else:
            assert line in dependencies, f"canonical training pin was changed or omitted: {line}"


def test_project_sync_keeps_the_final_venv_and_rejects_stale_locks(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def capture(args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(runner, "command", capture)
    original_env = {"UV_PROJECT_ENVIRONMENT": "inherited-environment"}
    target = tmp_path / "final/.venv"
    project = tmp_path / "source/remote_tutorial/envs/base"
    runner.sync_project(Path("uv"), project, target, "3.12.16", original_env)
    args, options = calls[0]
    assert args[args.index("--project") + 1] == project
    assert "--locked" in args and "--frozen" not in args
    assert options["env"]["UV_PROJECT_ENVIRONMENT"] == str(target)
    assert original_env["UV_PROJECT_ENVIRONMENT"] == "inherited-environment"


def test_prepare_source_does_not_modify_the_supplied_checkout(repository: Path, config: dict) -> None:
    pin = git(repository, "rev-parse", "HEAD")
    script = repository / "patches/geniac_prize/setup.sh"
    script.write_text(script.read_text().replace("ac680843fb60c464aa74c87a2815cb5917865be4", pin))
    before = git(repository, "status", "--porcelain")
    bundle = client.prepare_source(repository, config, str(repository))
    assert f"{pin} HEAD" in git(repository, "bundle", "list-heads", str(bundle))
    assert git(repository, "status", "--porcelain") == before
    assert git(repository, "rev-parse", "HEAD") == pin


def test_planning_never_uses_ssh(repository: Path, config: dict, monkeypatch) -> None:
    state = payload(repository, config)
    cfg = repository / ".remote.local.json"
    cfg.write_bytes(canonical(config))

    def forbidden(*args, **kwargs):
        pytest.fail("local planning must not use SSH")

    monkeypatch.setattr(client, "ssh", forbidden)
    assert client.main(["plan"], repo=repository) == 0
    plan = json.loads((Path(state["archive"]).parent / "plan.json").read_text())
    assert plan["network_operations_performed"] is False
    assert plan["transfer_on_compute"].startswith("srun --quiet ")
    assert "--gres=" not in plan["stages"]["setup"]
    assert "--gres=gpu:1" in plan["stages"]["check"]


def test_dependency_uses_last_submission_instead_of_sorted_job_keys(config: dict) -> None:
    state = {
        "release": "a" * 24,
        "jobs": {"check": "9", "collect": "10", "setup": "8"},
        "last_job": {"stage": "collect", "id": "10"},
    }
    args = client.batch_command(config, json.loads(canonical(state)), "convert")
    assert "--dependency=afterok:10" in args
    assert "--kill-on-invalid-dep=yes" in args
    assert not any(arg.startswith("--gres") for arg in args)
    for stage in ("setup", "check", "collect", "convert", "sft"):
        script = job_shell(str(work_root(config)), state["release"], stage)
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    assert "--mem=256G" in resources(config, "sft")


def test_generated_configs_pass_existing_preflights(config: dict) -> None:
    release = Path(str(work_root(config))) / "releases" / ("b" * 24)
    (release / "_tutorial").mkdir(parents=True)
    shutil.copytree(REPO / "configs", release / "configs")
    code = fake_checkout(release / "geniac_prize_2026_code", ["Qwen/Qwen3.8-27B"], url_env=True)
    (release / "_tutorial/qwen.jinja").write_text("{{ messages }}")
    paths = runner.configure(release, {"release": release.name, "config": config})
    for name in ("collect", "eval-base"):
        spec = load_spec(paths[name])
        preflight(spec)
        assert spec.capx_root == code
        assert spec.tasks[0].task == "cube_restack"
        assert spec.tasks[0].config.is_absolute()
        assert spec.port_offset == spec.vllm_port_offset == 2000
    train = load_config(paths["sft"])
    assert train.max_steps == 2 and train.require_val
    assert not train.wandb.enabled
    assert train.output_dir.is_absolute()
    value = yaml.safe_load(paths["collect"].read_text())
    value["num_workers"] = 2
    paths["collect"].write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="overwrite edited"):
        runner.configure(release, {"release": release.name, "config": config})


def test_receive_checks_integrity_and_preserves_existing_release(repository: Path, config: dict, monkeypatch) -> None:
    Path(config["shared_root"]).mkdir()
    state = payload(repository, config)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(sys, "path", list(sys.path))
    with Path(state["archive"]).open("rb") as stream:
        target = receive.receive(config, state["release"], state["sha256"], stream)
    assert (target / "README.md").read_text() == "initial source\n"
    assert (target / "_tutorial/jobs/setup.sh").is_file()
    (target / "README.md").write_text("remote edit\n")
    with Path(state["archive"]).open("rb") as stream:
        assert receive.receive(config, state["release"], state["sha256"], stream) == target
    assert (target / "README.md").read_text() == "remote edit\n"
    assert not list(target.parents[1].glob(".receive-*"))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        receive.receive(config, state["release"], state["sha256"], io.BytesIO(b"corrupted"))


@pytest.mark.parametrize("kind", ["traversal", "absolute", "symlink", "duplicate"])
def test_receive_rejects_unsafe_archives(config: dict, kind: str, monkeypatch) -> None:
    Path(config["shared_root"]).mkdir()
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as archive:
        name = "../escape" if kind == "traversal" else "/escape" if kind == "absolute" else "file"
        member = tarfile.TarInfo(name)
        if kind == "symlink":
            member.type, member.linkname = tarfile.SYMTYPE, "../escape"
        archive.addfile(member)
        if kind == "duplicate":
            archive.addfile(member)
    raw = data.getvalue()
    with pytest.raises(ValueError, match="unsafe archive"):
        receive.receive(config, "c" * 24, hashlib.sha256(raw).hexdigest(), io.BytesIO(raw))


def test_receive_requires_compute_job(config: dict, monkeypatch) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="Slurm compute"):
        receive.receive(config, "c" * 24, "d" * 64, io.BytesIO())


def test_receive_rejects_workspace_symlink(config: dict, tmp_path: Path, monkeypatch) -> None:
    Path(config["shared_root"]).mkdir()
    other = tmp_path / "other"
    other.mkdir()
    (Path(config["shared_root"]) / config["remote_user"]).symlink_to(other, target_is_directory=True)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    with pytest.raises(ValueError, match="symlink"):
        receive.receive(config, "c" * 24, "d" * 64, io.BytesIO())
    assert not list(other.iterdir())


def test_sft_requires_successful_examples_in_both_splits(tmp_path: Path) -> None:
    (tmp_path / "sft.jsonl").write_text('{"messages": []}\n')
    with pytest.raises(ValueError, match="insufficient"):
        runner.require_training_rows(tmp_path)
    (tmp_path / "val.jsonl").write_text('{"messages": []}\n')
    assert runner.require_training_rows(tmp_path) == {"sft.jsonl": 1, "val.jsonl": 1}


def test_dependency_check_rejects_new_warnings(monkeypatch, tmp_path: Path) -> None:
    def checked(args, **kwargs):
        return subprocess.CompletedProcess(args, 1, "\n".join(sorted(runner.KNOWN_SIM_WARNINGS)), "")

    monkeypatch.setattr(subprocess, "run", checked)
    assert runner.dependency_check(tmp_path, tmp_path, tmp_path, {}, simulator=True)["exit_code"] == 1
    with pytest.raises(RuntimeError, match="unexpected dependency"):
        runner.dependency_check(tmp_path, tmp_path, tmp_path, {})

    def unknown(args, **kwargs):
        return subprocess.CompletedProcess(args, 1, "The package `new-package` requires `missing`", "")

    monkeypatch.setattr(subprocess, "run", unknown)
    with pytest.raises(RuntimeError, match="unexpected dependency"):
        runner.dependency_check(tmp_path, tmp_path, tmp_path, {}, simulator=True)


def test_setup_refuses_tampered_archive_before_ssh(repository: Path, config: dict, monkeypatch) -> None:
    state = payload(repository, config)
    (repository / ".remote.local.json").write_bytes(canonical(config))
    archive = Path(state["archive"])
    archive.write_bytes(archive.read_bytes() + b"tampered")
    assert sha256(archive) != state["sha256"]

    def forbidden(*args, **kwargs):
        pytest.fail("modified archive must not be sent")

    monkeypatch.setattr(client, "ssh", forbidden)
    assert client.main(["setup"], repo=repository) == 1


def test_transfer_uses_the_prepared_receiver(repository: Path, config: dict) -> None:
    expected = (repository / "remote_tutorial/receive.py").read_text()
    state = payload(repository, config)
    (repository / "remote_tutorial/receive.py").write_text("changed after prepare\n")
    args = client.transfer_command(config, state)
    assert args[args.index("-c") + 1] == expected
