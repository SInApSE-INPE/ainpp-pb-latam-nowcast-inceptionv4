"""A generated project must work, and the runtime must stay framework free.

Both are architectural invariants rather than behaviour, and both are the kind
of thing that rots silently without a test holding it.
"""

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from nowcast_runtime import scaffold
from nowcast_runtime.artifacts import verify_bundle

RUNTIME_PYPROJECT = Path(scaffold.__file__).resolve().parents[2] / "pyproject.toml"
FRAMEWORKS = ("torch", "tensorflow", "jax", "keras", "paddlepaddle", "mxnet")


def test_the_runtime_depends_on_no_deep_learning_framework():
    """No port mentions one, so no dependency should drag one in either.

    A model on NumPy, TensorFlow or JAX must not be made to install torch, and
    nothing in this package imports it.
    """
    declared = tomllib.loads(RUNTIME_PYPROJECT.read_text(encoding="utf-8"))
    requirements = declared["project"]["dependencies"]
    offenders = [
        requirement
        for requirement in requirements
        for framework in FRAMEWORKS
        if requirement.lower().startswith(framework)
    ]
    assert not offenders, f"The runtime must stay framework free, but declares {offenders}"


def test_no_runtime_module_imports_a_framework():
    source = Path(scaffold.__file__).resolve().parent
    offenders = []
    for path in source.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for framework in FRAMEWORKS:
            if f"import {framework}" in text:
                offenders.append(f"{path.name} imports {framework}")
    assert not offenders, offenders


def plan_for(target: Path) -> scaffold.Plan:
    return scaffold.plan_from(
        scaffold.parser().parse_args(
            ["--name", "Demo Net", "--product", "demo_rain", "--into", str(target)]
        )
    )


def test_a_generated_project_is_complete_and_declares_itself(tmp_path):
    plan = plan_for(tmp_path / "demo")
    scaffold.generate(plan)
    root = plan.target
    for relative in (
        "pyproject.toml",
        "README.md",
        "Dockerfile",
        "compose.yaml",
        "config/default.json",
        "tests/test_conformance.py",
        f"src/{plan.package}/entrypoint.py",
        f"src/{plan.package}/model.py",
        "artifacts/v1/manifest.json",
    ):
        assert (root / relative).is_file(), f"{relative} was not generated"

    declared = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert declared["project"]["name"] == "demo-net-nowcast"
    assert any(item.startswith("nowcast-runtime") for item in declared["project"]["dependencies"])
    assert declared["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        f"src/{plan.package}"
    ]


def test_the_generated_bundle_verifies_against_its_own_manifest(tmp_path):
    plan = plan_for(tmp_path / "demo")
    scaffold.generate(plan)
    manifest, artifacts = verify_bundle(plan.target / "artifacts" / "v1", ("checkpoint",))
    assert manifest["placeholder"] is True
    assert artifacts["checkpoint"].is_file()


def test_every_placeholder_is_marked_so_it_can_be_found(tmp_path):
    plan = plan_for(tmp_path / "demo")
    scaffold.generate(plan)
    marked = [
        path.relative_to(plan.target).as_posix()
        for path in plan.target.rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".md"}
        and scaffold.PLACEHOLDER in path.read_text(encoding="utf-8")
    ]
    assert "src/demo_net_nowcast/model.py" in marked
    assert "src/demo_net_nowcast/entrypoint.py" in marked
    assert "README.md" in marked


def test_generated_sources_are_syntactically_valid(tmp_path):
    plan = plan_for(tmp_path / "demo")
    scaffold.generate(plan)
    for path in (plan.target / "src").rglob("*.py"):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    config = json.loads((plan.target / "config" / "default.json").read_text(encoding="utf-8"))
    assert config["product"] == "demo_rain"


def test_a_generated_project_passes_its_own_conformance_suite(tmp_path):
    """The whole promise of the scaffold: green before a line is written.

    Run in a subprocess with the generated src on the path, so the generated
    package is imported exactly as it would be once installed.
    """
    plan = plan_for(tmp_path / "demo")
    scaffold.generate(plan)
    environment = {
        "PYTHONPATH": str(plan.target / "src"),
        "PATH": str(Path(sys.executable).parent),
        "SYSTEMROOT": "C:\\Windows",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            str(plan.target / "tests"),
            f"--basetemp={tmp_path / 'pytest'}",
        ],
        capture_output=True,
        text=True,
        cwd=plan.target,
        env=environment,
    )
    if result.returncode != 0:
        pytest.fail(f"The generated suite failed:\n{result.stdout}\n{result.stderr}")
    assert "passed" in result.stdout


def test_an_unsafe_slug_is_refused():
    with pytest.raises(ValueError, match="path and filename safe"):
        scaffold.plan_from(
            scaffold.parser().parse_args(["--name", "Bad", "--slug", "../escape"])
        )


def test_a_non_empty_target_is_not_overwritten_without_force(tmp_path):
    target = tmp_path / "demo"
    target.mkdir()
    (target / "keep.txt").write_text("mine", encoding="utf-8")
    plan = plan_for(target)
    with pytest.raises(FileExistsError, match="not empty"):
        scaffold.generate(plan)
    scaffold.generate(plan, force=True)
    assert (target / "keep.txt").read_text(encoding="utf-8") == "mine"
    assert (target / "pyproject.toml").is_file()
