"""Render user systemd units into this workspace; does not install or activate them."""

import sys
from pathlib import Path


def venv_python(project: Path) -> Path:
    """Resolve the project interpreter without assuming a POSIX venv layout."""
    environment = (project / ".venv").resolve()
    current = Path(sys.executable).resolve()
    if current.is_relative_to(environment):
        return current
    for candidate in (environment / "bin/python", environment / "Scripts/python.exe"):
        if candidate.exists():
            return candidate
    raise SystemExit("Run uv sync --locked before rendering the units")


def quote_path(path: Path, *, executable: bool = False) -> str:
    value = str(path).replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    if "\n" in value or "\r" in value:
        raise ValueError("Newlines are not supported in installation paths")
    if executable:
        value = value.replace("$", "$$")
    return f'"{value}"'


def render_units(
    project: Path, module: str, model_name: str, service_name: str, config: str
):
    project = project.resolve()
    deploy = project / "deploy"
    executable = venv_python(project)
    generated = deploy / "generated"
    generated.mkdir(exist_ok=True)
    # isprintable() is False for newlines and control characters, which would
    # otherwise split or corrupt the generated ExecStart line.
    if (
        not config
        or config != config.strip()
        or not config.isprintable()
        or '"' in config
    ):
        raise ValueError("Configuration path must be a single plain relative path")
    command = (
        f"{quote_path(executable, executable=True)} -m {module} "
        f"--config {config} --source ftp --scheduled"
    )
    service = f"""[Unit]
Description=Acquire GSMaP NOW and publish new {model_name} forecasts

[Service]
Type=oneshot
WorkingDirectory={str(project).replace("%", "%%")}
EnvironmentFile={str(project / ".env").replace("%", "%%")}
ExecStart={command}
TimeoutStartSec=20min
UMask=0077
"""
    (generated / f"{service_name}.service").write_text(service, encoding="utf-8")
    (generated / f"{service_name}.timer").write_text(
        (deploy / f"{service_name}.timer").read_text(encoding="utf-8"), encoding="utf-8"
    )
    print(f"Units prepared in {generated}; nothing has been activated")
