"""Lightweight virtual environment helper inspired by the project brief.

This module exposes a small CLI named ``moana`` which can be used to create
project-focused Python virtual environments.  The primary use-case in this
repository is the ``mini`` preset, which provisions a minimal environment with
video-gen installed in editable mode so that developers can quickly try the
workflow with the required dependencies.
"""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

__all__ = ["main", "plan_commands", "Preset", "PRESETS"]


@dataclass(frozen=True)
class Preset:
    """Describe how to bootstrap a specific environment preset."""

    name: str
    description: str
    packages_factory: Callable[[Path, Sequence[str]], Sequence[str]]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _pip_executable(env_path: Path) -> Path:
    bin_dir = env_path / ("Scripts" if os.name == "nt" else "bin")
    pip_name = "pip.exe" if os.name == "nt" else "pip"
    return bin_dir / pip_name


def _default_packages(project_root: Path, extras: Sequence[str]) -> Sequence[str]:
    extras_segment = ""
    if extras:
        extras_segment = f"[{','.join(sorted(set(extras)))}]"
    editable_target = f"{project_root}{extras_segment}"
    return ("-e", editable_target)


PRESETS = {
    "mini": Preset(
        name="mini",
        description="Minimal environment with video-gen installed in editable mode",
        packages_factory=lambda root, extras: _default_packages(root, extras),
    )
}


def plan_commands(
    *,
    preset: Preset,
    env_path: Path,
    python_executable: str,
    extras: Sequence[str],
    upgrade_deps: bool,
) -> List[List[str]]:
    """Return the subprocess commands required to create the environment."""

    commands: List[List[str]] = []
    venv_cmd = [python_executable, "-m", "venv", str(env_path)]
    if upgrade_deps:
        venv_cmd.append("--upgrade-deps")
    commands.append(venv_cmd)

    packages = preset.packages_factory(_project_root(), extras)
    if packages:
        pip_path = _pip_executable(env_path)
        install_cmd = [str(pip_path), "install"] + list(packages)
        commands.append(install_cmd)
    return commands


def _run_commands(commands: Iterable[Sequence[str]], *, dry_run: bool) -> None:
    for cmd in commands:
        display = " ".join(shlex.quote(part) for part in cmd)
        if dry_run:
            print(f"[dry-run] {display}")
            continue
        subprocess.check_call(cmd)


def _handle_create(args: argparse.Namespace) -> None:
    preset = PRESETS.get(args.preset)
    if not preset:
        available = ", ".join(sorted(PRESETS))
        raise SystemExit(f"Unknown preset '{args.preset}'. Available: {available}")

    env_path = Path(args.path).expanduser().resolve()
    if env_path.exists():
        if not args.force:
            raise SystemExit(
                f"Environment at {env_path} already exists. Use --force to overwrite."
            )
        if not args.dry_run:
            shutil.rmtree(env_path)

    commands = plan_commands(
        preset=preset,
        env_path=env_path,
        python_executable=args.python,
        extras=args.extras,
        upgrade_deps=args.upgrade_deps,
    )
    _run_commands(commands, dry_run=args.dry_run)
    if args.dry_run:
        return

    pip_path = _pip_executable(env_path)
    print(f"Environment ready at {env_path}")
    print(f"Activate it via: source {env_path}/bin/activate" if os.name != "nt" else f"Activate it via: {env_path}\\Scripts\\activate.bat")
    print(f"Installed packages can be managed with: {pip_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage project virtual environments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="Create a virtual environment")
    create_parser.add_argument("preset", choices=sorted(PRESETS), help="Environment preset to use")
    create_parser.add_argument(
        "--path",
        default=".venv-moana",
        help="Directory where the virtual environment will be created",
    )
    create_parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use for venv creation",
    )
    create_parser.add_argument(
        "--extras",
        nargs="*",
        default=(),
        help="Optional extras to install alongside the project (e.g. dev)",
    )
    create_parser.add_argument(
        "--upgrade-deps",
        action="store_true",
        help="Upgrade pip/setuptools/wheel inside the environment",
    )
    create_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the target directory if it already exists",
    )
    create_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without executing them",
    )
    create_parser.set_defaults(func=_handle_create)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
