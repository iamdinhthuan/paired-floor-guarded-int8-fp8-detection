#!/usr/bin/env python3
"""Fail-closed verification for a flat Editorial Manager source archive.

This is intentionally independent of ``submission_audit``.  Task 6 will wire it
to the final package once that package and its evidence exist.
"""

from __future__ import annotations

import re
import shutil
import stat
import subprocess
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath


LatexRunner = Callable[[list[str], Path], subprocess.CompletedProcess[str]]
_INPUT_COMMAND_RE = re.compile(
    r"\\input(?![A-Za-z@])\s*"
    r"(?P<target>\{[^{}]*\}|[^\s%{}\\]+)?"
)
_INCLUDE_COMMAND_RE = re.compile(
    r"\\include(?![A-Za-z@])\s*(?P<target>\{[^{}]*\})?"
)
_GRAPHICS_COMMAND_RE = re.compile(
    r"\\includegraphics\*?(?![A-Za-z@])\s*"
    r"(?:\[[^]]*\]\s*)?(?P<target>\{[^{}]*\})?"
)
_GRAPHIC_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg")
_TEXMF_VARIABLES = ("TEXMFROOT", "TEXMFDIST", "TEXMFLOCAL", "TEXMFVAR", "TEXMFCONFIG")


def _strip_tex_comments(tex: str) -> str:
    """Remove unescaped TeX comments while preserving line boundaries."""
    lines: list[str] = []
    for line in tex.splitlines(keepends=True):
        for index, character in enumerate(line):
            if character != "%":
                continue
            backslashes = 0
            before = index - 1
            while before >= 0 and line[before] == "\\":
                backslashes += 1
                before -= 1
            if backslashes % 2 == 0:
                lines.append(line[:index] + ("\n" if line.endswith("\n") else ""))
                break
        else:
            lines.append(line)
    return "".join(lines)


def _validate_member_name(name: str) -> str:
    """Return the extraction identity after rejecting every non-flat spelling."""
    if not name or name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        raise ValueError(f"absolute ZIP member is not allowed: {name!r}")
    if ".." in PurePosixPath(name.replace("\\", "/")).parts:
        raise ValueError(f"traversal ZIP member is not allowed: {name!r}")
    if "/" in name or "\\" in name:
        raise ValueError(f"ZIP member must be flat: {name!r}")
    if name in {".", ".."}:
        raise ValueError(f"traversal ZIP member is not allowed: {name!r}")
    return name


def _member_type_error(info: zipfile.ZipInfo) -> str | None:
    """Return why a ZIP member is not an ordinary extractable file, if so."""
    if info.is_dir() or info.filename.endswith("/"):
        return "directory"
    if info.create_system == 3:
        kind = stat.S_IFMT(info.external_attr >> 16)
        if kind not in (0, stat.S_IFREG):
            return {
                stat.S_IFDIR: "directory",
                stat.S_IFLNK: "symlink",
            }.get(kind, "special file")
    if info.create_system == 0 and info.external_attr & 0x10:
        return "directory"
    return None


def _validate_members(bundle: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    members: dict[str, zipfile.ZipInfo] = {}
    for info in bundle.infolist():
        invalid_type = _member_type_error(info)
        if invalid_type is not None:
            raise ValueError(
                f"non-regular {invalid_type} ZIP member is not allowed: {info.filename!r}"
            )
        identity = _validate_member_name(info.filename)
        if identity in members:
            raise ValueError(f"duplicate ZIP member: {info.filename!r}")
        members[identity] = info
    return members


def _validate_reference(target: str, *, kind: str) -> str:
    """Return a root-member target or reject an unsafe TeX path."""
    value = target.strip()
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if not value or normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        raise ValueError(f"absolute {kind} target is not allowed: {target!r}")
    if ":" in value:
        raise ValueError(f"external {kind} target is not allowed: {target!r}")
    if ".." in path.parts:
        raise ValueError(f"traversal {kind} target is not allowed: {target!r}")
    if "/" in normalized or len(path.parts) != 1 or path.parts[0] in {"", "."}:
        raise ValueError(f"{kind} target must be flat: {target!r}")
    return path.name


def _command_target(match: re.Match[str], *, kind: str) -> str:
    raw = match.group("target")
    if raw is None:
        raise ValueError(f"unsupported or missing {kind} target")
    if raw.startswith("{"):
        return raw[1:-1]
    return raw


def _input_member(target: str, members: dict[str, zipfile.ZipInfo]) -> str:
    name = _validate_reference(target, kind="TeX input")
    candidate = name if Path(name).suffix else f"{name}.tex"
    if candidate not in members:
        raise ValueError(f"missing TeX input member: {candidate!r}")
    return candidate


def _check_graphics(tex: str, members: dict[str, zipfile.ZipInfo]) -> None:
    for match in _GRAPHICS_COMMAND_RE.finditer(tex):
        name = _validate_reference(_command_target(match, kind="graphic"), kind="graphic")
        candidates = (
            (name,)
            if Path(name).suffix
            else tuple(f"{name}{extension}" for extension in _GRAPHIC_EXTENSIONS)
        )
        if not any(candidate in members for candidate in candidates):
            raise ValueError(f"missing graphic member for target: {name!r}")


def _check_tex_dependencies(
    bundle: zipfile.ZipFile, members: dict[str, zipfile.ZipInfo]
) -> None:
    """Recursively validate every archive-owned TeX input without following cycles."""
    visited: set[str] = set()

    def check(member: str) -> None:
        if member in visited:
            return
        visited.add(member)
        try:
            tex = bundle.read(members[member]).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"TeX input {member!r} must be UTF-8 text") from error
        active_tex = _strip_tex_comments(tex)
        inputs = [
            _input_member(_command_target(match, kind="TeX input"), members)
            for match in _INPUT_COMMAND_RE.finditer(active_tex)
        ]
        for match in _INCLUDE_COMMAND_RE.finditer(active_tex):
            if match.group("target") is None:
                raise ValueError("unbraced TeX include is not supported")
            inputs.append(
                _input_member(_command_target(match, kind="TeX input"), members)
            )
        _check_graphics(active_tex, members)
        for input_member in inputs:
            check(input_member)

    check("main.tex")


def _extract_flat_archive(
    bundle: zipfile.ZipFile, members: dict[str, zipfile.ZipInfo], root: Path
) -> None:
    root_resolved = root.resolve()
    for name, info in members.items():
        destination = root / name
        if not destination.resolve().is_relative_to(root_resolved):
            raise ValueError(f"extracted ZIP member escapes root: {name!r}")
        with bundle.open(info) as source, destination.open("wb") as output:
            shutil.copyfileobj(source, output)


def _texmf_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    for variable in _TEXMF_VARIABLES:
        try:
            result = subprocess.run(
                ["kpsewhich", f"-var-value={variable}"],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            continue
        if result.returncode == 0 and result.stdout.strip():
            roots.append(Path(result.stdout.strip()).resolve())
    return tuple(roots)


def _audit_recorder(root: Path) -> None:
    """Reject compilation inputs outside the extraction root or trusted TeX trees."""
    recorder = root / "main.fls"
    if not recorder.is_file():
        raise ValueError("LaTeX compilation did not produce recorder output")
    try:
        lines = recorder.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("LaTeX recorder output is not UTF-8 text") from error
    root = root.resolve()
    texmf_roots = _texmf_roots()
    has_root_main_input = False
    for line in lines:
        if not line.startswith("INPUT "):
            continue
        recorded = line.removeprefix("INPUT ")
        candidate = Path(recorded)
        resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        if resolved.is_relative_to(root):
            if resolved == root / "main.tex":
                has_root_main_input = True
            continue
        if any(resolved.is_relative_to(texmf_root) for texmf_root in texmf_roots):
            continue
        raise ValueError(f"LaTeX recorder loaded input outside extraction root: {recorded!r}")
    if not has_root_main_input:
        raise ValueError("LaTeX recorder must record root main.tex input")


def _run_latexmk(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)


def verify_flat_source_archive(
    archive: Path,
    *,
    latexmk_path: str | Path | None = None,
    runner: LatexRunner | None = None,
) -> None:
    """Verify a flat source ZIP and independently compile it when TeX is available.

    ``runner`` accepts ``(command, cwd)`` and is provided for deterministic tests.
    If no explicit ``latexmk_path`` is supplied, compilation is skipped only when
    ``latexmk`` cannot be found on ``PATH``.  Real compilation uses TeX recorder
    output to reject files read from outside the extracted package.
    """
    archive = Path(archive)
    if not archive.is_file():
        raise ValueError(f"flat source archive is missing: {archive}")

    try:
        with zipfile.ZipFile(archive) as bundle:
            members = _validate_members(bundle)
            required = {"main.tex", "references.bib"}
            missing = sorted(required - set(members))
            if missing:
                raise ValueError(f"flat source archive is missing required members: {missing}")
            _check_tex_dependencies(bundle, members)

            with tempfile.TemporaryDirectory(prefix="flat-source-") as directory:
                extracted_root = Path(directory)
                _extract_flat_archive(bundle, members, extracted_root)
                executable = str(latexmk_path) if latexmk_path is not None else shutil.which("latexmk")
                if executable is None:
                    return
                command = [
                    executable,
                    "-pdf",
                    "-recorder",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "main.tex",
                ]
                try:
                    result = (runner or _run_latexmk)(command, extracted_root)
                except OSError as error:
                    raise ValueError(f"could not start LaTeX compilation: {error}") from error
                if result.returncode != 0:
                    output = "\n".join(
                        part for part in (result.stdout, result.stderr) if part
                    ).strip()
                    raise ValueError(
                        "LaTeX compilation failed" + (f": {output}" if output else "")
                    )
                _audit_recorder(extracted_root)
    except zipfile.BadZipFile as error:
        raise ValueError(f"invalid flat source ZIP: {archive}") from error
