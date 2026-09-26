#!/usr/bin/env python3
"""Create compact, deterministic journal-upload and LaTeX-source bundles.

Targets the Neural Networks manuscript (``main_nn.tex``). The filename keeps
its historical ``build_cviu_*`` prefix; the archived CVIU manuscript
(``main.tex``) is a different document and is not packaged here.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import tempfile
import zipfile
from pathlib import Path


FIXED_ZIP_TIME = (2026, 9, 3, 8, 0, 0)
MAIN_TEX = "main_nn.tex"
MAIN_PDF = "main_nn.pdf"
CORE_SOURCES = (
    MAIN_TEX,
    "supplement.tex",
    "references.bib",
    "cas-dc.cls",
    "cas-common.sty",
    "elsarticle-num.bst",
)
UPLOADS = {
    MAIN_PDF: "Main_Manuscript.pdf",
    "supplement.pdf": "Supplementary_File_S1.pdf",
    "graphical_abstract.tif": "Graphical_Abstract.tif",
    "Highlights.docx": "Highlights.docx",
    "Cover_Letter.docx": "Cover_Letter.docx",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publication_dependencies(root: Path) -> tuple[list[Path], list[Path]]:
    input_pattern = re.compile(r"\\input\{([^}]+)\}")
    image_pattern = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}")
    inputs: set[Path] = set()
    images: set[Path] = set()
    for source_name in (MAIN_TEX, "supplement.tex"):
        text = (root / source_name).read_text(encoding="utf-8")
        for match in input_pattern.findall(text):
            candidate = root / match
            if not candidate.is_file() and candidate.suffix == "":
                with_tex = candidate.with_suffix(".tex")
                if with_tex.is_file():
                    candidate = with_tex
            inputs.add(candidate)
        for match in image_pattern.findall(text):
            candidate = root / match
            if not candidate.is_file():
                candidate = root / "figures" / match
            images.add(candidate)
    missing = [path for path in (*inputs, *images) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing publication dependencies: {missing}")
    return sorted(inputs), sorted(images)


def write_deterministic_zip(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            relative = path.relative_to(source).as_posix()
            info = zipfile.ZipInfo(relative, FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)


def build_structured(root: Path, stage: Path, inputs: list[Path], images: list[Path]) -> None:
    for name in CORE_SOURCES:
        shutil.copy2(root / name, stage / name)
    (stage / "generated").mkdir()
    (stage / "figures").mkdir()
    for path in inputs:
        shutil.copy2(path, stage / "generated" / path.name)
    for path in images:
        shutil.copy2(path, stage / "figures" / path.name)
    (stage / "README_OVERLEAF.txt").write_text(
        "NEURAL NETWORKS OVERLEAF SOURCE\n"
        "===============================\n\n"
        "Main document: main_nn.tex\n"
        "Supplement: supplement.tex (compile separately)\n"
        "Compiler: pdfLaTeX; bibliography: BibTeX\n\n"
        "The source contains only files referenced by the two LaTeX documents.\n",
        encoding="utf-8",
    )


def flatten_tex(text: str) -> str:
    text = text.replace(r"\graphicspath{{figures/}}", r"\graphicspath{{./}}")
    return re.sub(r"\\input\{generated/([^}]+)\}", r"\\input{\1}", text)


def build_flat(root: Path, stage: Path, inputs: list[Path], images: list[Path]) -> None:
    for name in CORE_SOURCES:
        source = root / name
        destination = stage / name
        if source.suffix == ".tex":
            destination.write_text(flatten_tex(source.read_text(encoding="utf-8")), encoding="utf-8")
        else:
            shutil.copy2(source, destination)
    for path in (*inputs, *images):
        destination = stage / path.name
        if destination.exists():
            raise RuntimeError(f"flat archive basename collision: {path.name}")
        shutil.copy2(path, destination)
    (stage / "README_ELSEVIER_SOURCE.txt").write_text(
        "NEURAL NETWORKS FLAT LATEX SOURCE\n"
        "=================================\n\n"
        "All source dependencies are in this directory because Elsevier Editorial\n"
        "Manager may not process nested source directories. Compile main_nn.tex\n"
        "and supplement.tex separately with pdfLaTeX; main_nn.tex uses BibTeX.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-root", type=Path, default=Path("paper"))
    parser.add_argument("--output", type=Path, default=Path("NN_SUBMISSION_READY"))
    args = parser.parse_args()
    root = args.paper_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    for name in (*CORE_SOURCES, *UPLOADS):
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    inputs, images = publication_dependencies(root)

    with tempfile.TemporaryDirectory(prefix="nn-package-") as temporary:
        temp = Path(temporary)
        structured = temp / "overleaf"
        flat = temp / "elsevier-flat"
        structured.mkdir()
        flat.mkdir()
        build_structured(root, structured, inputs, images)
        build_flat(root, flat, inputs, images)
        output.mkdir(parents=True)
        for source_name, output_name in UPLOADS.items():
            shutil.copy2(root / source_name, output / output_name)
        write_deterministic_zip(structured, output / "Overleaf_Source.zip")
        write_deterministic_zip(flat, output / "Elsevier_Flat_LaTeX_Source.zip")

    readme = (
        "NEURAL NETWORKS SUBMISSION FILE MAP\n"
        "===================================\n\n"
        "Main_Manuscript.pdf              Manuscript\n"
        "Supplementary_File_S1.pdf        Supplementary material\n"
        "Graphical_Abstract.tif           Graphical abstract\n"
        "Highlights.docx                  Highlights\n"
        "Cover_Letter.docx                Cover letter\n"
        "Overleaf_Source.zip              Editable structured source\n"
        "Elsevier_Flat_LaTeX_Source.zip   Flat Editorial Manager source\n\n"
        "Prior archived release (CVIU-era v2.1.0): https://doi.org/10.5281/zenodo.22275640\n"
        "All-versions concept DOI: https://doi.org/10.5281/zenodo.22031663\n"
        "These source ZIPs are publication sources, not the research-evidence\n"
        "archive. Cover letter is a draft pending all-author approval and\n"
        "exclusivity/COI confirmation. Do not submit it before those\n"
        "author-controlled checks.\n"
    )
    (output / "README_UPLOAD.txt").write_text(readme, encoding="utf-8")
    checksums = []
    for path in sorted(item for item in output.iterdir() if item.is_file() and item.name != "SHA256SUMS.txt"):
        checksums.append(f"{sha256(path)}  {path.name}")
    (output / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(f"created compact submission directory: {output}")


if __name__ == "__main__":
    main()
