from __future__ import annotations

import pytest

import apply_confirmatory_paper_evidence as apply_evidence


def test_replace_literal_abstract_preserves_document_and_strips_generated_comment() -> None:
    source = "before\n\\begin{abstract}\nold\n\\end{abstract}\nafter\n"
    generated = "% generated\nNew evidence abstract.\n"

    updated = apply_evidence.replace_literal_abstract(source, generated)

    assert updated == "before\n\\begin{abstract}\nNew evidence abstract.\n\\end{abstract}\nafter\n"


def test_replace_literal_abstract_refuses_missing_or_multiple_blocks() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        apply_evidence.replace_literal_abstract("no abstract", "new")
    duplicated = "\\begin{abstract}a\\end{abstract}" * 2
    with pytest.raises(ValueError, match="exactly one"):
        apply_evidence.replace_literal_abstract(duplicated, "new")
