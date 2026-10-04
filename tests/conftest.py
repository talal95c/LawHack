import json
from pathlib import Path

import pytest

from lawhack.ingest import load_pdf, load_text

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


@pytest.fixture
def judilibre_doc():
    row = json.loads((SAMPLES / "cass_civ2_2025-03-13_23-13219.json").read_text())
    return load_text(row["text"]), row


@pytest.fixture
def legifrance_doc():
    return load_pdf(SAMPLES / "cass_civ3_2022-12-14_21-24539.pdf")


@pytest.fixture
def registry(legifrance_doc):
    import asyncio

    from fakes import ZoneJev
    from lawhack.attributor import build_registry
    from lawhack.segmenter import segment
    from lawhack.zoning import zone_blocks

    return asyncio.run(build_registry("doc", segment(zone_blocks(legifrance_doc)), ZoneJev()))
