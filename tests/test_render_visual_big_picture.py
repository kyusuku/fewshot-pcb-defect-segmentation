from __future__ import annotations

from pathlib import Path

from PIL import Image

from scripts.render_visual_big_picture import (
    PAGE_SPECS,
    collect_step_numbers,
    extract_real_assets,
    load_f1_table,
    validate_source_assets,
)


def test_page_registry_covers_fourteen_pages_and_all_steps_in_order() -> None:
    assert [page.number for page in PAGE_SPECS] == list(range(1, 15))
    assert collect_step_numbers(PAGE_SPECS) == list(range(1, 17))


def test_every_page_has_teaching_copy_and_source_kind() -> None:
    for page in PAGE_SPECS:
        assert page.title
        assert page.what_enters
        assert page.what_happens
        assert page.what_comes_out
        assert page.what_to_notice
        assert page.source_kind in {"real", "illustration", "mixed"}


def test_source_assets_exist_and_frozen_montages_match_manifest() -> None:
    validated = validate_source_assets()

    assert validated["pcb1_success"]
    assert validated["pcb1_failure"]
    assert all(validated[f"pcb{index}_normal"] for index in range(1, 5))


def test_real_asset_extraction_preserves_expected_panel_size(tmp_path: Path) -> None:
    assets = extract_real_assets(tmp_path)

    for name in (
        "success_anomaly",
        "success_guided",
        "success_ac",
        "failure_guided",
        "failure_ac",
    ):
        with Image.open(assets[name]) as image:
            assert image.size == (280, 280)
    assert Image.open(assets["success_query"]).size == (1404, 1070)
    assert Image.open(assets["success_ground_truth"]).size == (1404, 1070)


def test_f1_table_is_loaded_from_frozen_primary_summary() -> None:
    assert load_f1_table() == {
        "Multi-DINO": {1: 0.210, 2: 0.228, 4: 0.244},
        "Guided SAM2": {1: 0.257, 2: 0.284, 4: 0.303},
        "AC-SAM2": {1: 0.268, 2: 0.297, 4: 0.317},
    }
