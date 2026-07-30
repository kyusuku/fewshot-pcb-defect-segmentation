from __future__ import annotations

from scripts.render_visual_big_picture import PAGE_SPECS, collect_step_numbers


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
