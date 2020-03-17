import bluesky.plans as bp


def test_legacy_plan_names():
    assert bp.outer_product_scan is bp.grid_scan
    assert bp.relative_outer_product_scan is bp.rel_grid_scan
    assert bp.relative_scan is bp.rel_scan
    assert bp.relative_spiral is bp.rel_spiral
    assert bp.relative_spiral_fermat is bp.rel_spiral_fermat
    assert bp.relative_list_scan is bp.rel_list_scan
    assert bp.relative_log_scan is bp.rel_log_scan
    assert bp.relative_adaptive_scan is bp.rel_adaptive_scan
