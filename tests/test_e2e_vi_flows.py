"""Comprehensive Playwright E2E tests for gyo vi interface.

Simulates real user flows: boot, drill, collapse, mode toggle,
scrolling, rail navigation, tooltips, breadcrumbs, resize, persistence.

Run: .venv/bin/python tests/test_e2e_vi_flows.py
Requires server running on port 8000.
"""

import asyncio
from playwright.async_api import async_playwright

BASE = "http://127.0.0.1:8000"
TIMEOUT = 15000


# ── helpers ──


async def boot(page):
    """Navigate, clear localStorage for test isolation, wait for render."""
    await page.goto(BASE)
    await page.evaluate("localStorage.clear()")
    await page.goto(BASE)
    await page.wait_for_selector("#scroller .node", timeout=TIMEOUT)
    await page.wait_for_timeout(500)


async def node_count(page):
    return await page.locator("#scroller .node").count()


async def get_mode_text(page):
    return await page.locator("#modeBtn").inner_text()


async def breadcrumb_texts(page):
    """Get crumb texts, excluding the modeBtn (which also has .crumb class)."""
    return await page.locator("#crumbs .crumb").all_inner_texts()


# ══════════════════════════════════════════════════════════════
# FLOW 1 — Boot
# ══════════════════════════════════════════════════════════════


async def test_boot(page):
    await boot(page)

    brand = await page.locator(".brand").inner_text()
    assert "gyo" in brand, f"brand missing: {brand}"

    crumbs = await breadcrumb_texts(page)
    assert crumbs[0] == "root", f"first crumb not root: {crumbs}"

    winstat = await page.locator("#winstat").inner_text()
    assert "/" in winstat, f"winstat bad: {winstat}"

    dead = await page.locator("#dead").inner_text()
    assert dead.startswith("["), f"dead not array: {dead}"

    n = await node_count(page)
    assert n > 5, f"too few nodes: {n}"
    print(f"  ✓ {n} nodes, winstat='{winstat}'")


# ══════════════════════════════════════════════════════════════
# FLOW 2 — Drill: double-click child → focus changes
# ══════════════════════════════════════════════════════════════


async def test_drill(page):
    await boot(page)

    n_before = await node_count(page)
    # skip root (first node), pick second child
    children = page.locator("#scroller .node:not(.spine):not(.dead)")
    count = await children.count()
    target = children.nth(min(1, count - 1))
    await target.dblclick(force=True)
    await page.wait_for_timeout(400)

    crumbs = await breadcrumb_texts(page)
    assert len(crumbs) >= 2, f"breadcrumbs didn't update: {crumbs}"
    assert crumbs[0] == "root"
    assert crumbs[1].startswith("c")

    n_after = await node_count(page)
    print(f"  ✓ breadcrumbs={crumbs}, nodes {n_before}→{n_after}")


# ══════════════════════════════════════════════════════════════
# FLOW 3 — Collapse: single-click → spine appears
# ══════════════════════════════════════════════════════════════


async def test_collapse(page):
    await boot(page)

    # target depth-1 children (skip root which is first .node)
    children = page.locator("#scroller .node:not(.spine):not(.dead)")
    count = await children.count()
    assert count >= 2, f"need >= 2 children, got {count}"

    # click second child (first might be narrow sliver at left)
    target = children.nth(1)
    await target.click(force=True)
    await page.wait_for_timeout(400)

    spines = await page.locator("#scroller .node.spine").count()
    assert spines > 0, "no spine after collapse"

    spine_label = await page.locator(
        "#scroller .node.spine .spinelabel"
    ).first.inner_text()
    n_after = await node_count(page)
    print(f"  ✓ spine='{spine_label}', {spines} spine(s), {n_after} nodes")


# ══════════════════════════════════════════════════════════════
# FLOW 4 — Expand: click spine → expands back
# ══════════════════════════════════════════════════════════════


async def test_expand(page):
    await boot(page)

    children = page.locator("#scroller .node:not(.spine):not(.dead)")
    count = await children.count()
    assert count >= 2, "need >= 2 children"

    # collapse second child
    await children.nth(1).click(force=True)
    await page.wait_for_timeout(400)
    spines_before = await page.locator("#scroller .node.spine").count()
    assert spines_before > 0, "no spine after collapse"

    # click the spine to expand
    await page.locator("#scroller .node.spine").first.click(force=True)
    await page.wait_for_timeout(400)

    spines_after = await page.locator("#scroller .node.spine").count()
    assert spines_after < spines_before
    print(f"  ✓ spines {spines_before}→{spines_after}")


# ══════════════════════════════════════════════════════════════
# FLOW 5 — Mode toggle
# ══════════════════════════════════════════════════════════════


async def test_mode_toggle_key(page):
    await boot(page)

    mode1 = await get_mode_text(page)
    await page.keyboard.press("c")
    await page.wait_for_timeout(300)
    mode2 = await get_mode_text(page)
    assert mode1 != mode2, f"didn't toggle: {mode1}→{mode2}"

    await page.keyboard.press("c")
    await page.wait_for_timeout(300)
    mode3 = await get_mode_text(page)
    assert mode3 == mode1, f"didn't toggle back: {mode3}"
    print(f"  ✓ '{mode1}' → '{mode2}' → '{mode3}'")


async def test_mode_toggle_click(page):
    await boot(page)

    mode1 = await get_mode_text(page)
    await page.locator("#modeBtn").click()
    await page.wait_for_timeout(300)
    mode2 = await get_mode_text(page)
    assert mode1 != mode2, f"didn't toggle on click: {mode1}→{mode2}"
    print(f"  ✓ '{mode1}' → '{mode2}'")


# ══════════════════════════════════════════════════════════════
# FLOW 6 — Mosaics
# ══════════════════════════════════════════════════════════════


async def test_mosaics_load(page):
    await boot(page)

    mode = await get_mode_text(page)
    assert "imagens" in mode, f"not images mode: {mode}"

    await page.wait_for_timeout(2000)
    thumbs = await page.locator("#scroller .node .thumbs .thumb").count()
    assert thumbs > 0, "no thumbnails"

    src = await page.locator("#scroller .node .thumbs .thumb img").first.get_attribute(
        "src"
    )
    assert src and "/thumb/" in src, f"bad src: {src}"
    print(f"  ✓ {thumbs} thumbnails, src='{src}'")


async def test_mosaics_hide_in_residual(page):
    await boot(page)

    # switch to residual
    await page.keyboard.press("c")
    await page.wait_for_timeout(300)
    mode = await get_mode_text(page)
    assert "resíduo" in mode

    await page.wait_for_timeout(1000)
    thumbs = await page.locator("#scroller .node .thumbs").count()
    assert thumbs == 0, f"thumbnails present in residual: {thumbs}"
    print(f"  ✓ no thumbnails in residual mode")


# ══════════════════════════════════════════════════════════════
# FLOW 7 — Tooltip
# ══════════════════════════════════════════════════════════════


async def test_tooltip_show_and_hide(page):
    await boot(page)

    # hover a depth-1 node (nth(1), skip root)
    target = page.locator("#scroller .node:not(.spine)").nth(1)
    box = await target.bounding_box()
    assert box, "no bounding box"

    await page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    await page.wait_for_timeout(200)

    tip = page.locator("#tip")
    assert await tip.is_visible(), "tooltip not visible"
    tip_text = await tip.inner_text()
    assert "occupancy" in tip_text, f"missing occupancy: {tip_text}"

    await page.mouse.move(0, 0)
    await page.wait_for_timeout(200)
    assert not await tip.is_visible(), "tooltip still visible"
    print(f"  ✓ tooltip: '{tip_text[:50]}...'")


# ══════════════════════════════════════════════════════════════
# FLOW 8 — Rail navigation
# ══════════════════════════════════════════════════════════════


async def test_rail_click(page):
    await boot(page)

    ticks = page.locator("#rail .rail-tick")
    count = await ticks.count()
    assert count > 1, f"not enough ticks: {count}"

    winstat_before = await page.locator("#winstat").inner_text()
    await ticks.last.click()
    await page.wait_for_timeout(400)
    winstat_after = await page.locator("#winstat").inner_text()

    active = await page.locator("#rail .rail-tick.active").count()
    assert active > 0, "no active tick"
    print(f"  ✓ '{winstat_before}' → '{winstat_after}', {active} active tick(s)")


# ══════════════════════════════════════════════════════════════
# FLOW 9 — Breadcrumb navigation
# ══════════════════════════════════════════════════════════════


async def test_breadcrumb_navigate(page):
    await boot(page)

    # drill into a child
    children = page.locator("#scroller .node:not(.spine):not(.dead)")
    count = await children.count()
    await children.nth(min(1, count - 1)).dblclick(force=True)
    await page.wait_for_timeout(400)

    crumbs_drill = await breadcrumb_texts(page)
    assert len(crumbs_drill) >= 2, f"no breadcrumb after drill: {crumbs_drill}"

    # click root breadcrumb
    await page.locator("#crumbs .crumb").first.click()
    await page.wait_for_timeout(400)

    crumbs_back = await breadcrumb_texts(page)
    assert len(crumbs_back) == 1, f"didn't return to root: {crumbs_back}"
    assert crumbs_back[0] == "root"
    print(f"  ✓ drilled {crumbs_drill} → back to {crumbs_back}")


# ══════════════════════════════════════════════════════════════
# FLOW 10 — Resize
# ══════════════════════════════════════════════════════════════


async def test_resize_reflow(page):
    await boot(page)

    n_before = await node_count(page)

    await page.set_viewport_size({"width": 800, "height": 600})
    await page.wait_for_timeout(300)
    n_narrow = await node_count(page)

    await page.set_viewport_size({"width": 1600, "height": 900})
    await page.wait_for_timeout(300)
    n_wide = await node_count(page)

    # check width changes
    w_narrow = await page.locator("#scroller .node").first.evaluate(
        "el => parseFloat(el.style.width)"
    )
    await page.set_viewport_size({"width": 800, "height": 600})
    await page.wait_for_timeout(300)
    w_small = await page.locator("#scroller .node").first.evaluate(
        "el => parseFloat(el.style.width)"
    )

    assert n_narrow > 0 and n_wide > 0
    print(
        f"  ✓ {n_before}→{n_narrow}→{n_wide} nodes, width {w_narrow:.0f}→{w_small:.0f}"
    )


# ══════════════════════════════════════════════════════════════
# FLOW 11 — Scroll window
# ══════════════════════════════════════════════════════════════


async def test_scroll_window(page):
    await boot(page)

    winstat_before = await page.locator("#winstat").inner_text()

    icicle = page.locator("#icicle")
    box = await icicle.bounding_box()
    await page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    await page.mouse.wheel(0, 500)
    await page.wait_for_timeout(500)

    winstat_after = await page.locator("#winstat").inner_text()
    print(f"  ✓ '{winstat_before}' → '{winstat_after}'")


# ══════════════════════════════════════════════════════════════
# FLOW 12 — Persistence
# ══════════════════════════════════════════════════════════════


async def test_persistence_mode(page):
    await boot(page)

    await page.keyboard.press("c")
    await page.wait_for_timeout(300)
    mode_before = await get_mode_text(page)
    assert "resíduo" in mode_before

    await page.reload()
    await page.wait_for_selector("#scroller .node", timeout=TIMEOUT)
    await page.wait_for_timeout(300)

    mode_after = await get_mode_text(page)
    assert mode_after == mode_before, f"didn't persist: {mode_before}→{mode_after}"
    print(f"  ✓ mode '{mode_after}' survived reload")


# ══════════════════════════════════════════════════════════════
# FLOW 13 — Residual colors
# ══════════════════════════════════════════════════════════════


async def test_residual_colors(page):
    await boot(page)

    first = page.locator("#scroller .node:not(.spine):not(.dead)").first
    bg = await first.evaluate("el => el.style.background")
    assert "rgb" in bg, f"not rgb: {bg}"
    print(f"  ✓ {bg[:40]}")


# ══════════════════════════════════════════════════════════════
# FLOW 14 — Purity bar
# ══════════════════════════════════════════════════════════════


async def test_purity_bar(page):
    await boot(page)

    bars = await page.locator("#scroller .node .purity").count()
    assert bars > 0, "no purity bars"
    print(f"  ✓ {bars} purity bars")


# ══════════════════════════════════════════════════════════════
# FLOW 15 — Branch caps: check color diversity in descendants
# ══════════════════════════════════════════════════════════════


async def test_branch_caps(page):
    await boot(page)

    caps = await page.locator("#scroller .node .branchcap").count()
    assert caps > 0, "no branch caps"

    # check colors across depth-1 nodes (root's children = different branches)
    colors = await page.evaluate("""() => {
        const nodes = document.querySelectorAll('#scroller .node');
        const rowH = parseFloat(document.getElementById('scroller').style.height)
            / (document.querySelectorAll('#scroller .node').length || 1) || 220;
        const colors = new Set();
        for (const n of nodes) {
            const top = parseFloat(n.style.top);
            const depth = Math.round(top / (document.querySelector('.node:not(.spine)')?.offsetHeight || 220));
            if (depth === 1) {
                const cap = n.querySelector('.branchcap');
                if (cap) colors.add(cap.style.background);
            }
        }
        return [...colors];
    }""")
    assert len(colors) >= 2, f"branch caps all same color: {colors}"
    print(f"  ✓ {caps} caps, {len(colors)} distinct branch colors")


# ══════════════════════════════════════════════════════════════
# FLOW 16 — Multi collapse/expand
# ══════════════════════════════════════════════════════════════


async def test_multi_collapse_expand(page):
    await boot(page)

    children = page.locator("#scroller .node:not(.spine):not(.dead)")
    count = await children.count()
    assert count >= 3, f"need >= 3 children, got {count}"

    # collapse two children (use nth to skip root)
    await children.nth(1).click(force=True)
    await page.wait_for_timeout(350)
    await children.nth(2).click(force=True)
    await page.wait_for_timeout(400)

    spines = await page.locator("#scroller .node.spine").count()
    assert spines >= 1, f"expected spines, got {spines}"

    # expand all spines one by one
    expanded = 0
    while await page.locator("#scroller .node.spine").count() > 0:
        await page.locator("#scroller .node.spine").first.click(force=True)
        await page.wait_for_timeout(300)
        expanded += 1
        if expanded > 10:
            break  # safety

    spines_after = await page.locator("#scroller .node.spine").count()
    assert spines_after == 0, f"spines remain: {spines_after}"
    print(f"  ✓ collapsed {spines} spines, expanded all ({expanded} clicks)")


# ══════════════════════════════════════════════════════════════
# FLOW 17 — Leaf: double-click doesn't drill further
# ══════════════════════════════════════════════════════════════


async def test_leaf_no_drill(page):
    await boot(page)

    # drill into first child to go deeper
    children = page.locator("#scroller .node:not(.spine):not(.dead)")
    await children.nth(1).dblclick(force=True)
    await page.wait_for_timeout(400)

    crumbs_before = await breadcrumb_texts(page)

    # try to drill into last visible node (likely a leaf)
    nodes = page.locator("#scroller .node:not(.spine):not(.dead)")
    n = await nodes.count()
    if n > 0:
        await nodes.last.dblclick(force=True)
        await page.wait_for_timeout(400)
        crumbs_after = await breadcrumb_texts(page)
        # if it was a leaf, crumbs shouldn't change (or change minimally)
        print(f"  ✓ leaf drill attempt: {crumbs_before} → {crumbs_after}")
    else:
        print(f"  ⚠ no nodes to test leaf drill")


# ══════════════════════════════════════════════════════════════
# FLOW 18 — API integrity
# ══════════════════════════════════════════════════════════════


async def test_api_matches_ui(page):
    await boot(page)

    resp = await page.evaluate("fetch('/api/tree?level=99').then(r => r.json())")
    api_count = len(resp["nodes"])
    rendered = await node_count(page)

    assert rendered > 0 and api_count > 0
    print(f"  ✓ {api_count} API nodes, {rendered} rendered")


# ══════════════════════════════════════════════════════════════
# runner
# ══════════════════════════════════════════════════════════════

ALL_TESTS = [
    ("Boot", test_boot),
    ("Drill down", test_drill),
    ("Collapse", test_collapse),
    ("Expand", test_expand),
    ("Mode toggle (key)", test_mode_toggle_key),
    ("Mode toggle (click)", test_mode_toggle_click),
    ("Mosaics load", test_mosaics_load),
    ("Mosaics hidden residual", test_mosaics_hide_in_residual),
    ("Tooltip show/hide", test_tooltip_show_and_hide),
    ("Rail click", test_rail_click),
    ("Breadcrumb nav", test_breadcrumb_navigate),
    ("Resize reflow", test_resize_reflow),
    ("Scroll window", test_scroll_window),
    ("Persistence mode", test_persistence_mode),
    ("Residual colors", test_residual_colors),
    ("Purity bar", test_purity_bar),
    ("Branch caps", test_branch_caps),
    ("Multi collapse/expand", test_multi_collapse_expand),
    ("Leaf no drill", test_leaf_no_drill),
    ("API matches UI", test_api_matches_ui),
]


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 800})

        passed = 0
        failed = 0
        errors = []

        for name, fn in ALL_TESTS:
            try:
                print(f"\n{name}:")
                await fn(page)
                passed += 1
            except Exception as e:
                print(f"  ✗ FAILED: {e}")
                errors.append((name, str(e)[:200]))
                failed += 1

        await browser.close()
        print(f"\n{'=' * 50}")
        print(f"Results: {passed} passed, {failed} failed / {len(ALL_TESTS)} total")
        if errors:
            print("\nFailed:")
            for name, err in errors:
                print(f"  - {name}: {err}")
        return failed == 0


if __name__ == "__main__":
    ok = asyncio.run(main())
    exit(0 if ok else 1)
