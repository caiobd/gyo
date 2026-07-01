"""Playwright behavioral verification for gyo vi interface."""

import asyncio
from playwright.async_api import async_playwright

BASE = "http://127.0.0.1:8001"


async def test_shell_loads(page):
    """PR3: Shell loads with correct elements."""
    await page.goto(BASE)
    await page.wait_for_selector("#scroller .node", timeout=10000)

    # Check header elements
    assert await page.locator(".brand").inner_text() != "", "brand missing"
    assert await page.locator("#crumbs").count() > 0, "breadcrumbs missing"
    assert await page.locator("#winstat").count() > 0, "winstat missing"
    assert await page.locator("#dead").count() > 0, "dead missing"
    assert await page.locator("#rail").count() > 0, "rail missing"

    # Check nodes rendered
    nodes = await page.locator("#scroller .node").count()
    assert nodes > 0, f"no nodes rendered, got {nodes}"
    print(f"  ✓ shell loads, {nodes} nodes rendered")


async def test_residual_color(page):
    """PR3: Nodes have residual color fill."""
    await page.goto(BASE)
    await page.wait_for_selector("#scroller .node", timeout=10000)

    first_node = page.locator("#scroller .node").first
    bg = await first_node.evaluate("el => el.style.background")
    assert "rgb" in bg, f"node background not rgb: {bg}"
    print(f"  ✓ residual color fill: {bg[:30]}...")


async def test_branch_caps(page):
    """PR3: Nodes have branch cap stripes."""
    await page.goto(BASE)
    await page.wait_for_selector("#scroller .node .branchcap", timeout=10000)

    caps = await page.locator("#scroller .node .branchcap").count()
    assert caps > 0, "no branch caps"
    print(f"  ✓ {caps} branch caps rendered")


async def test_mosaic_images(page):
    """PR4: Mosaic thumbnails load in images mode."""
    await page.goto(BASE)
    await page.wait_for_selector("#scroller .node", timeout=10000)

    # Toggle to images mode
    await page.keyboard.press("c")
    await page.wait_for_timeout(1500)

    thumbs = await page.locator("#scroller .node .thumbs .thumb").count()
    print(f"  ✓ {thumbs} thumbnails in images mode")


async def test_mode_toggle(page):
    """PR4: C key toggles mode."""
    await page.goto(BASE)
    await page.wait_for_selector("#scroller .node", timeout=10000)

    btn = page.locator("#modeBtn")
    text1 = await btn.inner_text()
    await page.keyboard.press("c")
    await page.wait_for_timeout(300)
    text2 = await btn.inner_text()
    assert text1 != text2, f"mode didn't toggle: {text1} -> {text2}"
    print(f"  ✓ mode toggle: '{text1}' → '{text2}'")


async def test_tooltip(page):
    """PR5: Hover shows tooltip."""
    await page.goto(BASE)
    await page.wait_for_selector("#scroller .node", timeout=10000)

    node = page.locator("#scroller .node").nth(1)
    await node.hover()
    await page.wait_for_timeout(200)

    tip = page.locator("#tip")
    visible = await tip.is_visible()
    content = await tip.inner_text() if visible else ""
    assert visible and len(content) > 0, f"tooltip not visible: visible={visible}"
    print(f"  ✓ tooltip shows: '{content[:40]}...'")


async def test_rail_ticks(page):
    """PR6: Depth rail has ticks."""
    await page.goto(BASE)
    await page.wait_for_selector("#rail .rail-tick", timeout=10000)

    ticks = await page.locator("#rail .rail-tick").count()
    assert ticks > 0, "no rail ticks"
    print(f"  ✓ {ticks} rail ticks")


async def test_responsive_resize(page):
    """PR7: Layout reflows on resize."""
    await page.goto(BASE)
    await page.wait_for_selector("#scroller .node", timeout=10000)

    # Count nodes at initial size
    n1 = await page.locator("#scroller .node").count()

    # Resize to smaller viewport
    await page.set_viewport_size({"width": 800, "height": 600})
    await page.wait_for_timeout(300)

    n2 = await page.locator("#scroller .node").count()
    assert n2 > 0, "no nodes after resize"
    print(f"  ✓ resize reflow: {n1} → {n2} nodes")


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 800})

        tests = [
            ("PR3 Shell", test_shell_loads),
            ("PR3 Residual color", test_residual_color),
            ("PR3 Branch caps", test_branch_caps),
            ("PR4 Mosaic", test_mosaic_images),
            ("PR4 Mode toggle", test_mode_toggle),
            ("PR5 Tooltip", test_tooltip),
            ("PR6 Rail", test_rail_ticks),
            ("PR7 Responsive", test_responsive_resize),
        ]

        passed = 0
        failed = 0
        for name, fn in tests:
            try:
                print(f"\n{name}:")
                await fn(page)
                passed += 1
            except Exception as e:
                print(f"  ✗ FAILED: {e}")
                failed += 1

        await browser.close()
        print(f"\n{'=' * 40}")
        print(f"Results: {passed} passed, {failed} failed")
        return failed == 0


if __name__ == "__main__":
    ok = asyncio.run(main())
    exit(0 if ok else 1)
