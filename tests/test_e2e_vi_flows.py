"""Self-contained browser verification for the Semantic Atlas.

Run with ``uv run python tests/test_e2e_vi_flows.py``.  This is deliberately a
standalone runner (and not a pytest module) so the Python suite needs no
pytest-playwright ``page`` fixture.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import socket
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import uvicorn
from PIL import Image, ImageDraw
from playwright.async_api import Page, async_playwright

TIMEOUT = 15_000
__test__ = False


def seed_run(run_dir: Path) -> None:
    """Create a small canonical two-level run with real PNG samples."""
    image_dir = run_dir / "images"
    image_dir.mkdir(parents=True)
    item_ids = [101, 205, 309, 412, 518, 623, 734, 845]
    paths: list[str] = []
    for position, item_id in enumerate(item_ids):
        path = f"sample-{item_id}.png"
        image = Image.new("RGB", (72, 72), (30 + position * 22, 60, 180 - position * 12))
        draw = ImageDraw.Draw(image)
        draw.rectangle((8 + position, 8, 54, 54), outline=(255, 230, 80), width=4)
        draw.text((22, 27), str(position), fill="white")
        image.save(image_dir / path)
        paths.append(path)

    pd.DataFrame({
        "idx": item_ids,
        "path": paths,
        "label": ["amber", "amber", "blue", "blue", "coral", "coral", "fern", "fern"],
    }).to_parquet(run_dir / "meta.parquet", index=False)
    codes = np.asarray([[2, 0], [2, 1], [2, 1], [2, 2], [5, 0], [5, 2], [7, 1], [7, 2]])
    pd.DataFrame({
        "idx": item_ids,
        "c_0": codes[:, 0],
        "c_1": codes[:, 1],
        "j": np.zeros(len(item_ids), dtype=int),
        "r_0": np.linspace(1.4, .7, len(item_ids)),
        "r_1": np.linspace(.8, .2, len(item_ids)),
        "final_residual": np.linspace(.08, .71, len(item_ids)),
    }).to_parquet(run_dir / "codes.parquet", index=False)

    embeddings = np.asarray([
        [-1.8, -.8], [-1.5, -.1], [-1.0, .4], [-.6, .9],
        [.5, -.9], [.9, -.2], [1.4, .5], [1.9, 1.0],
    ], dtype=np.float32)
    np.save(run_dir / "embeddings.npy", embeddings)
    codebook_dir = run_dir / "codebooks" / "v1"
    codebook_dir.mkdir(parents=True)
    level_0 = np.zeros((8, 2), dtype=np.float32)
    level_0[2], level_0[5], level_0[7] = (-1.2, 0), (.3, -.5), (1.4, .5)
    level_1 = np.asarray([[-.4, -.3], [0, .35], [.45, .1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.float32)
    np.save(codebook_dir / "level_0.npy", level_0)
    np.save(codebook_dir / "level_1.npy", level_1)
    (codebook_dir / "config.json").write_text(json.dumps({
        "num_levels": 2, "codebook_size": 8, "dim": 2, "proj_dim": None, "seed": 7,
    }))


def serve(run_dir: str, port: int) -> None:
    from gyo.api.server import create_app
    uvicorn.run(create_app(run_dir), host="127.0.0.1", port=port, log_level="warning")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def stop_server(process: multiprocessing.Process) -> None:
    if process.is_alive():
        process.terminate()
    process.join(timeout=5)
    if process.is_alive():
        process.kill()
        process.join(timeout=2)


def wait_until_ready(base: str, process: multiprocessing.Process, timeout: float = 8) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if not process.is_alive():
            raise RuntimeError(f"Atlas server exited early ({process.exitcode})")
        try:
            with urllib.request.urlopen(f"{base}/api/tree", timeout=.5) as response:
                if response.status == 200:
                    return
        except Exception as exc:  # server may still be binding
            last_error = exc
        time.sleep(.05)
    raise RuntimeError(f"Atlas server did not become ready: {last_error}")


def start_server(run_dir: Path, attempts: int = 4) -> tuple[multiprocessing.Process, str]:
    """Start on a dynamically allocated port, retrying the unavoidable bind race."""
    diagnostics: list[str] = []
    context = multiprocessing.get_context("spawn")
    for attempt in range(1, attempts + 1):
        port = free_port()
        base = f"http://127.0.0.1:{port}"
        process = context.Process(target=serve, args=(str(run_dir), port), daemon=True)
        process.start()
        try:
            wait_until_ready(base, process)
            return process, base
        except Exception as exc:
            diagnostics.append(
                f"attempt {attempt}/{attempts} on port {port}: {type(exc).__name__}: {exc} "
                f"(exit={process.exitcode})"
            )
            stop_server(process)
    raise RuntimeError("Atlas server startup failed:\n" + "\n".join(diagnostics))


async def boot(page: Page, base: str) -> None:
    await page.goto(base, wait_until="domcontentloaded")
    await page.locator("#atlas .territory").first.wait_for(timeout=TIMEOUT)
    await page.locator("#mapLoading").wait_for(state="hidden", timeout=TIMEOUT)


async def flow_boot(page: Page, base: str) -> None:
    await boot(page, base)
    assert await page.locator("#atlas .territory").count() >= 3
    assert await page.locator("#atlas .focus-boundary").count() == 1
    assert await page.locator("#atlas .focus-anchor").count() == 1
    assert await page.locator("#atlas .hierarchy-link").count() == await page.locator("#atlas .territory").count()
    target = page.locator("#atlas .territory").first
    await target.hover()
    assert await target.evaluate("node => node.classList.contains('is-path')")
    assert await page.locator("#atlas .focus-anchor").evaluate("node => node.classList.contains('is-path')")
    assert "gyo" in (await page.locator(".brand").inner_text()).lower()
    assert "Layout stress" in await page.locator("#projectionStatus").inner_text()
    assert await page.locator("#mapError").is_hidden()


async def select_internal(page: Page):
    territory = page.locator('#atlas .territory[data-prefix="2"]')
    circle = territory.locator("circle")
    await circle.wait_for(state="visible", timeout=TIMEOUT)
    box = await circle.bounding_box()
    assert box and box["width"] > 4 and box["height"] > 4, "internal territory has no hittable circle"
    # Use the same physical pointer path as a user; synthetic dispatch would
    # hide pointer-capture and hit-testing regressions.
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    hit = await page.evaluate("([x, y]) => { const e = document.elementFromPoint(x, y); return {tag: e?.tagName, prefix: e?.closest?.('.territory')?.dataset.prefix}; }", [x, y])
    assert hit.get("prefix") == "2", f"territory center hit-test mismatch: box={box}, hit={hit}"
    await page.mouse.click(x, y)
    await page.wait_for_function(
        "document.querySelector('[data-prefix=\"2\"]')?.getAttribute('aria-selected') === 'true'",
        timeout=TIMEOUT,
    )
    await page.locator("#inspector .metrics").wait_for(timeout=TIMEOUT)
    assert "Group 2" in await page.locator("#inspector h2").inner_text()
    return territory


async def flow_selection_and_samples(page: Page, base: str) -> None:
    await boot(page, base)
    await select_internal(page)
    text = await page.locator("#inspector").inner_text()
    assert "Parent distance" in text and "Token norm" in text
    images = page.locator("#inspector .sample-grid img")
    assert await images.count() > 0
    await images.first.wait_for(state="visible")
    assert await images.first.evaluate("image => image.complete && image.naturalWidth > 0")


async def flow_outliers_and_keyboard(page: Page, base: str) -> None:
    await boot(page, base)
    await select_internal(page)
    representative = page.locator("#inspector .sample-grid img").first
    await representative.wait_for(state="visible", timeout=TIMEOUT)
    representative_src = await representative.get_attribute("src")
    outliers = page.get_by_role("button", name="Outliers", exact=True)
    await outliers.click()
    assert await outliers.get_attribute("aria-pressed") == "true"
    outlier = page.locator("#inspector .sample-grid img").first
    await page.wait_for_function(
        "previous => document.querySelector('#inspector .sample-grid img')?.getAttribute('src') !== previous",
        arg=representative_src,
        timeout=TIMEOUT,
    )
    assert await outlier.get_attribute("src") != representative_src
    assert await outlier.evaluate("image => image.complete && image.naturalWidth > 0")
    target = page.locator('#atlas .territory[data-prefix="5"]')
    await target.focus()
    await target.press("Enter")
    await page.wait_for_function("document.querySelector('[data-prefix=\"5\"]')?.getAttribute('aria-selected') === 'true'")
    assert "Group 5" in await page.locator("#inspector h2").inner_text()


async def flow_enter_and_breadcrumbs(page: Page, base: str) -> None:
    await boot(page, base)
    await select_internal(page)
    await page.get_by_role("button", name="Enter group", exact=True).click()
    crumbs = page.locator("#breadcrumbs button")
    await crumbs.nth(1).wait_for()
    assert await crumbs.count() == 2
    assert await crumbs.nth(1).inner_text() == "2"
    assert await crumbs.nth(1).get_attribute("aria-current") == "page"
    assert all(prefix.startswith("2,") for prefix in await page.locator("#atlas .territory").evaluate_all("nodes => nodes.map(n => n.dataset.prefix)"))
    assert await page.locator("#atlas .territory").count() >= 2
    level = page.locator("#levelControl")
    assert await level.input_value() == "2"
    await level.focus()
    await level.press("Home")
    await level.press("Enter")
    await page.wait_for_function("document.querySelector('#breadcrumbs [aria-current=page]')?.textContent === 'Root'")
    await select_internal(page)
    await page.get_by_role("button", name="Enter group", exact=True).click()
    await page.wait_for_function("document.querySelector('#breadcrumbs [aria-current=page]')?.textContent === '2'")
    await page.get_by_role("button", name="Root", exact=True).click()
    await page.wait_for_function("document.querySelector('#breadcrumbs [aria-current=page]')?.textContent === 'Root'")
    assert "Root" in await page.locator("#breadcrumbs [aria-current=page]").inner_text()
    await select_internal(page)
    await page.get_by_role("button", name="Enter group", exact=True).click()
    await page.wait_for_function("document.querySelector('#breadcrumbs [aria-current=page]')?.textContent === '2'")
    await page.locator("#backBtn:not([disabled])").click()
    await page.wait_for_function("document.querySelector('#breadcrumbs [aria-current=page]')?.textContent === 'Root'")
    root_prefixes = await page.locator("#atlas .territory").evaluate_all("nodes => nodes.map(n => n.dataset.prefix)")
    assert "2" in root_prefixes and all("," not in prefix for prefix in root_prefixes)


async def flow_parent_comparison(page: Page, base: str) -> None:
    await boot(page, base)
    await select_internal(page)
    await page.get_by_role("button", name="Parent comparison", exact=True).click()
    comparison = page.locator("#inspector .comparison")
    assert "Current focus" in await comparison.inner_text()
    assert "Selected group" in await comparison.inner_text()
    assert await comparison.locator("section").count() == 2
    for section in await comparison.locator("section").all():
        images = section.locator("img")
        assert await images.count() > 0
        assert await images.first.evaluate("image => image.complete && image.naturalWidth > 0")


def parse_viewbox(value: str) -> tuple[float, float, float, float]:
    return tuple(map(float, value.split()))  # type: ignore[return-value]


async def flow_zoom_pan_reset(page: Page, base: str) -> None:
    await boot(page, base)
    atlas = page.locator("#atlas")
    initial_text = await atlas.get_attribute("viewBox")
    assert initial_text
    initial = parse_viewbox(initial_text)
    box = await atlas.bounding_box()
    assert box
    center = (box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    await page.mouse.move(*center)
    await page.mouse.wheel(0, -500)
    zoomed = parse_viewbox(await atlas.get_attribute("viewBox") or "")
    assert initial[2] / 10 <= zoomed[2] < initial[2]
    for _ in range(30):
        await page.mouse.wheel(0, -500)
    clamped = parse_viewbox(await atlas.get_attribute("viewBox") or "")
    assert clamped[2] >= initial[2] / 10 - .01
    await page.get_by_role("button", name="Reset view", exact=True).click()
    assert parse_viewbox(await atlas.get_attribute("viewBox") or "") == initial

    await select_internal(page)
    heading = await page.locator("#inspector h2").inner_text()
    before_pan = await atlas.get_attribute("viewBox")
    await page.mouse.move(*center)
    await page.mouse.down()
    await page.mouse.move(center[0] + 80, center[1] + 45, steps=8)
    await page.mouse.up()
    assert await atlas.get_attribute("viewBox") != before_pan
    assert await page.locator("#inspector h2").inner_text() == heading


async def flow_responsive(page: Page, base: str) -> None:
    await page.set_viewport_size({"width": 768, "height": 1024})
    await boot(page, base)
    await page.wait_for_function("getComputedStyle(document.querySelector('.workspace')).gridTemplateColumns.split(' ').length === 1")
    mobile_columns = await page.locator(".workspace").evaluate("e => getComputedStyle(e).gridTemplateColumns")
    assert len(mobile_columns.split()) == 1
    assert await page.locator("#atlas .territory").count() > 0
    assert await page.locator("#atlas").is_visible() and await page.locator("#inspector").is_visible()
    assert await page.locator("#resetViewBtn").is_visible() and await page.locator("#backBtn").is_visible()
    assert await page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 2")
    await select_internal(page)
    map_top = await page.locator(".map-panel").evaluate("element => element.getBoundingClientRect().top")
    await page.locator("#inspector").evaluate("element => element.scrollTo(0, element.scrollHeight)")
    assert await page.locator("#inspector").evaluate("element => element.scrollTop > 0")
    assert await page.locator(".map-panel").evaluate("element => element.getBoundingClientRect().top") == map_top
    assert await page.evaluate("document.documentElement.scrollHeight <= window.innerHeight + 2")
    tablet_image = page.locator("#inspector .sample-grid img").first
    await tablet_image.wait_for(state="visible", timeout=TIMEOUT)
    assert await tablet_image.evaluate("image => image.complete && image.naturalWidth > 0")
    tablet_image_box = await tablet_image.bounding_box()
    assert tablet_image_box and tablet_image_box["width"] >= 120 and tablet_image_box["height"] >= 120
    await page.set_viewport_size({"width": 760, "height": 400})
    await page.wait_for_function("document.querySelector('#inspector').getBoundingClientRect().bottom <= innerHeight + 2")
    for selector in ("#atlas", "#inspector", "#resetViewBtn", "#backBtn"):
        assert await page.locator(selector).evaluate("element => { const r = element.getBoundingClientRect(); return r.width > 0 && r.height > 0 && r.right > 0 && r.bottom > 0 && r.left < innerWidth && r.top < innerHeight; }")
    assert await page.evaluate("document.documentElement.scrollHeight <= window.innerHeight + 2")
    assert await page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 2")
    await page.set_viewport_size({"width": 320, "height": 180})
    short_state = await page.locator("#inspector").evaluate("element => { const r = element.getBoundingClientRect(); return { intersects: r.bottom > 0 && r.top < innerHeight, scrollable: document.documentElement.scrollHeight > innerHeight, bodyOverflow: getComputedStyle(document.body).overflowY }; }")
    assert short_state["intersects"] or (short_state["scrollable"] and short_state["bodyOverflow"] != "hidden"), short_state
    await page.locator("#inspector").evaluate("element => element.scrollIntoView()")
    assert await page.locator("#inspector").evaluate("element => { const r = element.getBoundingClientRect(); return r.height > 0 && r.bottom > 0 && r.top < innerHeight; }")
    assert await page.get_by_role("button", name="Representative", exact=True).evaluate("element => { element.scrollIntoView(); const r = element.getBoundingClientRect(); return r.bottom > 0 && r.top < innerHeight; }")
    assert await page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 2")
    await page.set_viewport_size({"width": 1440, "height": 900})
    await page.wait_for_function("getComputedStyle(document.querySelector('.workspace')).gridTemplateColumns.split(' ').length === 2")
    desktop_columns = await page.locator(".workspace").evaluate("e => getComputedStyle(e).gridTemplateColumns")
    assert len(desktop_columns.split()) == 2
    assert await page.locator("#atlas .territory").count() > 0
    assert await page.locator("#atlas").is_visible() and await page.locator("#inspector").is_visible()
    assert await page.locator("#resetViewBtn").is_visible() and await page.locator("#backBtn").is_visible()
    assert await page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 2")


async def flow_thumbnail_http(page: Page, base: str) -> None:
    await boot(page, base)
    urls = await page.locator("#atlas image").evaluate_all("nodes => [...new Set(nodes.map(n => n.href.baseVal))]")
    assert urls
    for url in urls:
        response = await page.request.get(f"{base}{url}" if url.startswith("/") else url)
        assert response.status == 200
        assert (response.headers.get("content-type") or "").startswith("image/")
        assert len(await response.body()) > 50


async def flow_accessibility_audit(page: Page, base: str) -> None:
    await page.emulate_media(reduced_motion="reduce")

    async def force_warning(route):
        response = await route.fetch()
        payload = await response.json()
        count = len(payload["children"])
        distances = [[0.0 if i == j else 1.0 for j in range(count)] for i in range(count)]
        if count > 1:
            distances[0][1] = distances[1][0] = 1000.0
        payload["projection"] = {"raw_stress": 0.777, "stress": 0.777, "distances": distances}
        await route.fulfill(response=response, json=payload)

    async def fail_thumbnail(route):
        await route.fulfill(status=200, content_type="image/png", body=b"not a png")

    await page.route("**/api/atlas/root", force_warning)
    await page.route("**/thumb/*", fail_thumbnail)
    await boot(page, base)
    try:
        warning = page.locator("#projectionStatus")
        assert "Layout stress" in await warning.inner_text()
        assert await warning.evaluate("element => element.classList.contains('warning')")
        assert await warning.evaluate("element => { const r = element.getBoundingClientRect(); return r.width > 0 && r.bottom > 0 && r.top < innerHeight; }")

        motion = await page.evaluate("""() => ['.territory', '.loading span', 'button'].map(selector => {
            const element = document.querySelector(selector); const style = getComputedStyle(element);
            const seconds = value => Math.max(...value.split(',').map(part => parseFloat(part) * (part.includes('ms') ? .001 : 1)));
            return [selector, seconds(style.transitionDuration), seconds(style.animationDuration)];
        })""")
        assert all(transition <= 0.001 and animation <= 0.001 for _, transition, animation in motion), motion
        assert await page.evaluate("""() => { const ids = [...document.querySelectorAll('[id]')].map(element => element.id); return new Set(ids).size === ids.length; }""")

        await page.reload(wait_until="domcontentloaded")
        await boot(page, base)
        await page.evaluate("document.activeElement?.blur()")
        tab_sequence = []
        for _ in range(20):
            await page.keyboard.press("Tab")
            active = await page.evaluate("""() => { const e = document.activeElement; return { id: e.id, role: e.getAttribute('role'), text: e.textContent.trim(), brand: e.classList.contains('brand'), prefix: e.dataset.prefix }; }""")
            tab_sequence.append(active)
            if active["role"] == "treeitem":
                break
        milestones = [
            next(i for i, item in enumerate(tab_sequence) if item["brand"]),
            next(i for i, item in enumerate(tab_sequence) if item["text"] == "Root"),
            next(i for i, item in enumerate(tab_sequence) if item["id"] == "resetViewBtn"),
            next(i for i, item in enumerate(tab_sequence) if item["role"] == "treeitem"),
        ]
        assert milestones == sorted(milestones), tab_sequence
        assert await page.locator('#atlas [role="treeitem"][tabindex="0"]').count() == 1
        tab_selected_prefix = tab_sequence[-1]["prefix"]
        await page.keyboard.press("Enter")
        await page.wait_for_function("prefix => document.querySelector(`[data-prefix=\"${prefix}\"]`)?.getAttribute('aria-selected') === 'true'", arg=tab_selected_prefix)

        await page.evaluate("document.activeElement?.blur()")
        tabbed_controls = []
        activated_representative = activated_outliers = False
        for _ in range(60):
            await page.keyboard.press("Tab")
            label = await page.evaluate("document.activeElement.textContent.trim()")
            tabbed_controls.append(label)
            if label == "Representative" and not activated_representative:
                await page.keyboard.press("Space")
                activated_representative = True
            elif label == "Outliers" and not activated_outliers:
                await page.keyboard.press("Enter")
                activated_outliers = True
            elif label == "Enter group" and activated_representative and activated_outliers:
                await page.keyboard.press("Space")
                break
        assert activated_representative and activated_outliers and "Enter group" in tabbed_controls, tabbed_controls
        await page.wait_for_function("document.querySelector('#breadcrumbs [aria-current=page]')?.textContent !== 'Root'")
        await page.evaluate("document.activeElement?.blur()")
        entered_sequence = []
        for _ in range(20):
            await page.keyboard.press("Tab")
            active_id = await page.evaluate("document.activeElement.id")
            entered_sequence.append(active_id)
            if active_id == "backBtn":
                await page.keyboard.press("Space")
                break
        assert "backBtn" in entered_sequence, entered_sequence
        await page.wait_for_function("document.querySelector('#breadcrumbs [aria-current=page]')?.textContent === 'Root'")

        first = page.locator('#atlas [role="treeitem"]').first
        await first.focus()
        first_prefix = await first.get_attribute("data-prefix")
        await page.keyboard.press("ArrowRight")
        target_prefix = await page.evaluate("document.activeElement.dataset.prefix")
        assert target_prefix and target_prefix != first_prefix
        await page.keyboard.press("Enter")
        await page.wait_for_function("prefix => document.querySelector(`[data-prefix=\"${prefix}\"]`)?.getAttribute('aria-selected') === 'true'", arg=target_prefix)

        inspector_retry = page.locator("#inspector .thumb-slot button", has_text="Retry").first
        preview_retry = page.locator("#atlas .svg-image-retry").first
        await inspector_retry.wait_for(timeout=TIMEOUT)
        await preview_retry.wait_for(timeout=TIMEOUT)
        inspector_box = await page.locator("#inspector .thumb-slot").first.bounding_box()
        preview_box = await preview_retry.bounding_box()
        assert inspector_box and inspector_box["width"] > 80 and inspector_box["height"] > 80
        assert preview_box and preview_box["width"] > 20 and preview_box["height"] > 20

        async def tab_to_retry(scope, limit=80):
            sequence = []
            for _ in range(limit):
                await page.keyboard.press("Tab")
                active = await page.evaluate("""() => { const e = document.activeElement; return { role: e.getAttribute('role') || (e.tagName === 'BUTTON' ? 'button' : null), name: e.getAttribute('aria-label') || e.textContent.trim(), scope: e.closest('#inspector') ? 'inspector' : e.closest('#atlas') ? 'preview' : 'other' }; }""")
                sequence.append(active)
                if active["role"] == "button" and active["scope"] == scope and active["name"].startswith("Retry"):
                    return active, sequence
            raise AssertionError(f"Tab did not reach {scope} retry: {sequence}")

        await page.locator(".brand").focus()
        preview_tab, preview_sequence = await tab_to_retry("preview")
        inspector_tab, inspector_sequence = await tab_to_retry("inspector")
        assert preview_tab["role"] == "button" and preview_tab["name"].startswith("Retry sample"), preview_sequence
        assert inspector_tab == {"role": "button", "name": "Retry", "scope": "inspector"}, inspector_sequence

        await page.unroute("**/thumb/*", fail_thumbnail)
        async with page.expect_response(lambda response: "/thumb/" in response.url and response.status == 200):
            await page.keyboard.press("Space")
        inspector_image = page.locator("#inspector .sample-grid img").first
        await page.wait_for_function("""() => { const image = document.querySelector('#inspector .sample-grid img'); return image?.complete && image.naturalWidth > 0; }""")
        image_box = await inspector_image.bounding_box()
        assert image_box and image_box["width"] > 80 and image_box["height"] > 80

        await page.locator(".brand").focus()
        preview_tab, preview_sequence = await tab_to_retry("preview")
        assert preview_tab["name"].startswith("Retry sample"), preview_sequence
        preview_handle = await page.evaluate_handle("document.activeElement")
        async with page.expect_response(lambda response: "/thumb/" in response.url and response.status == 200):
            await page.keyboard.press("Enter")
        await page.wait_for_function("element => !element.isConnected", arg=preview_handle, timeout=TIMEOUT)
        assert await page.locator('#atlas image[visibility="visible"]').count() > 0

        outliers = page.get_by_role("button", name="Outliers", exact=True)
        await outliers.focus()
        await page.keyboard.press("Enter")
        assert await outliers.get_attribute("aria-pressed") == "true"
        representatives = page.get_by_role("button", name="Representative", exact=True)
        await representatives.focus()
        await page.keyboard.press("Space")
        assert await representatives.get_attribute("aria-pressed") == "true"

        enter = page.get_by_role("button", name="Enter group", exact=True)
        await enter.focus()
        await page.keyboard.press("Space")
        await page.wait_for_function("document.querySelector('#breadcrumbs [aria-current=page]')?.textContent !== 'Root'")
        root_crumb = page.get_by_role("button", name="Root", exact=True)
        await root_crumb.focus()
        await page.keyboard.press("Enter")
        await page.wait_for_function("document.querySelector('#breadcrumbs [aria-current=page]')?.textContent === 'Root'")

        territory = page.locator('#atlas .territory[data-prefix="2"]')
        await territory.focus()
        await page.keyboard.press("Enter")
        await page.get_by_role("button", name="Enter group", exact=True).focus()
        await page.keyboard.press("Enter")
        await page.wait_for_function("document.querySelector('#breadcrumbs [aria-current=page]')?.textContent === '2'")
        await page.locator("#backBtn").focus()
        await page.keyboard.press("Space")
        await page.wait_for_function("document.querySelector('#breadcrumbs [aria-current=page]')?.textContent === 'Root'")

        await page.locator('#atlas .territory[data-prefix="2"]').focus()
        await page.keyboard.press("Enter")
        await page.set_viewport_size({"width": 720, "height": 450})
        await page.wait_for_function("document.querySelector('#inspector').getBoundingClientRect().bottom <= innerHeight + 2")
        assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth + 2 && document.documentElement.scrollHeight <= innerHeight + 2")
        assert await page.locator("#inspector").evaluate("element => { const r = element.getBoundingClientRect(); return r.width > 0 && r.height > 0 && r.top < innerHeight && r.bottom > 0; }")
        overlap_report = await page.evaluate("""() => {
            const visible = element => { const r = element.getBoundingClientRect(), s = getComputedStyle(element); return s.visibility !== 'hidden' && s.display !== 'none' && r.width > 0 && r.height > 0 && r.right > 0 && r.bottom > 0 && r.left < innerWidth && r.top < innerHeight; };
            const controls = [...document.querySelectorAll('.topbar a, .topbar button, .map-toolbar button, #atlas [role="treeitem"][tabindex="0"], #inspector button')].filter(visible);
            const overlaps = [];
            controls.forEach((a, i) => controls.slice(i + 1).forEach(b => { if (a.contains(b) || b.contains(a)) return; const x = a.getBoundingClientRect(), y = b.getBoundingClientRect(); if (Math.min(x.right,y.right) > Math.max(x.left,y.left) && Math.min(x.bottom,y.bottom) > Math.max(x.top,y.top)) overlaps.push([a.id || a.textContent.trim(), b.id || b.textContent.trim()]); }));
            const contained = (childSelector, panelSelector) => { const child = document.querySelector(childSelector)?.getBoundingClientRect(), panel = document.querySelector(panelSelector)?.getBoundingClientRect(); return child && panel && child.left >= panel.left - 1 && child.right <= panel.right + 1 && child.top >= panel.top - 1 && child.bottom <= panel.bottom + 1; };
            return { overlaps, inspectorHeading: contained('#inspector h2', '#inspector'), projection: contained('#projectionStatus', '.topbar'), breadcrumbs: contained('#breadcrumbs', '.topbar') };
        }""")
        assert not overlap_report["overlaps"], overlap_report
        assert overlap_report["inspectorHeading"] and overlap_report["projection"] and overlap_report["breadcrumbs"], overlap_report
    finally:
        await page.unroute("**/api/atlas/root", force_warning)
        await page.unroute("**/thumb/*", fail_thumbnail)


FLOWS = [
    ("boot", flow_boot),
    ("selection and representative samples", flow_selection_and_samples),
    ("outliers and keyboard selection", flow_outliers_and_keyboard),
    ("enter, breadcrumbs, and back", flow_enter_and_breadcrumbs),
    ("parent comparison", flow_parent_comparison),
    ("wheel zoom, drag pan, and reset", flow_zoom_pan_reset),
    ("responsive layout", flow_responsive),
    ("thumbnail HTTP responses", flow_thumbnail_http),
    ("accessibility audit", flow_accessibility_audit),
]


async def run_browser(base: str) -> bool:
    errors: list[str] = []
    failures: list[str] = []
    passed = 0
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            for name, flow in FLOWS:
                context = await browser.new_context(viewport={"width": 1440, "height": 900})
                try:
                    page = await context.new_page()
                    page.on("console", lambda message, flow=name: errors.append(f"{flow}: console {message.type}: {message.text}") if message.type == "error" else None)
                    page.on("pageerror", lambda error, flow=name: errors.append(f"{flow}: pageerror: {error}"))
                    await flow(page, base)
                    print(f"PASS  {name}", flush=True)
                    passed += 1
                except Exception as exc:
                    detail = f"{name}: {type(exc).__name__}: {exc}"
                    failures.append(detail)
                    print(f"FAIL  {detail}", flush=True)
                finally:
                    await context.close()
        finally:
            await browser.close()
    if errors:
        print("Unexpected browser errors:")
        for error in errors:
            print(f"  {error}")
    print(f"Summary: {passed}/{len(FLOWS)} flows passed; {len(errors)} browser errors; {len(failures)} flow failures")
    return passed == len(FLOWS) and not errors and not failures


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="gyo-atlas-e2e-") as temporary:
        run_dir = Path(temporary)
        seed_run(run_dir)
        process, base = start_server(run_dir)
        try:
            return 0 if asyncio.run(asyncio.wait_for(run_browser(base), timeout=120)) else 1
        finally:
            stop_server(process)


if __name__ == "__main__":
    raise SystemExit(main())
