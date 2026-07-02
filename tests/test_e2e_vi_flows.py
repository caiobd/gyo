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


def wait_until_ready(base: str, process: multiprocessing.Process) -> None:
    deadline = time.monotonic() + 15
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


async def boot(page: Page, base: str) -> None:
    await page.goto(base, wait_until="domcontentloaded")
    await page.locator("#atlas .territory").first.wait_for(timeout=TIMEOUT)
    await page.locator("#mapLoading").wait_for(state="hidden", timeout=TIMEOUT)


async def flow_boot(page: Page, base: str) -> None:
    await boot(page, base)
    assert await page.locator("#atlas .territory").count() >= 3
    assert "gyo" in (await page.locator(".brand").inner_text()).lower()
    assert "Stress" in await page.locator("#projectionStatus").inner_text()
    assert await page.locator("#mapError").is_hidden()


async def select_internal(page: Page):
    territory = page.locator('#atlas .territory[data-prefix="2"]')
    # Dispatch on the semantic treeitem itself. SVG child images may own the
    # hit-test point, while the user-facing interaction contract belongs to g.
    await territory.dispatch_event("click")
    await page.locator("#inspector .metrics").wait_for(timeout=TIMEOUT)
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
    outliers = page.get_by_role("button", name="Outliers", exact=True)
    await outliers.click()
    assert await outliers.get_attribute("aria-pressed") == "true"
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
    assert await page.locator("#atlas .territory").count() >= 2
    await page.get_by_role("button", name="Root", exact=True).click()
    await page.wait_for_function("document.querySelectorAll('#breadcrumbs button').length === 1")
    await select_internal(page)
    await page.get_by_role("button", name="Enter group", exact=True).click()
    await page.locator("#backBtn:not([disabled])").click()
    await page.wait_for_function("document.querySelectorAll('#breadcrumbs button').length === 1")


async def flow_parent_comparison(page: Page, base: str) -> None:
    await boot(page, base)
    await select_internal(page)
    await page.get_by_role("button", name="Parent comparison", exact=True).click()
    comparison = page.locator("#inspector .comparison")
    assert "Current focus" in await comparison.inner_text()
    assert "Selected group" in await comparison.inner_text()
    assert await comparison.locator("section").count() == 2
    assert await comparison.locator("section img").count() >= 2


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
    await page.set_viewport_size({"width": 760, "height": 900})
    await boot(page, base)
    mobile_columns = await page.locator(".workspace").evaluate("e => getComputedStyle(e).gridTemplateColumns")
    assert len(mobile_columns.split()) == 1
    assert await page.locator("#atlas .territory").count() > 0
    await page.set_viewport_size({"width": 1440, "height": 900})
    await page.wait_for_timeout(150)
    desktop_columns = await page.locator(".workspace").evaluate("e => getComputedStyle(e).gridTemplateColumns")
    assert len(desktop_columns.split()) == 2
    assert await page.locator("#atlas .territory").count() > 0


async def flow_thumbnail_http(page: Page, base: str) -> None:
    await boot(page, base)
    urls = await page.locator("#atlas image").evaluate_all("nodes => [...new Set(nodes.map(n => n.href.baseVal))]")
    assert urls
    for url in urls:
        response = await page.request.get(f"{base}{url}" if url.startswith("/") else url)
        assert response.status == 200
        assert (response.headers.get("content-type") or "").startswith("image/")
        assert len(await response.body()) > 50


FLOWS = [
    ("boot", flow_boot),
    ("selection and representative samples", flow_selection_and_samples),
    ("outliers and keyboard selection", flow_outliers_and_keyboard),
    ("enter, breadcrumbs, and back", flow_enter_and_breadcrumbs),
    ("parent comparison", flow_parent_comparison),
    ("wheel zoom, drag pan, and reset", flow_zoom_pan_reset),
    ("responsive layout", flow_responsive),
    ("thumbnail HTTP responses", flow_thumbnail_http),
]


async def run_browser(base: str) -> bool:
    errors: list[str] = []
    passed = 0
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1440, "height": 900})
        page.on("console", lambda message: errors.append(f"console {message.type}: {message.text}") if message.type == "error" else None)
        page.on("pageerror", lambda error: errors.append(f"pageerror: {error}"))
        for name, flow in FLOWS:
            try:
                await flow(page, base)
                print(f"PASS  {name}", flush=True)
                passed += 1
            except Exception as exc:
                print(f"FAIL  {name}: {type(exc).__name__}: {exc}", flush=True)
        await browser.close()
    if errors:
        print("Unexpected browser errors:")
        for error in errors:
            print(f"  {error}")
    print(f"Summary: {passed}/{len(FLOWS)} flows passed; {len(errors)} browser errors")
    return passed == len(FLOWS) and not errors


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="gyo-atlas-e2e-") as temporary:
        run_dir = Path(temporary)
        seed_run(run_dir)
        port = free_port()
        base = f"http://127.0.0.1:{port}"
        process = multiprocessing.Process(target=serve, args=(str(run_dir), port), daemon=True)
        process.start()
        try:
            wait_until_ready(base, process)
            return 0 if asyncio.run(run_browser(base)) else 1
        finally:
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)


if __name__ == "__main__":
    raise SystemExit(main())
