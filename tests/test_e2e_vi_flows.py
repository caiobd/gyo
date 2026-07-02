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
    assert "gyo" in (await page.locator(".brand").inner_text()).lower()
    assert "Stress" in await page.locator("#projectionStatus").inner_text()
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
    await page.set_viewport_size({"width": 760, "height": 900})
    await boot(page, base)
    await page.wait_for_function("getComputedStyle(document.querySelector('.workspace')).gridTemplateColumns.split(' ').length === 1")
    mobile_columns = await page.locator(".workspace").evaluate("e => getComputedStyle(e).gridTemplateColumns")
    assert len(mobile_columns.split()) == 1
    assert await page.locator("#atlas .territory").count() > 0
    assert await page.locator("#atlas").is_visible() and await page.locator("#inspector").is_visible()
    assert await page.locator("#resetViewBtn").is_visible() and await page.locator("#backBtn").is_visible()
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
