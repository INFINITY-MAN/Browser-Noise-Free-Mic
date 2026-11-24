"""Automated Headless Chromium E2E Test Suite (Playwright) for NEON Browser Noise-Free Mic.

Verifies against http://localhost:8000 (auto-starting uvicorn if not already running):
1. Zero uncaught page errors and zero console errors.
2. ONNX model status badge transitions to 'Ready'.
3. Clicking 'Load Fan + 50Hz Hum Demo' (#btn-demo-indian) processes audio.
4. Telemetry cards (#card-latency, #card-rtf, #card-snr, #card-memory) populate (non-'--').
5. Both spectrogram canvases (#canvas-noisy, #canvas-denoised) have non-zero pixel variance.
6. Mode toggles (A / B / C) work and submitting a 'clean' feedback rating returns 200 OK
   from POST /api/feedback.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import pytest
from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "http://localhost:8000"


def _is_port_open(host: str = "127.0.0.1", port: int = 8000) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


@pytest.fixture(scope="module")
def neon_server():
    """Connect to an already-running local server on :8000 or launch uvicorn."""
    if _is_port_open("127.0.0.1", 8000):
        yield BASE_URL
        return

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.time() + 25.0
    while time.time() < deadline:
        if _is_port_open("127.0.0.1", 8000):
            break
        if proc.poll() is not None:
            out, err = proc.communicate(timeout=5)
            raise RuntimeError(
                f"Uvicorn exited prematurely (code {proc.returncode}):\n{err.decode(errors='replace')}"
            )
        time.sleep(0.25)
    else:
        proc.terminate()
        raise TimeoutError("Timed out waiting for NEON FastAPI server on http://localhost:8000")

    try:
        yield BASE_URL
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _canvas_pixel_variance(page, selector: str) -> float:
    """Compute RGB luminance variance across a canvas element."""
    return float(
        page.evaluate(
            """(sel) => {
                const canvas = document.querySelector(sel);
                if (!canvas || canvas.width === 0 || canvas.height === 0) return 0.0;
                const ctx = canvas.getContext('2d');
                if (!ctx) return 0.0;
                const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
                let sum = 0, sumSq = 0, count = 0;
                for (let i = 0; i < data.length; i += 4) {
                    const lum = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
                    sum += lum;
                    sumSq += lum * lum;
                    count++;
                }
                if (count === 0) return 0.0;
                const mean = sum / count;
                return (sumSq / count) - (mean * mean);
            }""",
            selector,
        )
    )


def _click_first_visible(page, selectors: List[str], label: str) -> None:
    for sel in selectors:
        loc = page.locator(sel).first
        if loc.count() > 0 and loc.is_visible():
            loc.click()
            return
    raise AssertionError(f"Could not locate visible UI control for {label} (tried: {selectors})")


def test_neon_ui_e2e(neon_server: str):
    page_errors: List[str] = []
    console_errors: List[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                "--autoplay-policy=no-user-gesture-required",
            ],
        )
        context = browser.new_context()
        page = context.new_page()

        page.on("pageerror", lambda err: page_errors.append(str(err)))
        page.on(
            "console",
            lambda msg: console_errors.append(msg.text)
            if msg.type == "error"
            else None,
        )

        page.goto(neon_server, wait_until="domcontentloaded", timeout=30000)

        page.wait_for_function(
            r"""() => {
                const badge = document.querySelector('#model-status, #status-badge, [data-testid="model-status"]');
                if (badge && /ready/i.test(badge.textContent || '')) return true;
                return /model.*ready|onnx.*ready|\bready\b/i.test(document.body.innerText || '');
            }""",
            timeout=45000,
        )

        demo_btn = page.locator("#btn-demo-indian")
        demo_btn.wait_for(state="visible", timeout=15000)
        demo_btn.click()

        card_ids = ["#card-latency", "#card-rtf", "#card-snr", "#card-memory"]
        page.wait_for_function(
            r"""(ids) => ids.every((id) => {
                const el = document.querySelector(id);
                if (!el) return false;
                const txt = (el.textContent || '').trim();
                return txt.length > 0 && !txt.includes('--');
            })""",
            arg=card_ids,
            timeout=45000,
        )
        for cid in card_ids:
            val = page.locator(cid).inner_text().strip()
            print(f"[TELEMETRY] {cid}: {val}")
            assert val and "--" not in val, f"Telemetry card {cid} did not populate (got '{val}')"

        noisy_var = _canvas_pixel_variance(page, "#canvas-noisy")
        denoised_var = _canvas_pixel_variance(page, "#canvas-denoised")
        print(f"[CANVAS] noisy_var={noisy_var:.2f}, denoised_var={denoised_var:.2f}")
        assert noisy_var > 0.0, f"Expected non-zero variance on #canvas-noisy, got {noisy_var}"
        assert denoised_var > 0.0, f"Expected non-zero variance on #canvas-denoised, got {denoised_var}"

        _click_first_visible(
            page,
            ["#mode-a", "#btn-mode-a", '[data-mode="A"]', 'button:has-text("Mode A")'],
            "Mode A",
        )
        _click_first_visible(
            page,
            ["#mode-b", "#btn-mode-b", '[data-mode="B"]', 'button:has-text("Mode B")'],
            "Mode B",
        )
        _click_first_visible(
            page,
            ["#mode-c", "#btn-mode-c", '[data-mode="C"]', 'button:has-text("Mode C")'],
            "Mode C",
        )

        _click_first_visible(
            page,
            [
                '#btn-feedback-clean',
                '[data-feedback="clean"]',
                '[data-rating="clean"]',
                'button:has-text("Clean")',
            ],
            "Clean feedback button",
        )
        with page.expect_response(
            lambda resp: "/api/feedback" in resp.url and resp.request.method == "POST",
            timeout=15000,
        ) as resp_info:
            submit_btn = page.locator("#btn-submit-feedback").first
            if submit_btn.count() > 0 and submit_btn.is_visible():
                submit_btn.click()
        feedback_resp = resp_info.value
        print(f"[FEEDBACK] POST /api/feedback -> HTTP {feedback_resp.status}")
        assert feedback_resp.status == 200, (
            f"Expected 200 OK from POST /api/feedback, got {feedback_resp.status}"
        )

        browser.close()

    assert len(page_errors) == 0, f"Uncaught page errors detected: {page_errors}"
    assert len(console_errors) == 0, f"Console errors detected: {console_errors}"
