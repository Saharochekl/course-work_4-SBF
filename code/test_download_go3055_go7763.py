#!/usr/bin/env python3
"""Offline transport tests for the GO-3055/GO-7763 downloader."""

from __future__ import annotations

import io
import json
import re
import socket
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from unittest.mock import patch

import numpy as np
from astropy.io import fits

import download_go3055_go7763 as downloader


ETAG = '"fixture-etag"'
LAST_MODIFIED = "Wed, 01 Jan 2025 00:00:00 GMT"
RANGE_RE = re.compile(r"^bytes=(\d+)-(\d*)$")


def jwst_like_fits_bytes(
    *, program: str, filter_name: str, file_name: str, seed: int = 0
) -> bytes:
    """Create a tiny FITS with the identity fields checked by the downloader."""

    primary = fits.PrimaryHDU()
    primary.header["PROGRAM"] = program
    primary.header["FILTER"] = filter_name
    primary.header["FILENAME"] = file_name
    image = np.arange(seed, seed + 64, dtype=np.float32).reshape(8, 8)
    science = fits.ImageHDU(image, name="SCI")
    stream = io.BytesIO()
    fits.HDUList([primary, science]).writeto(stream, checksum=True)
    return stream.getvalue()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self.server.fixture.serve(self)  # type: ignore[attr-defined]

    def log_message(self, _format: str, *_args: object) -> None:
        return


class LocalDownloadServer:
    """Scriptable localhost server with Range and request-order accounting."""

    def __init__(
        self,
        payloads: dict[str, bytes],
        actions: list[dict[str, object]] | None = None,
    ) -> None:
        self.payloads = dict(payloads)
        self.actions = [dict(action) for action in (actions or [])]
        self.requests: list[dict[str, object]] = []
        self.max_active = 0
        self._active = 0
        self._lock = threading.Lock()
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._httpd.daemon_threads = True
        self._httpd.fixture = self  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def endpoint(self) -> str:
        host, port = self._httpd.server_address
        return f"http://{host}:{port}/download"

    def __enter__(self) -> "LocalDownloadServer":
        self._thread.start()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2)

    def _next_action(self) -> dict[str, object]:
        with self._lock:
            return self.actions.pop(0) if self.actions else {}

    @staticmethod
    def _send_headers(
        handler: BaseHTTPRequestHandler,
        status: int,
        content_length: int,
        *,
        content_range: str | None = None,
        retry_after: str | None = None,
    ) -> None:
        handler.send_response(status)
        handler.send_header("Content-Length", str(content_length))
        handler.send_header("Accept-Ranges", "bytes")
        handler.send_header("Content-Encoding", "identity")
        handler.send_header("ETag", ETAG)
        handler.send_header("Last-Modified", LAST_MODIFIED)
        if content_range is not None:
            handler.send_header("Content-Range", content_range)
        if retry_after is not None:
            handler.send_header("Retry-After", retry_after)
        handler.end_headers()

    def serve(self, handler: BaseHTTPRequestHandler) -> None:
        action = self._next_action()
        parsed = urlparse(handler.path)
        uri = parse_qs(parsed.query).get("uri", [""])[0]
        range_header = handler.headers.get("Range")
        record = {
            "uri": uri,
            "range": range_header,
            "if_range": handler.headers.get("If-Range"),
            "action": dict(action),
        }
        with self._lock:
            self.requests.append(record)
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            delay = float(action.get("delay", 0.0))
            if delay:
                time.sleep(delay)

            explicit_status = action.get("status")
            if explicit_status is not None:
                status = int(explicit_status)
                self._send_headers(
                    handler,
                    status,
                    0,
                    retry_after=str(action.get("retry_after"))
                    if action.get("retry_after") is not None
                    else None,
                )
                return

            payload = self.payloads.get(uri)
            if payload is None:
                self._send_headers(handler, 404, 0)
                return
            if action.get("mode") == "corrupt":
                payload = b"X" * len(payload)

            requested_start = 0
            requested_end = len(payload) - 1
            range_match = RANGE_RE.match(range_header or "")
            if range_match and action.get("mode") != "ignore-range":
                requested_start = int(range_match.group(1))
                if range_match.group(2):
                    requested_end = min(int(range_match.group(2)), len(payload) - 1)
                if requested_start >= len(payload):
                    self._send_headers(
                        handler,
                        416,
                        0,
                        content_range=f"bytes */{len(payload)}",
                    )
                    return
                body = payload[requested_start : requested_end + 1]
                status = 206
                content_range = (
                    f"bytes {requested_start}-{requested_start + len(body) - 1}/"
                    f"{len(payload)}"
                )
            else:
                body = payload
                status = 200
                content_range = None

            if action.get("mode") == "short":
                limit = min(int(action.get("bytes", 2048)), len(body))
                self._send_headers(
                    handler,
                    status,
                    len(body),
                    content_range=content_range,
                )
                handler.wfile.write(body[:limit])
                handler.wfile.flush()
                handler.close_connection = True
                try:
                    handler.connection.shutdown(socket.SHUT_WR)
                except OSError:
                    pass
                return

            self._send_headers(
                handler,
                status,
                len(body),
                content_range=content_range,
            )
            handler.wfile.write(body)
            handler.wfile.flush()
        finally:
            with self._lock:
                self._active -= 1


class SequentialDownloaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.data_dir = self.root / "data"
        self.data_dir.mkdir()

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def make_product(
        self,
        *,
        target: str = "NGC TEST",
        obsid: str = "jw03055-o001",
        role: str = "signal",
        filter_name: str = "F150W",
        file_name: str = "jw03055-o001_test_f150w_i2d.fits",
        uri: str = "mast:JWST/product/test-f150w",
        seed: int = 0,
        manifest_delta: int = 0,
    ) -> tuple[downloader.Product, bytes]:
        payload = jwst_like_fits_bytes(
            program="3055",
            filter_name=filter_name,
            file_name=file_name,
            seed=seed,
        )
        product = downloader.Product(
            program="3055",
            target=target,
            obsid=obsid,
            role=role,
            filter_name=filter_name,
            product_uri=uri,
            file_name=file_name,
            destination=self.data_dir / target / file_name,
            manifest_size=len(payload) + manifest_delta,
        )
        product.destination.parent.mkdir(parents=True, exist_ok=True)
        return product, payload

    def download(
        self,
        product: downloader.Product,
        *,
        attempts: int = 1,
        sleeps: list[float] | None = None,
    ) -> downloader.DownloadResult:
        sleep_log = sleeps if sleeps is not None else []
        return downloader.download_product(
            product=product,
            data_dir=self.data_dir,
            reserve_gib=0.0,
            max_attempts=attempts,
            timeout=2.0,
            chunk_size=512,
            sleep=sleep_log.append,
            random_source=lambda: 0.5,
        )

    def assert_valid_product(
        self, product: downloader.Product, payload: bytes
    ) -> None:
        self.assertEqual(product.destination.read_bytes(), payload)
        check = downloader.validate_fits(product.destination, product=product)
        self.assertTrue(check.ready, check.reason)
        self.assertFalse(downloader.partial_path(product.destination).exists())
        self.assertFalse(downloader.partial_metadata_path(product.destination).exists())

    def test_progress_description_formats_fractional_speed(self):
        description = downloader.progress_description(
            current_bytes=123_000_000,
            file_total=1_216_137_600,
            bytes_received=10_000_000,
            elapsed=100.0,
            queue_progress=None,
            planned_file_size=1_216_137_600,
        )
        self.assertIn("0.10 MB/s", description)
        self.assertIn("10.1%", description)

    def test_fresh_200_download(self):
        product, payload = self.make_product()
        with LocalDownloadServer({product.product_uri: payload}) as server, patch.object(
            downloader, "MAST_DOWNLOAD_ENDPOINT", server.endpoint
        ):
            result = self.download(product)

        self.assertEqual(result.status, "downloaded")
        self.assertEqual(result.attempts, 1)
        self.assertEqual(result.downloaded_bytes, len(payload))
        self.assertEqual([item["range"] for item in server.requests], [None])
        self.assert_valid_product(product, payload)

    def test_existing_valid_file_skips_http(self):
        product, payload = self.make_product()
        product.destination.write_bytes(payload)
        with LocalDownloadServer({product.product_uri: payload}) as server, patch.object(
            downloader, "MAST_DOWNLOAD_ENDPOINT", server.endpoint
        ):
            result = self.download(product)

        self.assertEqual(result.status, "already-ready")
        self.assertEqual(result.attempts, 0)
        self.assertEqual(server.requests, [])
        self.assert_valid_product(product, payload)

    def test_206_resume_uses_part_metadata(self):
        product, payload = self.make_product()
        split = 2304
        downloader.partial_path(product.destination).write_bytes(payload[:split])
        downloader.save_partial_metadata(
            product,
            downloader.RemoteInfo(len(payload), ETAG, LAST_MODIFIED),
        )
        with LocalDownloadServer({product.product_uri: payload}) as server, patch.object(
            downloader, "MAST_DOWNLOAD_ENDPOINT", server.endpoint
        ):
            result = self.download(product)

        self.assertEqual(result.status, "resumed")
        self.assertEqual(server.requests[0]["range"], f"bytes={split}-")
        self.assertEqual(server.requests[0]["if_range"], ETAG)
        self.assert_valid_product(product, payload)

    def test_legacy_truncated_destination_is_replaced_only_after_full_download(self):
        product, payload = self.make_product()
        split = 1800
        product.destination.write_bytes(payload[:split])
        with LocalDownloadServer({product.product_uri: payload}) as server, patch.object(
            downloader, "MAST_DOWNLOAD_ENDPOINT", server.endpoint
        ):
            result = self.download(product)

        self.assertEqual(result.status, "downloaded")
        self.assertEqual([item["range"] for item in server.requests], [None])
        self.assert_valid_product(product, payload)

    def test_server_ignoring_range_restarts_without_appending(self):
        product, payload = self.make_product()
        split = 1700
        downloader.partial_path(product.destination).write_bytes(payload[:split])
        downloader.save_partial_metadata(
            product,
            downloader.RemoteInfo(len(payload), ETAG, LAST_MODIFIED),
        )
        with LocalDownloadServer(
            {product.product_uri: payload}, actions=[{"mode": "ignore-range"}]
        ) as server, patch.object(
            downloader, "MAST_DOWNLOAD_ENDPOINT", server.endpoint
        ):
            result = self.download(product)

        self.assertEqual(result.status, "downloaded")
        self.assertEqual(server.requests[0]["range"], f"bytes={split}-")
        self.assertEqual(product.destination.stat().st_size, len(payload))
        self.assert_valid_product(product, payload)

    def test_short_response_keeps_part_and_next_invocation_resumes(self):
        product, payload = self.make_product()
        sleeps: list[float] = []
        with LocalDownloadServer(
            {product.product_uri: payload}, actions=[{"mode": "short", "bytes": 2048}]
        ) as server, patch.object(
            downloader, "MAST_DOWNLOAD_ENDPOINT", server.endpoint
        ):
            first = self.download(product, attempts=1, sleeps=sleeps)
            part = downloader.partial_path(product.destination)
            self.assertEqual(first.status, "failed")
            self.assertFalse(product.destination.exists())
            self.assertTrue(part.exists())
            saved_size = part.stat().st_size
            self.assertGreater(saved_size, 0)
            self.assertTrue(downloader.partial_metadata_path(product.destination).exists())

            second = self.download(product, attempts=1, sleeps=sleeps)

        self.assertEqual(second.status, "resumed")
        self.assertEqual(server.requests[1]["range"], f"bytes={saved_size}-")
        self.assert_valid_product(product, payload)

    def test_http_503_is_retried(self):
        product, payload = self.make_product()
        sleeps: list[float] = []
        with LocalDownloadServer(
            {product.product_uri: payload},
            actions=[{"status": 503, "retry_after": "0"}, {}],
        ) as server, patch.object(
            downloader, "MAST_DOWNLOAD_ENDPOINT", server.endpoint
        ):
            result = self.download(product, attempts=2, sleeps=sleeps)

        self.assertEqual(result.status, "downloaded")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(sleeps, [0.0])
        self.assertEqual(len(server.requests), 2)
        self.assert_valid_product(product, payload)

    def test_http_404_is_permanent_failure(self):
        product, payload = self.make_product()
        with LocalDownloadServer(
            {product.product_uri: payload}, actions=[{"status": 404}]
        ) as server, patch.object(
            downloader, "MAST_DOWNLOAD_ENDPOINT", server.endpoint
        ):
            result = self.download(product, attempts=3)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.attempts, 1)
        self.assertIn("HTTP 404", result.message)
        self.assertEqual(len(server.requests), 1)
        self.assertFalse(product.destination.exists())

    def test_corrupt_exact_size_response_is_rejected_then_replaced(self):
        product, payload = self.make_product()
        sleeps: list[float] = []
        with LocalDownloadServer(
            {product.product_uri: payload},
            actions=[{"mode": "corrupt"}, {}],
        ) as server, patch.object(
            downloader, "MAST_DOWNLOAD_ENDPOINT", server.endpoint
        ):
            result = self.download(product, attempts=2, sleeps=sleeps)

        self.assertEqual(result.status, "downloaded")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(len(server.requests), 2)
        self.assertFalse(downloader.partial_path(product.destination).exists())
        self.assert_valid_product(product, payload)

    def test_existing_corrupt_exact_size_is_not_treated_as_ready(self):
        product, payload = self.make_product()
        product.destination.write_bytes(b"Z" * len(payload))
        with LocalDownloadServer({product.product_uri: payload}) as server, patch.object(
            downloader, "MAST_DOWNLOAD_ENDPOINT", server.endpoint
        ):
            result = self.download(product)

        self.assertEqual(result.status, "downloaded")
        self.assertEqual(len(server.requests), 1)
        self.assertIsNone(server.requests[0]["range"])
        self.assert_valid_product(product, payload)

    def test_http_416_promotes_complete_valid_part(self):
        product, payload = self.make_product(manifest_delta=downloader.FITS_BLOCK_SIZE)
        downloader.partial_path(product.destination).write_bytes(payload)
        downloader.save_partial_metadata(
            product,
            downloader.RemoteInfo(len(payload), ETAG, LAST_MODIFIED),
        )
        with LocalDownloadServer({product.product_uri: payload}) as server, patch.object(
            downloader, "MAST_DOWNLOAD_ENDPOINT", server.endpoint
        ):
            result = self.download(product)

        self.assertEqual(result.status, "resumed")
        self.assertEqual(result.downloaded_bytes, 0)
        self.assertEqual(server.requests[0]["range"], f"bytes={len(payload)}-")
        self.assertEqual(product.destination.read_bytes(), payload)
        check = downloader.validate_fits(
            product.destination,
            product=product,
            authoritative_size=len(payload),
            use_manifest_tolerance=False,
        )
        self.assertTrue(check.ready, check.reason)
        self.assertFalse(downloader.partial_path(product.destination).exists())
        self.assertFalse(
            downloader.partial_metadata_path(product.destination).exists()
        )

    def test_main_downloads_products_in_strict_sequence(self):
        first, first_payload = self.make_product(
            target="ORDERED",
            obsid="jw03055-o010",
            role="signal",
            filter_name="F150W",
            file_name="first_f150w_i2d.fits",
            uri="mast:JWST/product/first",
            seed=10,
        )
        second, second_payload = self.make_product(
            target="ORDERED",
            obsid="jw03055-o010",
            role="color",
            filter_name="F090W",
            file_name="second_f090w_i2d.fits",
            uri="mast:JWST/product/second",
            seed=20,
        )
        status_path = self.root / "download_status.json"
        with (
            LocalDownloadServer(
                {
                    first.product_uri: first_payload,
                    second.product_uri: second_payload,
                },
                actions=[{"delay": 0.03}, {"delay": 0.03}],
            ) as server,
            patch.object(downloader, "MAST_DOWNLOAD_ENDPOINT", server.endpoint),
            patch.object(downloader, "read_products", return_value=([first, second], [])),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            returncode = downloader.main(
                [
                    "--program",
                    "3055",
                    "--download",
                    "--workers",
                    "1",
                    "--data-dir",
                    str(self.data_dir),
                    "--reserve-gib",
                    "0",
                    "--attempts",
                    "1",
                    "--timeout",
                    "2",
                    "--chunk-mib",
                    "1",
                    "--status-file",
                    str(status_path),
                ]
            )

        self.assertEqual(returncode, 0)
        self.assertEqual(
            [request["uri"] for request in server.requests],
            [first.product_uri, second.product_uri],
        )
        self.assertEqual(server.max_active, 1)
        status = json.loads(status_path.read_text(encoding="utf-8"))
        self.assertTrue(status["sequential_downloads"])
        self.assertFalse(status["parallel_downloads"])
        self.assertEqual(status["download_workers"], 1)
        self.assertEqual(len(status["results"]), 2)
        self.assert_valid_product(first, first_payload)
        self.assert_valid_product(second, second_payload)

    def test_main_downloads_products_in_parallel(self):
        first, first_payload = self.make_product(
            target="PARALLEL",
            obsid="jw03055-o011",
            role="signal",
            filter_name="F150W",
            file_name="parallel_f150w_i2d.fits",
            uri="mast:JWST/product/parallel-first",
            seed=30,
        )
        second, second_payload = self.make_product(
            target="PARALLEL",
            obsid="jw03055-o011",
            role="color",
            filter_name="F090W",
            file_name="parallel_f090w_i2d.fits",
            uri="mast:JWST/product/parallel-second",
            seed=40,
        )
        status_path = self.root / "parallel_status.json"
        with (
            LocalDownloadServer(
                {
                    first.product_uri: first_payload,
                    second.product_uri: second_payload,
                },
                actions=[{"delay": 0.10}, {"delay": 0.10}],
            ) as server,
            patch.object(downloader, "MAST_DOWNLOAD_ENDPOINT", server.endpoint),
            patch.object(downloader, "read_products", return_value=([first, second], [])),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            returncode = downloader.main(
                [
                    "--program",
                    "3055",
                    "--download",
                    "--workers",
                    "2",
                    "--data-dir",
                    str(self.data_dir),
                    "--reserve-gib",
                    "0",
                    "--attempts",
                    "1",
                    "--timeout",
                    "2",
                    "--chunk-mib",
                    "1",
                    "--status-file",
                    str(status_path),
                ]
            )

        self.assertEqual(returncode, 0)
        self.assertGreaterEqual(server.max_active, 2)
        status = json.loads(status_path.read_text(encoding="utf-8"))
        self.assertFalse(status["sequential_downloads"])
        self.assertTrue(status["parallel_downloads"])
        self.assertEqual(status["download_workers"], 2)
        self.assertEqual(len(status["results"]), 2)
        self.assert_valid_product(first, first_payload)
        self.assert_valid_product(second, second_payload)

    def test_workers_must_be_between_one_and_sixteen(self):
        for value in ("0", "17"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                downloader.main(["--workers", value])


if __name__ == "__main__":
    unittest.main()
