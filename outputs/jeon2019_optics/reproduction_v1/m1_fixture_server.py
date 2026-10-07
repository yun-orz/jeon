# -*- coding: utf-8 -*-
"""M1 续传器测试用的本地HTTP样例服务；可控制 Range 行为与异常中断。

只在回环地址与随机端口监听，不访问外网。行为由查询参数与请求头控制：

- 默认支持 Range，返回 206 与 Content-Range。
- ``?ignore_range=1``：忽略 Range，始终返回 200 与整包（用于测试200替代206）。
- ``?truncate=1``：声明完整 Content-Range，但只写一半正文后关闭连接（长度不符）。
- 请求头 ``X-Fixture-Truncate-Blocks: N``：对前 N 次分块请求写一半后断开（异常中断）。

每次请求都记入内存日志，供测试断言服务端实际收到的 Range。
"""
from __future__ import annotations

import hashlib
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

RANGE = re.compile(r"^bytes=(\d+)-(\d+)$")


class Fixture:
    def __init__(self, payload=b""):
        self.payload = payload
        self.requests = []
        self.lock = threading.Lock()
        self.ignore_range = False
        self.truncate = False
        self.truncate_blocks = 0
        self.range_requests = 0

    def note(self, record):
        with self.lock:
            self.requests.append(record)

    def sha256(self):
        return hashlib.sha256(self.payload).hexdigest()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    fixture: Fixture = None

    def log_message(self, *args):  # 静默：测试输出只保留结论
        return

    def _payload_range(self, query):
        start, end = 0, len(self.fixture.payload) - 1
        header = self.headers.get("Range")
        if header and not (self.fixture.ignore_range or query.get("ignore_range")):
            match = RANGE.match(header.strip())
            if not match:
                return None
            start, end = int(match.group(1)), int(match.group(2))
            if start > end or end >= len(self.fixture.payload):
                return None
        return start, end

    def do_HEAD(self):
        payload = self.fixture.payload
        self.fixture.note({"method": "HEAD", "range": None})
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Last-Modified", "Tue, 29 Oct 2024 23:16:21 GMT")
        self.end_headers()

    def do_GET(self):
        query = parse_qs(urlparse(self.path).query)
        span = self._payload_range(query)
        payload = self.fixture.payload
        if span is None:
            self.fixture.note({"method": "GET", "range": self.headers.get("Range"), "status": 416})
            self.send_response(416)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        start, end = span
        body = payload[start:end + 1]
        partial = bool(self.headers.get("Range")) and not (self.fixture.ignore_range or query.get("ignore_range"))
        header_limit = self.headers.get("X-Fixture-Truncate-Blocks")
        header_limit = int(header_limit) if header_limit and header_limit.isdigit() else 0
        with self.fixture.lock:
            self.fixture.range_requests += 1
            attempt = self.fixture.range_requests
            truncate_now = (self.fixture.truncate or query.get("truncate")
                            or attempt <= max(self.fixture.truncate_blocks, header_limit))
        self.fixture.note({"method": "GET", "range": self.headers.get("Range"),
                           "status": 206 if partial else 200, "truncated": bool(truncate_now)})
        self.send_response(206 if partial else 200)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "bytes")
        if partial:
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, len(payload)))
        self.end_headers()
        if truncate_now:
            self.wfile.write(body[:max(1, len(body) // 2)])
            self.wfile.flush()
            self.close_connection = True
            return
        self.wfile.write(body)


def start_fixture(payload):
    """启动样例服务；返回 (fixture, server, base_url)。调用方负责关闭。"""
    fixture = Fixture(payload)

    class Bound(Handler):
        pass

    Bound.fixture = fixture
    server = ThreadingHTTPServer(("127.0.0.1", 0), Bound)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return fixture, server, "http://127.0.0.1:%d/torch-fixture.whl" % server.server_address[1]
