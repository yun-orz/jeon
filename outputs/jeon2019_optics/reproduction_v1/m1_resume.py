# -*- coding: utf-8 -*-
"""M1 分块续传器：HTTP Range 续传、逐块SHA、整包SHA核验、可重复执行与中断恢复。

设计约束（对应 tasks/M1_gpu_environment.md 第2步）：

1. 清单记录 URL、总长、官方整包SHA、已有前缀偏移、每块字节范围与块SHA。
2. 只接受 206 且 Content-Range/长度完全匹配的分块；服务端返回 200（忽略 Range）
   时，只有在"单次请求即可覆盖剩余全部字节"时才允许接受，否则明确失败。
3. 未通过长度与SHA核验的块绝不写入已完成清单；.tmp 残片保留作证据，不被当成块。
4. 中断、进程被杀、清单损坏或目录已存在都不能造成不可恢复：每次运行先重新扫描
   磁盘、逐块复核SHA，只补齐真正缺失或损坏的部分。
5. 已有前缀（旧部分下载）只读复制，不改动、不删除；整包SHA是它有效性的唯一裁决。
6. 不删除任何文件；块文件、下载清单与失败记录全部保留。
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

SCHEMA = "jeon2019-m1-resume-v1"
DEFAULT_BLOCK_BYTES = 32 * 1024 * 1024
READ_BYTES = 1024 * 1024
USER_AGENT = "jeon2019-m1-resume/1"


class ResumeError(RuntimeError):
    """续传过程中的可恢复或不可恢复错误；均保留已有文件。"""


def sha256_file(path, limit=None):
    """分块计算SHA；limit用于只核对前缀字节，避免整包重复读取。"""
    digest = hashlib.sha256()
    remaining = limit
    with Path(path).open("rb") as stream:
        while remaining is None or remaining > 0:
            size = READ_BYTES if remaining is None else min(remaining, READ_BYTES)
            block = stream.read(size)
            if not block:
                break
            digest.update(block)
            if remaining is not None:
                remaining -= len(block)
    return digest.hexdigest()


def write_json(path, payload):
    """原子写入JSON：临时文件加替换，避免中断留下半个清单。"""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plan_blocks(offset, total, block_bytes=DEFAULT_BLOCK_BYTES):
    """按固定块长切分 [offset, total)；返回 [(索引, 起始, 结束)]，结束为闭区间。"""
    if not 0 <= offset <= total:
        raise ResumeError("偏移非法：%r" % (offset,))
    if block_bytes <= 0:
        raise ResumeError("块长必须为正")
    blocks, index = [], 0
    start = offset
    while start < total:
        end = min(start + block_bytes, total) - 1
        blocks.append((index, start, end))
        index += 1
        start = end + 1
    return blocks


def content_range(offset, end, total):
    return "bytes %d-%d/%d" % (offset, end, total)


class Resumer:
    """一个 URL 到本地整包的可恢复下载器。"""

    def __init__(self, url, total_bytes, sha256, destination, block_bytes=DEFAULT_BLOCK_BYTES,
                 timeout=120, attempts=3, prefix=None, logger=print, backoff=2.0, expected_headers=None):
        self.url = url
        self.total_bytes = int(total_bytes)
        self.sha256 = sha256.lower()
        self.destination = Path(destination)
        self.block_bytes = int(block_bytes)
        self.timeout = float(timeout)
        self.attempts = int(attempts)
        self.logger = logger
        self.backoff = float(backoff)
        self.expected_headers = dict(expected_headers or {})
        self.directory = self.destination.parent
        self.prefix = Path(prefix) if prefix else None
        self.offset = 0
        self.records = []
        self.events = []
        self.manifest_path = self.directory / "download_manifest.json"
        self.state_path = self.directory / "resume_state.json"
        self.prefix_path = self.directory / "prefix.bin"

    # ---- 状态与清单 -------------------------------------------------
    def load_state(self):
        """尽力读取已有状态；损坏清单只记为事件，不阻止重新核对磁盘。"""
        if not self.state_path.exists():
            return None
        try:
            state = read_json(self.state_path)
        except (OSError, ValueError) as exc:
            self.event("state_unreadable", error=repr(exc))
            return None
        if state.get("schema") != SCHEMA or state.get("url") != self.url:
            self.event("state_mismatch", schema=state.get("schema"))
            return None
        if int(state.get("total_bytes", -1)) != self.total_bytes or state.get("sha256") != self.sha256:
            self.event("state_mismatch", note="总长或官方SHA不一致")
            return None
        self.offset = int(state.get("offset", 0))
        self.records = list(state.get("blocks", []))
        return state

    def save_state(self, status, extra=None):
        payload = {"schema": SCHEMA, "url": self.url, "total_bytes": self.total_bytes,
                   "sha256": self.sha256, "offset": self.offset, "block_bytes": self.block_bytes,
                   "blocks": self.records, "status": status, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "events": self.events, "destination": str(self.destination),
                   "prefix": str(self.prefix) if self.prefix else None}
        if extra:
            payload.update(extra)
        write_json(self.state_path, payload)
        return payload

    def event(self, kind, **fields):
        record = {"kind": kind, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        record.update(fields)
        self.events.append(record)
        self.logger("[事件] %s %s" % (kind, fields if fields else ""))
        return record

    def record_of(self, index):
        for item in self.records:
            if int(item["index"]) == index:
                return item
        return None

    def block_path(self, index):
        return self.directory / ("part_%03d.bin" % index)

    # ---- 前缀 -------------------------------------------------------
    def ensure_prefix(self, keep=True):
        """把已有部分下载只读复制成 prefix.bin；不修改、不删除原文件。"""
        if self.prefix is None:
            self.offset = 0
            return None
        source = Path(self.prefix)
        if not source.is_file():
            raise ResumeError("找不到已有前缀文件：" + str(source))
        size = source.stat().st_size
        if not 0 < size <= self.total_bytes:
            raise ResumeError("已有前缀长度非法：%d" % size)
        if self.offset not in (0, size) and self.prefix_path.exists():
            self.event("prefix_offset_conflict", recorded=self.offset, actual=size)
        if self.prefix_path.exists() and self.prefix_path.stat().st_size == size:
            digest = sha256_file(self.prefix_path)
            self.event("prefix_reused", path=str(self.prefix_path), bytes=size, sha256=digest)
        else:
            started = time.perf_counter()
            with source.open("rb") as inp, self.prefix_path.open("wb") as out:
                while True:
                    chunk = inp.read(READ_BYTES)
                    if not chunk:
                        break
                    out.write(chunk)
            if source.stat().st_size != size or self.prefix_path.stat().st_size != size:
                raise ResumeError("复制前缀期间长度改变，拒绝继续")
            digest = sha256_file(self.prefix_path)
            self.event("prefix_copied", source=str(source), path=str(self.prefix_path), bytes=size,
                       sha256=digest, seconds=round(time.perf_counter() - started, 3))
        self.offset = size
        self.prefix_sha256 = digest
        if not keep:
            raise ResumeError("前缀必须保留：整包校验前不得删除任何来源文件")
        return digest

    # ---- 网络 -------------------------------------------------------
    def probe(self):
        """核对官方头信息与冻结总长；不下载正文。"""
        request = urllib.request.Request(self.url, method="HEAD", headers={"User-Agent": USER_AGENT})
        with self._open(request) as response:
            headers = {key.lower(): value for key, value in response.headers.items()}
        observed = {"status": int(response.status), "content_length": headers.get("content-length"),
                    "accept_ranges": headers.get("accept-ranges"),
                    "last_modified": headers.get("last-modified"), "etag": headers.get("etag")}
        problems = []
        if observed["status"] != 200:
            problems.append("HEAD 返回 %s，不是200" % observed["status"])
        if observed["content_length"] is None or int(observed["content_length"]) != self.total_bytes:
            problems.append("服务端长度 %s 与冻结总长 %d 不一致" % (observed["content_length"], self.total_bytes))
        if observed["accept_ranges"] != "bytes":
            problems.append("服务端未声明 Accept-Ranges: bytes")
        for key, value in self.expected_headers.items():
            if headers.get(key.lower()) != value:
                problems.append("头 %s 期望 %s，实际 %s" % (key, value, headers.get(key.lower())))
        self.event("probe", **observed)
        return {"observed": observed, "problems": problems, "passed": not problems}

    def _open(self, request):
        return urllib.request.urlopen(request, timeout=self.timeout)

    def _download_block(self, index, start, end):
        """下载单块到 .tmp，校验 Content-Range 与长度后原子改名为正式块。

        返回的记录范围可与请求范围不同：仅当服务端忽略 Range 返回200、且该响应覆盖
        本次起始偏移之后的全部剩余字节时，本块被接受为覆盖 [start, total-1] 的单块。
        请求范围在每次重试时都重新构造，失败尝试不会改变后续请求。
        """
        target = self.block_path(index)
        temporary = self.directory / ("part_%03d.bin.tmp" % index)
        last_error = None
        accepted = None
        for attempt in range(1, self.attempts + 1):
            if temporary.exists():
                self.event("stale_tmp_kept", path=str(temporary), bytes=temporary.stat().st_size,
                           note="上次中断残片保留；本次成功后才被覆盖")
            expected = content_range(start, end, self.total_bytes)
            wanted = end - start + 1
            headers = {"Range": "bytes=%d-%d" % (start, end), "User-Agent": USER_AGENT}
            request = urllib.request.Request(self.url, headers=headers)
            try:
                with self._open(request) as response:
                    status = int(response.status)
                    range_header = response.headers.get("Content-Range")
                    declared = int(response.headers.get("Content-Length") or -1)
                    if status == 206:
                        if range_header != expected:
                            raise ResumeError("Content-Range 不符：期望 %s，实际 %s" % (expected, range_header))
                        if declared != wanted:
                            raise ResumeError("Content-Length %d 与请求 %d 字节不符" % (declared, wanted))
                        written = self._copy(response, temporary, want=wanted)
                        accepted_range = (start, end, written)
                    elif status == 200:
                        # 服务端忽略 Range 返回整包：只有当响应能覆盖 [start, total) 时才可接受。
                        # 声明长度必须不小于剩余字节；开头 start 字节（已持有的前缀）跳过后再核验实际写入量。
                        span = self.total_bytes - start
                        if declared != -1 and declared < span:
                            raise ResumeError("200 响应声明 %d 字节，不足以覆盖剩余 %d 字节，拒绝接受"
                                              % (declared, span))
                        self.event("server_ignored_range", block=index, status=status, declared=declared,
                                   span=span, note="200 覆盖剩余全部字节，接受为单块（跳过已持有前缀）")
                        written = self._copy_response(response, temporary, skip=start, want=span)
                        accepted_range = (start, self.total_bytes - 1, written)
                    else:
                        raise ResumeError("服务端返回 %s（Content-Length=%d，Content-Range=%s），"
                                          "既非合规206也不覆盖剩余全部字节" % (status, declared, range_header))
                accepted = accepted_range
                break
            except (urllib.error.URLError, urllib.error.HTTPError, OSError, ResumeError, ValueError,
                    http.client.HTTPException) as exc:
                # 服务端提前断开时 http.client 抛 IncompleteRead，属于长度不符，按可重试失败记录。
                last_error = repr(exc)
                accepted = None
                self.event("block_attempt_failed", block=index, attempt=attempt, start=start, end=end,
                           error=last_error)
                if attempt >= self.attempts:
                    self.save_state("block_failed", {"failed_block": index, "last_error": last_error})
                    raise ResumeError("块%d在%d次尝试后仍失败：%s（保留全部文件，可重跑续传）"
                                      % (index, self.attempts, last_error))
                time.sleep(self.backoff * attempt)
        if accepted is None:
            raise ResumeError("块%d未取得合规响应" % index)
        start, end, wanted = accepted
        digest = sha256_file(temporary)
        if temporary.stat().st_size != wanted:
            raise ResumeError("块%d临时文件长度在读取期间改变" % index)
        os.replace(temporary, target)
        return {"index": index, "start": start, "end": end, "bytes": wanted,
                "range": content_range(start, end, self.total_bytes), "sha256": digest,
                "url": self.url, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    def _copy(self, response, target, want=None):
        written = 0
        with Path(target).open("wb") as out:
            while True:
                chunk = response.read(READ_BYTES)
                if not chunk:
                    break
                out.write(chunk)
                written += len(chunk)
        if want is not None and written != want:
            raise ResumeError("响应正文写入 %d 字节，期望 %d" % (written, want))
        return written

    def _copy_response(self, response, target, skip=0, want=None):
        """流式写入响应；可跳过开头 skip 字节（已持有的前缀），并核对实际写入字节数。"""
        remaining = int(skip)
        written = 0
        with Path(target).open("wb") as out:
            while True:
                chunk = response.read(READ_BYTES)
                if not chunk:
                    break
                if remaining:
                    if len(chunk) <= remaining:
                        remaining -= len(chunk)
                        continue
                    chunk = chunk[remaining:]
                    remaining = 0
                out.write(chunk)
                written += len(chunk)
        if want is not None and written != want:
            raise ResumeError("响应正文写入 %d 字节，期望 %d" % (written, want))
        return written

    def collect_blocks(self, blocks):
        """逐块复核磁盘文件长度与SHA。

        磁盘文件本身（长度与SHA）是唯一裁决依据：已登记的记录不豁免重新核验，
        磁盘上未登记的块经核验后也可复用。任何未通过核验的块都不进入待下载之外
        的集合，也绝不写入"已完成"清单；已完成只由整包SHA通过后的
        download_manifest.json 认定。
        """
        pending, verified = [], set()
        for index, start, end in blocks:
            record = self.record_of(index)
            path = self.block_path(index)
            wanted = end - start + 1
            if path.exists() and path.stat().st_size == wanted:
                digest = sha256_file(path)
                if record is not None and record.get("source") != "recovered_from_disk" \
                        and record.get("sha256") != digest:
                    self.event("block_corrupt", block=index, bytes=wanted, note="磁盘内容与记录SHA不符，重新下载")
                    pending.append((index, start, end))
                    continue
                if record is None:
                    record = {"index": index, "start": start, "end": end, "bytes": wanted,
                              "range": content_range(start, end, self.total_bytes), "sha256": digest,
                              "url": self.url, "source": "recovered_from_disk",
                              "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
                    self.records.append(record)
                    self.event("orphan_block_adopted", block=index, bytes=wanted, sha256=digest,
                               note="磁盘上未登记的块经长度与SHA核验后复用")
                verified.add(index)
                continue
            if record is not None:
                self.event("block_corrupt", block=index,
                           bytes=path.stat().st_size if path.exists() else None, wanted=wanted,
                           note="文件缺失或长度不符，重新下载")
            pending.append((index, start, end))
        return pending, verified

    def verify_blocks(self, blocks):
        """兼容入口：返回 (待下载列表, 已核验记录列表)。"""
        pending, verified = self.collect_blocks(blocks)
        return pending, [self.record_of(index) for index in sorted(verified)]

    # ---- 主流程 -----------------------------------------------------
    def run(self, keep_prefix=True, skip_if_assembled=True):
        started = time.perf_counter()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.load_state()
        if skip_if_assembled and self.destination.is_file():
            size = self.destination.stat().st_size
            if size == self.total_bytes:
                digest = sha256_file(self.destination)
                if digest == self.sha256:
                    self.event("already_complete", path=str(self.destination), sha256=digest)
                    self.save_state("complete", {"whole_sha256": digest,
                                                 "seconds": round(time.perf_counter() - started, 3)})
                    return self.result(True, digest, "整包已存在且SHA通过，未重新下载")
                self.event("existing_whole_sha_mismatch", path=str(self.destination), sha256=digest)
            else:
                self.event("existing_whole_size_mismatch", path=str(self.destination), bytes=size)
        prefix_sha = self.ensure_prefix(keep_prefix)
        # 核验阶段不把任何块登记进状态文件：只有整包SHA通过才认定完成。
        self.save_state("verifying", {"records": [], "verified_blocks": 0})
        plan = plan_blocks(self.offset, self.total_bytes, self.block_bytes)
        self.event("plan", offset=self.offset, total=self.total_bytes, blocks=len(plan),
                   block_bytes=self.block_bytes)
        # 先整体核验一次磁盘状态，只补齐真正缺失或损坏的块。
        pending, verified = self.collect_blocks(plan)
        redo = {index for index, _, _ in pending}
        self.event("verified", offset=self.offset, ok=len(verified), pending=len(pending))
        self.save_state("downloading", {"records": [], "verified_blocks": len(verified)})
        done = set(verified) - redo
        for index, start, end in plan:
            if index in done:
                continue
            block = self.record_of(index)
            if block is not None and int(block["start"]) == start and int(block["end"]) == end \
                    and index not in redo:
                done.add(index)
                continue                  # 该块已由磁盘复核或更早记录覆盖
            self.logger("[下载] 块%d %d-%d" % (index, start, end))
            record = self._download_block(index, start, end)
            if record.get("sha256") != sha256_file(self.block_path(index)):
                raise ResumeError("块%d写入后SHA复核不符，拒绝登记" % index)
            self.drop_overlapping(record)
            self.records.append(record)
            self.records.sort(key=lambda item: (int(item["start"]), int(item["index"])))
            # 服务端可能把本块范围扩展到覆盖剩余全部字节（200替代206）：
            # 该记录会覆盖后续计划块，因此把它们的索引一并标记为已满足。
            for other, other_start, _ in plan:
                if other != index and start <= other_start <= int(record["end"]):
                    done.add(other)
            done.add(index)
            self.save_state("downloading", {"records": [], "verified_blocks": len(done)})
        whole = self.assemble()
        return self.result(True, whole["sha256"], whole["note"])

    def drop_overlapping(self, record):
        """删除与新区块范围重叠的旧记录；新块覆盖的范围以新记录为准。"""
        start, end = int(record["start"]), int(record["end"])
        keep = []
        for item in self.records:
            other_start, other_end = int(item["start"]), int(item["end"])
            if other_end < start or other_start > end:
                keep.append(item)
            else:
                self.event("record_replaced", dropped_index=item["index"], dropped_range=item["range"],
                           new_range=record["range"])
        self.records = keep

    def recorded_cover(self):
        """按记录的范围核对块是否无缝覆盖 [offset, total)；范围可能与固定分块不同。"""
        ordered = sorted(self.records, key=lambda item: int(item["start"]))
        cursor = self.offset
        for item in ordered:
            start, end = int(item["start"]), int(item["end"])
            if start != cursor:
                raise ResumeError("块覆盖不连续：期望起点 %d，实际 %d" % (cursor, start))
            if end < start or end >= self.total_bytes:
                raise ResumeError("块范围越界：%d-%d" % (start, end))
            cursor = end + 1
        if cursor != self.total_bytes:
            raise ResumeError("块覆盖不足：到 %d，应为 %d；拒绝拼接不完整数据" % (cursor, self.total_bytes))
        return ordered

    def assemble(self):
        """按记录范围拼接前缀与全部分块，核对整包官方SHA；失败保留中间产物。"""
        ordered = self.recorded_cover()
        assembled = self.directory / (self.destination.name + ".assembling")
        written = 0
        with assembled.open("wb") as out:
            if self.offset:
                if not self.prefix_path.is_file():
                    raise ResumeError("缺少前缀副本，无法拼接：" + str(self.prefix_path))
                with self.prefix_path.open("rb") as source:
                    while True:
                        chunk = source.read(READ_BYTES)
                        if not chunk:
                            break
                        out.write(chunk)
                        written += len(chunk)
            for item in ordered:
                path = self.block_path(int(item["index"]))
                if not path.is_file():
                    raise ResumeError("缺少已记录块文件，拒绝拼接：" + str(path))
                with path.open("rb") as source:
                    while True:
                        chunk = source.read(READ_BYTES)
                        if not chunk:
                            break
                        out.write(chunk)
                        written += len(chunk)
        if written != self.total_bytes:
            self.save_state("assemble_length_failed", {"assembled_bytes": written})
            raise ResumeError("拼接长度 %d 与冻结总长 %d 不符；保留全部分块与拼接文件" % (written, self.total_bytes))
        digest = sha256_file(assembled)
        if digest != self.sha256:
            self.save_state("whole_sha_failed", {"assembled_sha256": digest, "assembled_bytes": written})
            raise ResumeError("整包SHA不符：实际 %s，官方 %s；保留块与拼接文件，不安装" % (digest, self.sha256))
        os.replace(assembled, self.destination)
        self.event("whole_sha_passed", path=str(self.destination), bytes=written, sha256=digest)
        self.save_state("complete", {"whole_sha256": digest})
        return {"sha256": digest, "bytes": written, "note": "整包官方SHA通过"}

    def result(self, passed, digest, note):
        manifest = {"schema": SCHEMA, "url": self.url, "total_bytes": self.total_bytes,
                    "official_sha256": self.sha256, "offset": self.offset, "block_bytes": self.block_bytes,
                    "prefix_source": str(self.prefix) if self.prefix else None,
                    "prefix_bytes": self.offset, "prefix_sha256": getattr(self, "prefix_sha256", None),
                    "blocks": self.records, "verified_blocks": len(self.records),
                    "destination": str(self.destination), "assembled_sha256": digest,
                    "whole_sha256_passed": bool(digest == self.sha256 and passed),
                    "events": self.events, "status": "complete" if passed else "failed", "note": note}
        write_json(self.manifest_path, manifest)
        return manifest
