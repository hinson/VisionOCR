"""Swift 垫片桥接：按需编译 + 子进程调用，输出 schema v1 文档 JSON。

垫片源码随包分发（vision_ocr/swift/recognize_document.swift），首次调用时以源码
哈希为缓存键编译到用户缓存目录；pyobjc 无法驱动 Swift-only 的
RecognizeDocumentsRequest，子进程是唯一可行通道。
"""

import fcntl
import hashlib
import json
import logging
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 编译大文件较慢，与识别超时分开控制
_COMPILE_TIMEOUT_SECONDS = 300.0
_RECOGNIZE_TIMEOUT_SECONDS = 120.0

# 同时运行的垫片进程上限；Vision 文档识别后端无法承受高并发
_RECOGNIZE_CONCURRENCY = 2
_RECOGNIZE_SEMAPHORE = threading.BoundedSemaphore(_RECOGNIZE_CONCURRENCY)

_SOURCE_PATH = Path(__file__).parent / "swift" / "recognize_document.swift"

# 客户端惯用的 ISO 639-1 旧码到 Vision 期望的 BCP-47 的映射
_LANG_ALIASES = {"zh-cn": "zh-Hans", "zh-tw": "zh-Hant"}


class SwiftShimError(Exception):
    """Swift 垫片缺失、编译失败或运行异常。"""


class ImageDecodeError(SwiftShimError):
    """垫片无法解码输入图片（非图片数据或已损坏）。"""


class RecognitionError(SwiftShimError):
    """Vision 识别请求失败（图片合法但识别环节出错）。"""


def _cache_binary_path() -> Path:
    """返回按源码内容寻址的缓存二进制路径，源码升级自动换键。"""
    digest = hashlib.sha256(_SOURCE_PATH.read_bytes()).hexdigest()[:16]
    base = os.environ.get("VISION_OCR_CACHE_DIR")
    root = Path(base) if base else Path.home() / ".cache" / "vision_ocr"
    return root / digest / "recognize_document"


def get_shim_binary() -> Path:
    """返回垫片可执行文件路径，缺失时惰性编译（flock 防并发重复编译）。"""
    binary = _cache_binary_path()
    if binary.is_file() and os.access(binary, os.X_OK):
        return binary

    swiftc = shutil.which("swiftc")
    if swiftc is None:
        raise SwiftShimError(
            "swiftc not found; document OCR requires Xcode Command Line Tools on macOS 26+"
        )

    binary.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(str(binary) + ".lock")
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            # 拿到锁后二次检查，避免并发请求重复编译
            if binary.is_file() and os.access(binary, os.X_OK):
                return binary
            logger.info("Compiling Swift document-OCR shim to %s", binary)
            tmp_binary = binary.with_name(f"{binary.name}.tmp.{os.getpid()}")
            try:
                proc = subprocess.run(
                    [swiftc, "-O", "-o", str(tmp_binary), str(_SOURCE_PATH)],
                    capture_output=True,
                    timeout=_COMPILE_TIMEOUT_SECONDS,
                )
                if proc.returncode != 0:
                    detail = proc.stderr.decode(errors="replace")[-500:]
                    raise SwiftShimError(f"swiftc failed: {detail}")
                # 原子替换，杜绝并发请求读到半成品
                os.replace(tmp_binary, binary)
            finally:
                tmp_binary.unlink(missing_ok=True)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return binary


def _normalize_lang(lang: str) -> str:
    """把旧式语言码映射为 BCP-47，其余原样透传。"""
    return _LANG_ALIASES.get(lang.strip().lower(), lang)


def recognize_document(
    image_data: bytes,
    lang: Optional[str] = None,
    timeout: float = _RECOGNIZE_TIMEOUT_SECONDS,
) -> dict:
    """调用垫片识别文档，返回 schema v1 的 dict；失败按 exit code 映射异常。"""
    binary = get_shim_binary()
    command = [str(binary)]
    if lang:
        command += ["--lang", _normalize_lang(lang)]

    # Vision 后端对并发识别敏感：并发垫片进程过多会楔死系统识别服务（实测发生过），
    # 故服务端作为唯一 spawn 源在进程内限流，其余请求排队
    with _RECOGNIZE_SEMAPHORE:
        try:
            proc = subprocess.run(
                command,
                input=image_data,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise SwiftShimError(f"shim timed out after {timeout}s") from exc

    stderr_tail = proc.stderr.decode(errors="replace").strip()[-500:]
    if proc.returncode == 0:
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise SwiftShimError(f"shim printed invalid JSON: {exc}") from exc
    if proc.returncode == 2:
        raise ImageDecodeError(stderr_tail or "cannot decode image")
    if proc.returncode == 3:
        raise RecognitionError(stderr_tail or "recognition failed")

    # 常见场景：macOS < 26 时 dyld 加载 Vision 符号失败
    raise SwiftShimError(
        f"shim exited with code {proc.returncode}: {stderr_tail or 'no stderr output'}"
    )
