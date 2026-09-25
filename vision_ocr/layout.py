"""启发式版面分析：从文档图中检测内嵌图片（照片/插图）区域。

Vision API 只输出文本结构，没有提取文档内嵌图片的能力。原理：文字/表格/列表
bbox 掩码之外的"非白内容"做连通域分析，面积与彩色密度达标的域判为图片。
在降采样图上分析控制耗时，bbox 映射回原图坐标后由调用方裁剪。
"""

import logging
from typing import List, Tuple

from Foundation import NSData
from Quartz import (
    CGDataProviderCopyData,
    CGImageGetBitsPerPixel,
    CGImageGetBytesPerRow,
    CGImageGetDataProvider,
    CGImageGetHeight,
    CGImageGetWidth,
    CGImageSourceCreateWithData,
    CGImageSourceCreateThumbnailAtIndex,
    kCGImageSourceCreateThumbnailFromImageAlways,
    kCGImageSourceThumbnailMaxPixelSize,
)

from .document import BBox, DocumentResult

logger = logging.getLogger(__name__)

# 分析在降采样图上进行：最长边像素上限（纯 Python 连通域的耗时控制）
_THUMBNAIL_MAX_PIXEL_SIZE = 400
# 像素为"内容"的亮度阈值（任一通道低于此值即非白）
_CONTENT_CHANNEL_MAX = 235
# 掩码外扩（缩略图像素），缓解 bbox 边缘偏移导致的文字残留
_MASK_DILATION_PX = 2
# 连通域像素面积下限（占缩略图面积比例）
_REGION_MIN_AREA_RATIO = 0.005
# 域最短边下限（缩略图像素），过滤细长噪声
_REGION_MIN_SIDE_PX = 16
# 彩色像素判定：通道极差下限；域内彩色占比达到该值才认为是照片/图表
_COLORFUL_CHANNEL_SPREAD = 24
_REGION_MIN_COLORFUL_RATIO = 0.15


def _decode_thumbnail(image_bytes: bytes) -> Tuple[bytes, int, int, int, int]:
    """解码降采样图，返回 (像素 bytes, 宽, 高, 每行字节数, 每像素字节数)。

    解码失败时各字段返回 0/空 bytes。
    """
    ns_data = NSData.dataWithBytes_length_(image_bytes, len(image_bytes))
    source = CGImageSourceCreateWithData(ns_data, None)
    if source is None:
        return b"", 0, 0, 0, 0
    image = CGImageSourceCreateThumbnailAtIndex(
        source,
        0,
        {
            kCGImageSourceCreateThumbnailFromImageAlways: True,
            kCGImageSourceThumbnailMaxPixelSize: _THUMBNAIL_MAX_PIXEL_SIZE,
        },
    )
    if image is None:
        return b"", 0, 0, 0, 0

    buf = bytes(CGDataProviderCopyData(CGImageGetDataProvider(image)))
    bits_pp = CGImageGetBitsPerPixel(image)
    return (
        buf,
        CGImageGetWidth(image),
        CGImageGetHeight(image),
        CGImageGetBytesPerRow(image),
        bits_pp // 8,
    )


def _is_colorful(pixel: bytes) -> bool:
    """判断像素是否为彩色（通道极差大），通道顺序无关。"""
    return max(pixel[:3]) - min(pixel[:3]) >= _COLORFUL_CHANNEL_SPREAD


def detect_image_regions(image_bytes: bytes, doc: DocumentResult) -> List[BBox]:
    """检测文档中的内嵌图片区域，返回原图像素坐标（top-left）的 bbox 列表。

    图片判据：位于已知文本/表格/列表区块掩码之外、连通域面积与最短边达标、
    且域内彩色像素占比达标。灰度照片会被漏检（启发式的已知局限）。
    """
    buf, width, height, stride, bpp = _decode_thumbnail(image_bytes)
    if not buf or width == 0 or height == 0 or bpp < 1:
        logger.warning("Cannot decode thumbnail for image-region detection")
        return []
    if doc.image_width <= 0 or doc.image_height <= 0:
        logger.warning("Unknown image size; skip image-region detection")
        return []

    scale_x = width / doc.image_width
    scale_y = height / doc.image_height
    channels = min(3, bpp)

    def pixel_offset(x: int, y: int) -> int:
        return y * stride + x * bpp

    # 内容掩码：非白像素
    is_content = bytearray(width * height)
    for y in range(height):
        row = y * stride
        for x in range(width):
            off = row + x * bpp
            if any(buf[off + i] < _CONTENT_CHANNEL_MAX for i in range(channels)):
                is_content[y * width + x] = 1

    # 排除已知文本/表格/列表区块（bbox 外扩后置灰）
    scale_x = width / doc.image_width
    scale_y = height / doc.image_height
    for block in doc.blocks:
        if block.bbox is None:
            continue
        x0, y0, w, h = block.bbox
        ex0 = max(0, int(x0 * scale_x) - _MASK_DILATION_PX)
        ey0 = max(0, int(y0 * scale_y) - _MASK_DILATION_PX)
        ex1 = min(width, int((x0 + w) * scale_x) + _MASK_DILATION_PX)
        ey1 = min(height, int((y0 + h) * scale_y) + _MASK_DILATION_PX)
        for y in range(ey0, ey1):
            for x in range(ex0, ex1):
                is_content[y * width + x] = 0

    # 8 邻接连通域（BFS），统计面积/包围盒/彩色像素
    visited = bytearray(width * height)
    regions: List[BBox] = []
    min_area = max(1, int(width * height * _REGION_MIN_AREA_RATIO))
    for start in range(width * height):
        if not is_content[start] or visited[start]:
            continue
        sy, sx = divmod(start, width)
        stack = [(sx, sy)]
        visited[start] = 1
        min_x = max_x = sx
        min_y = max_y = sy
        area = 0
        colorful = 0
        while stack:
            x, y = stack.pop()
            area += 1
            off = pixel_offset(x, y)
            if _is_colorful(buf[off : off + channels]):
                colorful += 1
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        idx = ny * width + nx
                        if is_content[idx] and not visited[idx]:
                            visited[idx] = 1
                            stack.append((nx, ny))

        if area < min_area:
            continue
        if min(max_x - min_x, max_y - min_y) < _REGION_MIN_SIDE_PX:
            continue
        if area and colorful / area < _REGION_MIN_COLORFUL_RATIO:
            continue

        # 映射回原图坐标
        regions.append(
            (
                min_x / scale_x,
                min_y / scale_y,
                (max_x - min_x + 1) / scale_x,
                (max_y - min_y + 1) / scale_y,
            )
        )

    return regions
