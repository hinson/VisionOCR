"""OCR 产物生成：整页 Markdown 渲染、文档内嵌图片裁剪 PNG、ZIP 内存打包。

Markdown 中表格用 HTML 语法（GFM 管道表格不支持合并单元格的 rowspan/colspan）；
images/ 目录存放从原文档中检测并裁剪出的内嵌图片（版面分析，见 layout.py），
而非表格截图。
"""

import html
import io
import json
import logging
import zipfile
from typing import Dict, List, Optional

from Foundation import NSData, NSMutableData
from Quartz import (
    CGImageCreateWithImageInRect,
    CGImageDestinationAddImage,
    CGImageDestinationCreateWithData,
    CGImageDestinationFinalize,
    CGImageGetHeight,
    CGImageGetWidth,
    CGImageSourceCreateImageAtIndex,
    CGImageSourceCreateWithData,
)

from . import layout
from .document import DocumentResult, ImageBlock, TableBlock, TableCell

logger = logging.getLogger(__name__)

# 裁剪时向四周扩的像素余量，缓解 bbox 边缘偏移
_CROP_PADDING_PX = 4
_PNG_UTI = "public.png"


def _decode_image(image_bytes: bytes):
    """把图片字节解码为 CGImage，失败返回 None。"""
    ns_data = NSData.dataWithBytes_length_(image_bytes, len(image_bytes))
    source = CGImageSourceCreateWithData(ns_data, None)
    if source is None:
        return None
    return CGImageSourceCreateImageAtIndex(source, 0, None)


def _cgimage_to_png(cg_image) -> Optional[bytes]:
    """把 CGImage 编码为 PNG 字节流。"""
    buffer = NSMutableData.data()
    destination = CGImageDestinationCreateWithData(buffer, _PNG_UTI, 1, None)
    if destination is None:
        return None
    CGImageDestinationAddImage(destination, cg_image, None)
    if not CGImageDestinationFinalize(destination):
        return None
    return bytes(buffer)


def _crop_regions(
    image_bytes: bytes, bboxes: List[tuple], name_prefix: str
) -> Dict[str, bytes]:
    """按原图坐标 bbox 列表逐个裁剪为 PNG，命名 <name_prefix>_NNN.png。"""
    cg_image = _decode_image(image_bytes)
    if cg_image is None:
        logger.warning("Cannot decode source image for cropping: %s", name_prefix)
        return {}

    width = CGImageGetWidth(cg_image)
    height = CGImageGetHeight(cg_image)

    images: Dict[str, bytes] = {}
    for x, y, w, h in bboxes:
        left = max(0, int(x) - _CROP_PADDING_PX)
        top = max(0, int(y) - _CROP_PADDING_PX)
        right = min(width, int(x + w) + _CROP_PADDING_PX)
        bottom = min(height, int(y + h) + _CROP_PADDING_PX)
        if right - left < 1 or bottom - top < 1:
            continue
        cropped = CGImageCreateWithImageInRect(
            cg_image, ((left, top), (right - left, bottom - top))
        )
        if cropped is None:
            continue
        png_bytes = _cgimage_to_png(cropped)
        if png_bytes:
            images[f"{name_prefix}_{len(images):03d}.png"] = png_bytes
    return images


def _escape_cell_text(text: str) -> str:
    """单元格文本的 HTML 转义与换行处理（换行转 <br>）。"""
    return html.escape(text, quote=False).replace("\n", "<br>").strip()


def _render_table(doc_table: TableBlock) -> List[str]:
    """把表格区块渲染为 HTML 表格——GFM 管道表格不支持合并单元格，HTML 可用
    rowspan/colspan 保真还原合并语义（cells 只含合并区左上代表格）。"""
    lines: List[str] = []

    if doc_table.n_rows <= 0 or doc_table.n_cols <= 0:
        return lines

    # 锚点矩阵：cell 左上角所在位置；被合并区覆盖的其余位置跳过不输出
    grid: List[List[Optional[TableCell]]] = [
        [None] * doc_table.n_cols for _ in range(doc_table.n_rows)
    ]
    covered: set = set()
    for cell in doc_table.cells:
        if not (0 <= cell.row < doc_table.n_rows and 0 <= cell.col < doc_table.n_cols):
            continue
        grid[cell.row][cell.col] = cell
        for r in range(cell.row, min(cell.row + cell.row_span, doc_table.n_rows)):
            for c in range(cell.col, min(cell.col + cell.col_span, doc_table.n_cols)):
                if (r, c) != (cell.row, cell.col):
                    covered.add((r, c))

    def _td(cell: TableCell) -> str:
        attrs = ""
        if cell.row_span > 1:
            attrs += f' rowspan="{cell.row_span}"'
        if cell.col_span > 1:
            attrs += f' colspan="{cell.col_span}"'
        tag = "th" if cell.row == 0 else "td"
        return f"<{tag}{attrs}>{_escape_cell_text(cell.text)}</{tag}>"

    lines.append("<table>")
    for r in range(doc_table.n_rows):
        cells_html: List[str] = []
        for c in range(doc_table.n_cols):
            cell = grid[r][c]
            if cell is not None:
                cells_html.append(_td(cell))
            elif (r, c) not in covered:
                # 未被任何 cell 覆盖的稀疏空位补空格
                cells_html.append("<td></td>")
        lines.append("<tr>" + "".join(cells_html) + "</tr>")
    lines.append("</table>")
    return lines


def render_markdown(doc: DocumentResult) -> str:
    """把文档结构渲染为整页 Markdown（标题、段落、列表、HTML 表格、内嵌图片）。"""
    lines: List[str] = []
    if doc.title:
        lines += ["# " + doc.title.strip(), ""]

    for block in doc.blocks:
        kind = type(block).__name__
        if kind == "ParagraphBlock":
            if block.lines:
                lines += ["\n".join(block.lines), ""]
        elif kind == "ListBlock":
            for item in block.items:
                indent = "  " * item.level
                lines.append(f"{indent}- {item.text.strip()}")
            if block.items:
                lines.append("")
        elif kind == "TableBlock":
            lines += _render_table(block)
            lines.append("")
        elif kind == "ImageBlock":
            if block.image_name:
                lines += [f"![image](images/{block.image_name})", ""]

    return "\n".join(lines).strip() + "\n"


def _doc_to_payload(doc: DocumentResult) -> dict:
    """把 DocumentResult 序列化为 schema v1 结构（document.json 单一数据源）。"""
    blocks = []
    for block in doc.blocks:
        kind = type(block).__name__
        if kind == "ParagraphBlock":
            blocks.append(
                {
                    "kind": "paragraph",
                    "index": block.index,
                    "bbox": block.bbox,
                    "lines": block.lines,
                }
            )
        elif kind == "ListBlock":
            blocks.append(
                {
                    "kind": "list",
                    "index": block.index,
                    "bbox": block.bbox,
                    "items": [
                        {"level": item.level, "marker": item.marker, "text": item.text}
                        for item in block.items
                    ],
                }
            )
        elif kind == "TableBlock":
            blocks.append(
                {
                    "kind": "table",
                    "index": block.index,
                    "bbox": block.bbox,
                    "n_rows": block.n_rows,
                    "n_cols": block.n_cols,
                    "cells": [
                        {
                            "row": cell.row,
                            "col": cell.col,
                            "row_span": cell.row_span,
                            "col_span": cell.col_span,
                            "text": cell.text,
                        }
                        for cell in block.cells
                    ],
                }
            )
        elif kind == "ImageBlock":
            blocks.append(
                {
                    "kind": "image",
                    "index": block.index,
                    "bbox": block.bbox,
                    "image_name": block.image_name,
                }
            )
    return {
        "schema_version": 1,
        "title": doc.title,
        "image_size": {"width": doc.image_width, "height": doc.image_height},
        "blocks": blocks,
    }


def build_zip(image_bytes: bytes, doc: DocumentResult) -> bytes:
    """生成 ZIP 产物：document.json（结构化数据）+ document.md + images/ 内嵌图片。"""
    # 1) 版面分析检测内嵌图片区域，按检测序裁剪命名
    regions = layout.detect_image_regions(image_bytes, doc)
    embedded = _crop_regions(image_bytes, regions, "image")

    # 2) 把图片块按位置插入文档流，统一编号
    image_blocks = [
        ImageBlock(index=0, bbox=bbox, image_name=f"image_{i:03d}.png")
        for i, bbox in enumerate(regions)
    ]
    doc.blocks.extend(image_blocks)
    doc.blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]) if b.bbox else (0.0, 0.0))
    for i, block in enumerate(doc.blocks):
        block.index = i

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "document.json",
            json.dumps(_doc_to_payload(doc), ensure_ascii=False, indent=2),
        )
        archive.writestr("document.md", render_markdown(doc))
        for name, png_bytes in embedded.items():
            archive.writestr(f"images/{name}", png_bytes)
    return buffer.getvalue()
