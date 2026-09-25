"""文档结构 dataclass 与 Swift 垫片 JSON（schema v1）的解析。"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

BBox = Tuple[float, float, float, float]
_SCHEME_VERSION = 1


@dataclass
class TableCell:
    """表格单元格；row/col 为合并区左上角坐标，span 为跨越数。"""

    row: int
    col: int
    row_span: int
    col_span: int
    text: str


@dataclass
class TableBlock:
    """表格区块。"""

    index: int
    bbox: Optional[BBox]
    n_rows: int
    n_cols: int
    cells: List[TableCell]


@dataclass
class ListItem:
    """列表项；level 为嵌套层级（schema v1 恒为 0）。"""

    level: int
    marker: str
    text: str


@dataclass
class ListBlock:
    """列表区块。"""

    index: int
    bbox: Optional[BBox]
    items: List[ListItem]


@dataclass
class ParagraphBlock:
    """段落区块，lines 为识别出的文本行。"""

    index: int
    bbox: Optional[BBox]
    lines: List[str]


@dataclass
class ImageBlock:
    """文档内嵌图片区块，由版面分析检测（非 Vision API 输出）。"""

    index: int
    bbox: Optional[BBox]
    image_name: Optional[str] = None


@dataclass
class DocumentResult:
    """整页文档解析结果，blocks 已按阅读序排列。"""

    image_width: float
    image_height: float
    title: Optional[str] = None
    blocks: List[Union[ParagraphBlock, ListBlock, TableBlock, ImageBlock]] = field(
        default_factory=list
    )

    @property
    def tables(self) -> List[TableBlock]:
        """文档中的全部表格（按文档顺序）。"""
        return [block for block in self.blocks if isinstance(block, TableBlock)]


def _parse_bbox(raw: object) -> Optional[BBox]:
    """容错解析 [x, y, w, h]，缺失或非法时返回 None。"""
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        return (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
    except (TypeError, ValueError):
        return None


def parse_document(payload: dict) -> DocumentResult:
    """把垫片 JSON 解析为 DocumentResult；未知 block 类型跳过并告警。"""
    if not isinstance(payload, dict):
        raise ValueError("document payload must be a dict")
    schema_version = payload.get("schema_version")
    if schema_version != _SCHEME_VERSION:
        raise ValueError(f"unsupported schema_version: {schema_version!r}")

    image_size = payload.get("image_size") or {}
    result = DocumentResult(
        image_width=float(image_size.get("width", 0.0)),
        image_height=float(image_size.get("height", 0.0)),
        title=payload.get("title"),
    )

    for raw_index, raw_block in enumerate(payload.get("blocks") or []):
        kind = raw_block.get("kind")
        index = raw_block.get("index", raw_index)
        bbox = _parse_bbox(raw_block.get("bbox"))
        if kind == "paragraph":
            result.blocks.append(
                ParagraphBlock(
                    index=index,
                    bbox=bbox,
                    lines=[str(line) for line in raw_block.get("lines") or []],
                )
            )
        elif kind == "list":
            items = [
                ListItem(
                    level=int(item.get("level", 0)),
                    marker=str(item.get("marker", "")),
                    text=str(item.get("text", "")),
                )
                for item in raw_block.get("items") or []
            ]
            result.blocks.append(ListBlock(index=index, bbox=bbox, items=items))
        elif kind == "image":
            # 预留：未来垫片直接输出图片区域时的解析路径
            result.blocks.append(ImageBlock(index=index, bbox=bbox))
        elif kind == "table":
            cells = [
                TableCell(
                    row=int(cell.get("row", 0)),
                    col=int(cell.get("col", 0)),
                    row_span=max(1, int(cell.get("row_span", 1))),
                    col_span=max(1, int(cell.get("col_span", 1))),
                    text=str(cell.get("text", "")),
                )
                for cell in raw_block.get("cells") or []
            ]
            result.blocks.append(
                TableBlock(
                    index=index,
                    bbox=bbox,
                    n_rows=max(0, int(raw_block.get("n_rows", 0))),
                    n_cols=max(0, int(raw_block.get("n_cols", 0))),
                    cells=cells,
                )
            )
        else:
            logger.warning("Skipping unknown document block kind: %r", kind)

    return result
