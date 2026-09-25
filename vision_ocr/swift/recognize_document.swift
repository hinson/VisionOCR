// Swift CLI 垫片：stdin 收图片字节，调 RecognizeDocumentsRequest，stdout 吐 schema v1 JSON。
// 协议：成功 exit 0 且 stdout 仅含一个 JSON；失败 stderr 输出 "error: <原因>"，
// exit 2=图片解码失败、3=识别失败、4=参数错误。识别成功但零结果不是错误。
// 用法: recognize_document [--lang <BCP-47，逗号分隔>]

import Foundation
import Vision
import CoreGraphics
import ImageIO

struct StderrStream: TextOutputStream {
    mutating func write(_ string: String) {
        FileHandle.standardError.write(Data(string.utf8))
    }
}

var standardError = StderrStream()

func fail(_ code: Int32, _ message: String) -> Never {
    print("error: \(message)", to: &standardError)
    exit(code)
}

// MARK: - 参数解析

var languages: [String] = []
var argsIterator = CommandLine.arguments.dropFirst().makeIterator()
while let arg = argsIterator.next() {
    switch arg {
    case "--lang":
        guard let value = argsIterator.next() else {
            fail(4, "--lang requires a value")
        }
        languages.append(contentsOf: value.split(separator: ",").map(String.init))
    default:
        fail(4, "unknown argument: \(arg)")
    }
}

// MARK: - 图片读取与解码

let stdinData = FileHandle.standardInput.readDataToEndOfFile()
if stdinData.isEmpty {
    fail(2, "empty stdin")
}

guard let source = CGImageSourceCreateWithData(stdinData as CFData, nil),
      let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
else {
    fail(2, "cannot decode image")
}
let imageSize = CGSize(width: image.width, height: image.height)

// MARK: - 识别

var request = RecognizeDocumentsRequest()
if !languages.isEmpty {
    request.textRecognitionOptions.recognitionLanguages = languages.map { Locale.Language(identifier: $0) }
}

let handler = ImageRequestHandler(image)

func rectPx(_ region: any BoundingRegionProviding) -> [Double] {
    let rect = region.boundingRegion.boundingBox.toImageCoordinates(imageSize, origin: .upperLeft)
    return [Double(rect.origin.x), Double(rect.origin.y), Double(rect.width), Double(rect.height)]
}

/// 单元格等嵌套内容扁平化为文本；嵌套表格行加 "> " 前缀（schema v1 不输出二级结构）。
func flatten(_ container: DocumentObservation.Container) -> String {
    var parts: [String] = []
    let transcript = container.text.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
    if !transcript.isEmpty {
        parts.append(transcript)
    }
    for table in container.tables {
        for row in table.rows {
            let cells = row.map { $0.content.text.transcript }
            parts.append("> " + cells.joined(separator: " | "))
        }
    }
    return parts.joined(separator: "\n")
}

/// 判断区块 a 是否大部分落在区块 b 内（用于剔除 paragraphs 中混入的表格/列表文本）。
func mostlyInside(_ inner: [Double], _ outer: [Double]) -> Bool {
    let ix0 = max(inner[0], outer[0]), iy0 = max(inner[1], outer[1])
    let ix1 = min(inner[0] + inner[2], outer[0] + outer[2])
    let iy1 = min(inner[1] + inner[3], outer[1] + outer[3])
    let intersection = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    let area = max(inner[2] * inner[3], 1e-9)
    return intersection / area > 0.5
}

do {
    let results = try await handler.perform(request)
    guard let observation = results.first else {
        fail(3, "no document observation returned")
    }
    let doc = observation.document

    var tableBoxes: [[Double]] = []
    var listBoxes: [[Double]] = []
    var blocks: [[String: Any]] = []

    for table in doc.tables {
        let bbox = rectPx(table)
        tableBoxes.append(bbox)
        var cells: [[String: Any]] = []
        for (r, row) in table.rows.enumerated() {
            for cell in row where cell.rowRange.lowerBound == r {
                // 行内是压缩表示：真实行列号取自 range 的下界，而非循环下标
                cells.append([
                    "row": cell.rowRange.lowerBound,
                    "col": cell.columnRange.lowerBound,
                    "row_span": cell.rowRange.count,
                    "col_span": cell.columnRange.count,
                    "text": flatten(cell.content),
                ])
            }
        }
        blocks.append([
            "kind": "table",
            "bbox": bbox,
            "n_rows": table.rows.count,
            "n_cols": table.columns.count,
            "cells": cells,
        ])
    }

    for list in doc.lists {
        let bbox = rectPx(list)
        listBoxes.append(bbox)

        // 注意：Item.content 会自引用地包含父列表（实测对平面列表 content.lists 非空），
        // 递归下钻会无限复制，故 schema v1 只输出顶层 items
        let items: [[String: Any]] = list.items.map { item in
            ["level": 0, "marker": item.markerString, "text": item.itemString]
        }
        blocks.append(["kind": "list", "bbox": bbox, "items": items])
    }

    let titleText = doc.title?.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
    for (paragraphIndex, paragraph) in doc.paragraphs.enumerated() {
        let bbox = rectPx(paragraph)
        // paragraphs 会混入表格单元格与列表行文本，按重叠面积过滤
        if tableBoxes.contains(where: { mostlyInside(bbox, $0) })
            || listBoxes.contains(where: { mostlyInside(bbox, $0) }) {
            continue
        }
        // 首段与标题文本相同时去重，避免 Markdown 里标题出现两次
        if paragraphIndex == 0,
           titleText == paragraph.transcript.trimmingCharacters(in: .whitespacesAndNewlines) {
            continue
        }
        var lines = paragraph.lines.map(\.transcript)
        if lines.isEmpty {
            lines = [paragraph.transcript]
        }
        blocks.append(["kind": "paragraph", "bbox": bbox, "lines": lines])
    }

    // 按阅读序排序：自上而下、同行自左而右；标题单独置于最前
    blocks.sort { (a: [String: Any], b: [String: Any]) -> Bool in
        let ba = a["bbox"] as! [Double], bb = b["bbox"] as! [Double]
        if abs(ba[1] - bb[1]) > 2.0 { return ba[1] < bb[1] }
        return ba[0] < bb[0]
    }

    var index = 0
    for i in blocks.indices {
        blocks[i]["index"] = index
        index += 1
    }

    var payload: [String: Any] = [
        "schema_version": 1,
        "image_size": ["width": Double(imageSize.width), "height": Double(imageSize.height)],
        "blocks": blocks,
    ]
    if let title = doc.title?.transcript {
        payload["title"] = title
    }

    let json = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted])
    FileHandle.standardOutput.write(json)
    exit(0)
} catch {
    fail(3, "recognition failed: \(error)")
}
