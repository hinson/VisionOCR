# VisionOCR

本项目实现一个简单的服务器和客户端接口，用来调用MacOS上的Vision框架的OCR功能。Vision框架的OCR在中文生僻字识别方面准确率比较高。

This project implements a simple server and client interface to call the OCR function of the Vision framework on MacOS. This OCR has a relatively high accuracy in recognizing rare Chinese characters.


## 1. 安装 Setup
```bash
git clone https://github.com/hinson/VisionOCR.git
cd VisionOCR
```

### 1.1. 只使用客户端或非MacOS系统 Client only or non-MacOS
```bash
pip install .
```

### 1.2. 使用MacOS服务端 Server on MacOS
```bash
pip install ".[server]"
```

## 2. 使用 Usage

### 2.1. 服务端 Server
```bash
python -m uvicorn vision_ocr.server:app --host 0.0.0.0 --port 9394
```

### 2.2. 客户端 Client
#### 2.2.1. requests
```python
import requests

url = "http://localhost:9394/ocr?lang=zh-cn"

with open("test.png", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

print(response.json())
```

#### 2.2.2. vision_ocr

```python
from vision_ocr import OCRlient, OCRResult

image_paths = ["example1.png", "example2.jpg"]
```

##### 同步 Synchronous
```python
with OCRClient(base_url="http://localhost:9394", lang="zh-cn") as client:
    # Single file
    result: OCRResult = client.recognize(image_paths[0])
    print(f"Sync result: {result.text}")

    # Batch processing
    results: list[OCRResult] = client.recognize_batch(image_paths)
    for i, result in enumerate(results):
        print(f"Result {i+1} ({result.file_name}): {result.success}")
        if result.success:
            print(f"Text: {result.text}")
        else:
            print(f"Error: {result.error}")
```

##### 异步 Asynchronous
```python
import asyncio

async def example_async_usage():
    async with OCRClient(base_url="http://localhost:9394", lang="zh-cn") as client:
        # Single image from bytes
        with open(image_paths[0], "rb") as f:
            result: OCRResult = await client.recognize_async(f.read())
            print(f"Async result: {result.text}")

        # Batch processing
        results: list[OCRResult] = await client.recognize_batch_async(image_paths)
        for i, result in enumerate(results):
            print(f"Result {i+1} ({result.file_name}): {result.success}")
            if result.success:
                print(f"Text: {result.text}")
            else:
                print(f"Error: {result.error}")

asyncio.run(example_async_usage())
```

### 2.3. 表格识别 Document / Table OCR

`POST /ocr/document` 基于 macOS 26+ 的 `RecognizeDocumentsRequest`，识别整页文档结构（标题、段落、列表、表格，支持合并单元格），并通过版面分析提取文档内嵌图片，返回 ZIP 打包产物：

```
document.json        # 结构化数据（blocks + 表格 cells + 内嵌图片区域坐标）
document.md          # 整页文档 Markdown（表格用 HTML 语法保真合并单元格，引用内嵌图片）
images/image_000.png # 文档中检测到的内嵌图片（无内嵌图片时无该目录）
```

环境要求：服务端 macOS 26+，且已安装 Xcode 或 Command Line Tools（首次请求时自动用 swiftc 编译内置的 Swift 垫片并缓存到 `~/.cache/vision_ocr/`）。

`lang` 参数使用 BCP-47 格式（如 `zh-Hans`、`en-US`），旧式代码 `zh-cn`/`zh-tw` 会自动映射。`document.json` 的 `blocks` 按阅读序排列，`kind` 取值：`paragraph`（正文行）、`list`（列表项）、`table`（表格，`cells` 含 `row_span`/`col_span` 合并信息）、`image`（内嵌图片区域），文本类 block 均带 `bbox` 像素坐标（top-left 原点）。

内嵌图片提取为启发式版面分析（文字/表格掩码外的彩色连通域），对白底文档中的照片、图表效果较好；灰度照片会漏检，检测阈值可在 `vision_ocr/layout.py` 顶部按文档类型调整。

#### 2.3.1. requests
```python
import requests

url = "http://localhost:9394/ocr/document?lang=zh-Hans"

with open("table.png", "rb") as f:
    response = requests.post(url, files={"file": f})

with open("doc.zip", "wb") as f:
    f.write(response.content)
```

#### 2.3.2. vision_ocr
```python
from vision_ocr import OCRClient, DocumentOCRResult

with OCRClient(base_url="http://localhost:9394", lang="zh-Hans") as client:
    result: DocumentOCRResult = client.recognize_document("table.png")
    if result.success:
        # 将产物解包到本地目录（document.json / document.md / images/）
        result.save_to("./output")
```
