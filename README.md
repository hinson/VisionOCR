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