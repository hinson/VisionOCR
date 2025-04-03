import asyncio
import io
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, List, Optional, Union

import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Dataclass to hold OCR results and metadata"""

    text: str
    success: bool
    error: Optional[str] = None
    processing_time: Optional[float] = None
    file_name: Optional[str] = None


class OCRClient:
    """Client for making parallel OCR requests to the macOS Vision API service"""

    def __init__(
        self,
        base_url: str = "http://localhost:9394",
        lang: Optional[str] = "zh-cn",
        max_connections: int = 10,
        retries: int = 3,
        timeout: int = 30,
    ):
        """
        Initialize the OCR client.

        Args:
            base_url: Base URL of the OCR service
            lang: ISO 639-1 standard language code
            max_connections: Maximum number of concurrent connections
            retries: Number of retries for failed requests
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.lang = lang
        self.max_connections = max_connections
        self.retries = retries
        self.timeout = timeout
        self._loop = None
        self._session = None

    async def _async_init(self):
        """Initialize async components"""
        self._session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=self.max_connections),
            timeout=aiohttp.ClientTimeout(total=self.timeout),
        )

    def _ensure_async_init(self):
        """Ensure async components are initialized"""
        if self._session is None:
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._async_init())

    async def __aenter__(self):
        """Async context manager entry"""
        await self._async_init()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def close(self):
        """Close the client session"""
        if self._session:
            await self._session.close()
            self._session = None

    def __enter__(self):
        """Synchronous context manager entry"""
        self._ensure_async_init()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Synchronous context manager exit"""
        if self._loop:
            self._loop.run_until_complete(self.close())
            self._loop.close()
            self._loop = None

    async def _make_request(
        self, image_data: Union[bytes, BinaryIO], file_name: Optional[str] = None, attempt: int = 1
    ) -> OCRResult:
        """
        Internal method to make a single OCR request with retries.

        Args:
            image_data: Image data as bytes or file-like object
            file_name: Optional name of the file for identification
            attempt: Current attempt number (for retries)

        Returns:
            OCRResult object with the recognition results
        """
        start_time = time.time()
        result = OCRResult(text="", success=False, file_name=file_name)

        try:
            if isinstance(image_data, io.IOBase):
                image_data.seek(0)
                image_bytes = image_data.read()
            else:
                image_bytes = image_data

            data = aiohttp.FormData()
            data.add_field(
                "file",
                image_bytes,
                filename=file_name or "image.jpg",
                content_type="application/octet-stream",
            )

            async with self._session.post(
                f"{self.base_url}/ocr", params=dict(lang=self.lang), data=data
            ) as response:
                if response.status == 200:
                    json_response = await response.json()
                    result.text = json_response.get("text", "")
                    result.success = json_response.get("success", False)
                    result.error = json_response.get("error")
                else:
                    error_text = await response.text()
                    result.error = f"HTTP {response.status}: {error_text}"

        except Exception as e:
            result.error = str(e)
            if attempt <= self.retries:
                logger.warning(f"Attempt {attempt} failed for {file_name or 'image'}. Retrying...")
                await asyncio.sleep(1 * attempt)  # Exponential backoff would be better
                return await self._make_request(image_data, file_name, attempt + 1)

        result.processing_time = time.time() - start_time
        return result

    async def recognize_async(
        self, image_data: Union[bytes, BinaryIO], file_name: Optional[str] = None
    ) -> OCRResult:
        """
        Recognize text from a single image asynchronously.

        Args:
            image_data: Image data as bytes or file-like object
            file_name: Optional name of the file for identification

        Returns:
            OCRResult object with the recognition results
        """
        if not self._session:
            raise RuntimeError("Client session not initialized. Use async context manager.")
        return await self._make_request(image_data, file_name)

    async def recognize_batch_async(
        self,
        images: List[Union[bytes, BinaryIO, str, Path]],
        file_names: Optional[List[str]] = None,
    ) -> List[OCRResult]:
        """
        Recognize text from multiple images in parallel.

        Args:
            images: List of image data (bytes, file-like, or file paths)
            file_names: Optional list of file names corresponding to images

        Returns:
            List of OCRResult objects in the same order as input
        """
        if not self._session:
            raise RuntimeError("Client session not initialized. Use async context manager.")

        if file_names and len(file_names) != len(images):
            raise ValueError("file_names must match length of images if provided")

        tasks = []
        for i, image in enumerate(images):
            file_name = file_names[i] if file_names else None

            if isinstance(image, (str, Path)):
                # Handle file paths
                image_path = Path(image)
                tasks.append(self._process_file(image_path, file_name))
            else:
                # Handle bytes or file-like objects
                tasks.append(self.recognize_async(image, file_name))

        return await asyncio.gather(*tasks)

    async def _process_file(self, file_path: Path, file_name: Optional[str] = None) -> OCRResult:
        """Helper method to process a file path"""
        try:
            with open(file_path, "rb") as f:
                return await self.recognize_async(f, file_name or file_path.name)
        except Exception as e:
            return OCRResult(
                text="", success=False, error=str(e), file_name=file_name or str(file_path)
            )

    # Synchronous wrappers for convenience
    def recognize(
        self, image_data: Union[bytes, BinaryIO, str, Path], file_name: Optional[str] = None
    ) -> OCRResult:
        """
        Synchronous version of recognize_async.
        """
        self._ensure_async_init()

        if isinstance(image_data, (str, Path)):
            image_path = Path(image_data)
            with open(image_path, "rb") as f:
                return self._run_sync(self.recognize_async(f.read(), file_name or image_path.name))
        else:
            if isinstance(image_data, io.IOBase):
                image_data.seek(0)
                image_bytes = image_data.read()
            else:
                image_bytes = image_data
            return self._run_sync(self.recognize_async(image_bytes, file_name))

    def recognize_batch(
        self,
        images: List[Union[bytes, BinaryIO, str, Path]],
        file_names: Optional[List[str]] = None,
    ) -> List[OCRResult]:
        """
        Synchronous version of recognize_batch_async.
        """
        self._ensure_async_init()
        return self._run_sync(self.recognize_batch_async(images, file_names))

    def _run_sync(self, coro):
        """Run an async coroutine synchronously"""
        if self._loop is None:
            self._loop = asyncio.new_event_loop()

        try:
            return self._loop.run_until_complete(coro)
        except Exception as e:
            logger.error(f"Error running coroutine synchronously: {e}")
            raise
