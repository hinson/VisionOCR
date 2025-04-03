import logging
from typing import Optional

from Cocoa import NSData, NSMutableDictionary
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from Quartz import (
    CGImageSourceCreateImageAtIndex,
    CGImageSourceCreateWithData,
    kCGImageSourceShouldCache,
)
from Vision import (
    VNImageRequestHandler,
    VNRecognizeTextRequest,
    VNRequestTextRecognitionLevelAccurate,
)

app = FastAPI(
    title="macOS Vision OCR API",
    description="A web service that performs OCR using macOS's Vision framework",
    version="1.0.0",
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCRResponse(BaseModel):
    text: str
    success: bool
    error: Optional[str] = None


def nsdata_to_cgimage(nsdata):
    """Convert NSData to a CGImage using Core Graphics"""
    options = NSMutableDictionary.dictionary()
    options.setObject_forKey_(False, kCGImageSourceShouldCache)

    data_provider = CGImageSourceCreateWithData(nsdata, options)
    if data_provider is None:
        raise ValueError("Could not create image source from data")

    cg_image = CGImageSourceCreateImageAtIndex(data_provider, 0, options)
    if cg_image is None:
        raise ValueError("Could not create CGImage from image source")

    return cg_image


def perform_ocr(image_data, lang=None):
    """Perform OCR on image data using Vision framework"""
    try:
        # Convert bytes to NSData
        ns_data = NSData.dataWithBytes_length_(image_data, len(image_data))

        # Create CGImage from NSData
        cg_image = nsdata_to_cgimage(ns_data)
        if cg_image is None:
            raise ValueError("Failed to create CGImage from data")

        # Create Vision request handler
        handler = VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, None)

        # Create text recognition request
        request = VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(VNRequestTextRecognitionLevelAccurate)
        if lang is not None:
            request.setRecognitionLanguages_([lang])

        # Perform the request
        success, error = handler.performRequests_error_([request], None)

        if not success:
            raise RuntimeError(
                f"Vision request failed: {error}" if error else "Unknown Vision error"
            )

        # Extract results
        observations = request.results()
        if not observations:
            return ""

        text_results = []
        for observation in observations:
            text = observation.topCandidates_(1)[0].string()
            text_results.append(text)

        return "\n".join(text_results)

    except Exception as e:
        logger.error(f"OCR processing failed: {str(e)}")
        raise


@app.post("/ocr", response_model=OCRResponse)
async def process_image(file: UploadFile = File(...), lang: Optional[str] = None):
    """Endpoint that accepts an image file and returns OCR results"""
    try:
        # Read the uploaded file
        image_data = await file.read()
        if not image_data:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Perform OCR
        text = perform_ocr(image_data, lang)

        # Return the result
        return OCRResponse(text=text, success=True)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return OCRResponse(text="", success=False, error=f"Processing failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9394)
