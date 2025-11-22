from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from extractor import GeminiExtractor
from config import GEMINI_API_KEY
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Student Report Extractor API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    # Allow frontend dev servers (3000/3001) and local testing. Use stricter
    # origins in production.
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

extractor = GeminiExtractor()

@app.post("/api/extract")
async def extract_text(image: UploadFile = File(...)):
    logger.info(f"Received extraction request for file: {image.filename}")

    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read uploaded bytes and perform in-memory extraction (avoids file locking)
        content = await image.read()
        logger.info(f"Read uploaded {len(content)} bytes from {image.filename}")

        # Extract text using Gemini (in-memory)
        # Prefer extractor.extract_bytes when available
        if hasattr(extractor, 'extract_bytes'):
            result = extractor.extract_bytes(content)
        else:
            # Fallback: write a temp file as before
            import tempfile, os
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name
            logger.info(f"Saved image to temporary file: {temp_path}")
            result = extractor.extract_text(temp_path)
            try:
                os.unlink(temp_path)
            except Exception:
                logger.warning("Could not delete temp file")

        logger.info(f"Extraction result: {result}")

        if result["type"] == "success":
            return {"success": True, "data": result["data"]}
        else:
            raise HTTPException(status_code=500, detail=result["error"])

    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "extractor_available": extractor.is_available()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)