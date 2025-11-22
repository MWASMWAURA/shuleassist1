"""Gemini AI implementation for extracting text from student report images."""
import google.generativeai as genai
from pathlib import Path
from PIL import Image
import logging
from config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)

class GeminiExtractor:
    def __init__(self):
        self._available = False
        if GEMINI_API_KEY:
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                # Use the configured model name (from env) so users can override easily
                model_name = GEMINI_MODEL or 'gemini-2.5-flash'
                self.model = genai.GenerativeModel(model_name)
                self._available = True
                logger.info(f"Gemini extractor initialized successfully using model '{model_name}'")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini model '{GEMINI_MODEL}': {e}")
        else:
            logger.warning("GEMINI_API_KEY not found, extractor unavailable")

    def is_available(self):
        return self._available

    def extract_text(self, image_path: str):
        """Extract text from image using Gemini AI.

        Returns a dict with either type 'success' and 'data', or type 'error'.
        """
        if not self._available:
            return {"type": "error", "error": "Gemini API not available"}

        p = Path(image_path)
        if not p.exists():
            return {"type": "error", "error": "File not found"}

        try:
            # Open and process image (use context manager to ensure file is closed)
            extracted_text = None
            with Image.open(image_path) as image:
                # Create prompt for student report extraction
                prompt = """
            Extract information from this student report image. Please provide:
            1. Student name
            2. Grade level
            3. Report year
            4. General comments (if any)
            5. Subject grades and teacher comments (if available)

            Format the response as JSON with these fields:
            - student_name: string
            - grade_level: string
            - report_year: string
            - general_comments: string
            - subjects: array of objects with name, grade, teacher_comment
            - raw_text: the full extracted text

            If information is not available, use empty strings or empty arrays.
            """

                # Generate content
                try:
                    response = self.model.generate_content([prompt, image])
                    # Parse the response (assuming it returns JSON-like text)
                    extracted_text = response.text.strip()
                except Exception as e:
                    # If the model is not available or not supported for this API
                    # return a clear error so the caller can respond appropriately.
                    logger.exception("Model generation error")
                    return {"type": "error", "error": str(e)}

            # Try to parse as JSON, fallback to raw text
            try:
                import json
                data = json.loads(extracted_text)
            except json.JSONDecodeError:
                # Fallback: extract basic info from text
                data = {
                    "student_name": self._extract_field(extracted_text, "student", "name"),
                    "grade_level": self._extract_field(extracted_text, "grade"),
                    "report_year": self._extract_field(extracted_text, "year"),
                    "general_comments": self._extract_field(extracted_text, "comments"),
                    "subjects": [],
                    "raw_text": extracted_text
                }

            logger.info(f"Successfully extracted data for {p.name}")
            return {"type": "success", "data": data}

        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}")
            return {"type": "error", "error": f"Extraction failed: {str(e)}"}

    def extract_bytes(self, image_bytes: bytes):
        """Extract using raw image bytes (avoids writing temp files)."""
        from io import BytesIO

        if not self._available:
            return {"type": "error", "error": "Gemini API not available"}

        try:
            extracted_text = None
            with Image.open(BytesIO(image_bytes)) as image:
                prompt = """
                Extract information from this student report image. Please provide:
                1. Student name
                2. Grade level
                3. Report year
                4. General comments (if any)
                5. Subject grades and teacher comments (if available)

                Format the response as JSON with these fields:
                - student_name: string
                - grade_level: string
                - report_year: string
                - general_comments: string
                - subjects: array of objects with name, grade, teacher_comment
                - raw_text: the full extracted text

                If information is not available, use empty strings or empty arrays.
                """

                try:
                    response = self.model.generate_content([prompt, image])
                    extracted_text = response.text.strip()
                except Exception as e:
                    logger.exception("Model generation error")
                    return {"type": "error", "error": str(e)}

            try:
                import json
                data = json.loads(extracted_text)
            except Exception:
                data = {
                    "student_name": self._extract_field(extracted_text, "student", "name"),
                    "grade_level": self._extract_field(extracted_text, "grade"),
                    "report_year": self._extract_field(extracted_text, "year"),
                    "general_comments": self._extract_field(extracted_text, "comments"),
                    "subjects": [],
                    "raw_text": extracted_text
                }

            logger.info("Successfully extracted data from bytes input")
            return {"type": "success", "data": data}
        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}")
            return {"type": "error", "error": f"Extraction failed: {str(e)}"}

    def _extract_field(self, text: str, *keywords):
        """Simple field extraction from text."""
        # Basic implementation - could be improved
        text_lower = text.lower()
        for keyword in keywords:
            if keyword in text_lower:
                # Find the line containing the keyword
                lines = text.split('\n')
                for line in lines:
                    if keyword in line.lower():
                        return line.strip()
        return ""
