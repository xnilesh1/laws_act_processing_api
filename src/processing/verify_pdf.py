import os
from typing import Tuple
from pypdf import PdfReader
from pypdf.errors import PdfReadError
import logging

logger = logging.getLogger(__name__)



def _scan_for_active_content(file_path: str) -> Tuple[bool, str]:
    """
    Scan the raw PDF bytes for indicators of active/unsafe content.
    Returns (is_safe, message). If unsafe, message describes why.
    """
    indicators = [
        b"/JS",
        b"/JavaScript",
        b"/AA",           # additional actions
        b"/OpenAction",
        b"/Launch",
        b"/EmbeddedFile",
        b"/RichMedia",
    ]
    try:
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                for token in indicators:
                    if token in chunk:
                        return False, "PDF contains active content (JavaScript/Embedded/OpenAction)"
        return True, "OK"
    except Exception as e:
        return False, f"Failed during active content scan: {e}"


def verify_pdf_file(file_path: str) -> Tuple[bool, str]:
    """
    Verify a PDF file is valid and safe to process.

    Returns (True, "OK") if valid; otherwise (False, reason).
    Reasons include: not found, empty, bad header, encrypted, corrupted,
    unsafe content, or parse errors.
    """
    try:
        if not os.path.exists(file_path):
            return False, "File not found"
        if not os.path.isfile(file_path):
            return False, "Path is not a file"
        if os.path.getsize(file_path) == 0:
            return False, "File is empty"

        # Check PDF magic header
        with open(file_path, "rb") as f:
            header = f.read(8)
        if not header.startswith(b"%PDF-"):
            return False, "Missing %PDF- header"

        # Quick scan for potentially unsafe active content
        is_safe, safe_msg = _scan_for_active_content(file_path)
        if not is_safe:
            return False, safe_msg

        # Try to parse with pypdf
        try:
            reader = PdfReader(file_path, strict=False)
        except PdfReadError as e:
            return False, f"Corrupted PDF: {e}"
        except Exception as e:
            return False, f"Failed to open PDF: {e}"

        if getattr(reader, "is_encrypted", False):
            # Do not attempt to decrypt; treat as unsecure for this pipeline
            return False, "PDF is encrypted (unsecure for processing)"

        # Force parse of basic structures
        try:
            num_pages = len(reader.pages)
            if num_pages <= 0:
                return False, "PDF has no pages"
            # Try accessing first page content to ensure readable
            _ = reader.pages[0]
        except PdfReadError as e:
            return False, f"Corrupted PDF while reading pages: {e}"
        except Exception as e:
            return False, f"Failed while inspecting pages: {e}"

        return True, "OK"

    except Exception as e:
        logger.error(f"Unexpected verification error: {e}")
        return False, f"Unexpected error: {e}"

# if __name__ == "__main__":
#     a = verify_pdf_file("/home/nilesh/code/laws_act_processing_api/src/SpacePDF_corrupted_downloaded_file.pdf")
#     print(a)