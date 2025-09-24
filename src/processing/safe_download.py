import os
import tempfile
import shutil
import logging
from typing import Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


logger = logging.getLogger(__name__)


def _build_session(
    total_retries: int = 5,
    backoff_factor: float = 0.5,
    status_forcelist: Tuple[int, ...] = (429, 500, 502, 503, 504),
) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        status=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _is_probable_pdf(content_type: Optional[str], first_bytes: bytes) -> bool:
    if content_type and "pdf" in content_type.lower():
        return True
    return first_bytes.startswith(b"%PDF-")


def download_pdf(
    url: str,
    output_path: str = "downloaded_file.pdf",
    timeout: Tuple[float, float] = (10.0, 60.0),
    max_bytes: int = 100 * 1024 * 1024,
) -> str:
    """
    Download a PDF from a URL and save it atomically as output_path.

    - Streams response to avoid memory spikes
    - Retries with exponential backoff for transient errors
    - Validates Content-Type/first bytes for PDF
    - Enforces a maximum size limit
    - Writes to a temp file and renames atomically

    Returns the final output_path on success.
    Raises exceptions on failure with meaningful messages.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept": "application/pdf,application/octet-stream,*/*;q=0.8",
        "Connection": "keep-alive",
    }

    session = _build_session()

    # Ensure target directory exists
    target_dir = os.path.dirname(output_path) or "."
    os.makedirs(target_dir, exist_ok=True)

    # Create temp file in same filesystem for atomic rename
    temp_fd, temp_path = tempfile.mkstemp(prefix="download_", suffix=".pdf", dir=target_dir)
    os.close(temp_fd)

    bytes_written = 0
    first_chunk = b""
    response = None
    try:
        response = session.get(url, headers=headers, stream=True, timeout=timeout)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                expected = int(content_length)
                if expected > max_bytes:
                    raise ValueError(f"Remote file too large: {expected} bytes > limit {max_bytes}")
            except ValueError:
                # Ignore invalid Content-Length; continue with streaming checks
                pass

        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                if not first_chunk:
                    first_chunk = chunk[:8]
                    # Validate early for PDF signature/content-type
                    if not _is_probable_pdf(content_type, first_chunk):
                        raise ValueError("Downloaded content does not appear to be a PDF")

                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    raise ValueError(f"Downloaded file exceeds size limit of {max_bytes} bytes")
                f.write(chunk)

        # Optional: if Content-Length was valid, verify it matches
        if content_length is not None and content_length.isdigit():
            if int(content_length) != bytes_written:
                raise ValueError(
                    f"Size mismatch: expected {content_length} bytes, got {bytes_written} bytes"
                )

        # Atomic replace
        shutil.move(temp_path, output_path)
        return output_path

    except requests.Timeout as e:
        logger.error(f"Timeout downloading {url}: {e}")
        raise
    except requests.RequestException as e:
        logger.error(f"Network error downloading {url}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to download PDF from {url}: {e}")
        raise
    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
        # Clean up temp file if it still exists
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except Exception:
            pass


def download_pdf_as_default(url: str) -> str:
    """Convenience wrapper that always saves to 'downloaded_file.pdf'."""
    return download_pdf(url, output_path="downloaded_file.pdf")


# if __name__ == "__main__":
#     # Simple manual test
#     path = download_pdf_as_default("https://pdfobject.com/pdf/sample.pdf")
#     print(path)


