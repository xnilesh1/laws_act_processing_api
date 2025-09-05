import logging
import os
from typing import Annotated

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# Import your processing functions from the other files
from acts import process_acts
from laws_processing import process_laws

# --- 1. Basic Setup ---

# Load environment variables from a .env file
load_dotenv()

# Configure basic logging to see requests and errors in the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- 2. Centralized Configuration ---

# Use Pydantic's BaseSettings to manage configuration in one place
# This class will automatically read variables from the environment
class Settings(BaseSettings):
    """Manages application-wide settings."""
    api_password: str = Field(alias="API_PASSWORD")

    class Config:
        # Allows the model to read from a .env file if load_dotenv() is used
        env_file = ".env"
        extra = "ignore"

# Create a single instance of the settings
settings = Settings()


# --- 3. FastAPI Application Initialization ---

app = FastAPI(
    title="PDF Processing API",
    description="An API to process Act and Law PDF documents from URLs and store them as vector embeddings.",
    version="1.0.0",
)


# --- 4. Security / Authentication ---

async def verify_password(x_api_password: Annotated[str, Header()]):
    """
    Dependency to verify the `x-api-password` header against the server's environment variable.
    """
    if not settings.api_password:
        logger.error("API_PASSWORD environment variable is not set on the server.")
        raise HTTPException(status_code=500, detail="Server configuration error.")
    
    if x_api_password != settings.api_password:
        logger.warning("Invalid API password attempt.")
        raise HTTPException(status_code=401, detail="Invalid or missing API Password")

# A common dependency to apply password verification to all endpoints
ProtectedEndpoint = Depends(verify_password)


# --- 5. API Request and Response Models ---

class ActPayload(BaseModel):
    """Request body for processing an 'Act' document."""
    pdf_link: str = Field(..., description="A direct public URL to the Act PDF file.", example="https://www.indiacode.nic.in/bitstream/123456789/15639/1/a1872-09.pdf")
    acts_page_link: str = Field(..., description="The source URL of the page where the PDF was found.", example="https://www.indiacode.nic.in/handle/123456789/15639")

class LawPayload(BaseModel):
    """Request body for processing a 'Law' document."""
    pdf_link: str = Field(..., description="A direct public URL to the Law PDF file.", example="https://www.un.org/en/genocideprevention/documents/atrocity-crimes/Doc.1_Convention%20on%20the%20Prevention%20and%20Punishment%20of%20the%20Crime%20of%20Genocide.pdf")

class ProcessingResponse(BaseModel):
    """Standard success response model for processing jobs."""
    message: str = Field(..., example="PDF processed successfully.")
    details: str = Field(..., example="This PDF ID is: caseone-acts")


# --- 6. API Endpoints ---

@app.post(
    "/act",
    response_model=ProcessingResponse,
    summary="Process an Act PDF",
    description="Accepts a URL to an Act PDF, processes it, and upserts the vectorized content to Pinecone."
)
async def create_act_processing_job(payload: ActPayload, _=ProtectedEndpoint):
    """
    Processes an Act PDF synchronously. The function will wait until processing is complete.
    """
    try:
        logger.info(f"Starting processing for Act PDF: {payload.pdf_link}")
        result = process_acts(payload.pdf_link, payload.acts_page_link)
        logger.info(f"Successfully processed Act PDF: {payload.pdf_link}")
        return {"message": "Act PDF processed successfully.", "details": result}
    except Exception as e:
        logger.error(f"Error processing Act PDF {payload.pdf_link}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during processing: {str(e)}")


@app.post(
    "/laws",
    response_model=ProcessingResponse,
    summary="Process a Law PDF",
    description="Accepts a URL to a Law PDF, processes it, and upserts the vectorized content to Pinecone."
)
async def create_law_processing_job(payload: LawPayload, _=ProtectedEndpoint):
    """
    Processes a Law PDF synchronously. The function will wait until processing is complete.
    """
    try:
        logger.info(f"Starting processing for Law PDF: {payload.pdf_link}")
        result = process_laws(payload.pdf_link)
        logger.info(f"Successfully processed Law PDF: {payload.pdf_link}")
        return {"message": "Law PDF processed successfully.", "details": result}
    except Exception as e:
        logger.error(f"Error processing Law PDF {payload.pdf_link}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during processing: {str(e)}")


# --- 7. Main Execution Block ---

if __name__ == "__main__":
    # Runs the application using Uvicorn, a fast ASGI server.
    # The host '0.0.0.0' makes the server accessible on your local network.
    uvicorn.run(app)
