import os
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, HttpUrl
from typing import Optional
import uvicorn

from acts import process_acts
from laws_processing import process_laws
from dotenv import load_dotenv

load_dotenv()

API_PASSWORD = os.getenv("API_PASSWORD")

app = FastAPI()

# Password checking dependency
async def verify_password(x_api_password: str = Header(None)):
    if not API_PASSWORD:
        raise HTTPException(status_code=500, detail="API_PASSWORD environment variable not set on the server.")
    if not x_api_password:
        raise HTTPException(status_code=401, detail="x-api-password header is missing.")
    if x_api_password != API_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid API Password")

class ActPayload(BaseModel):
    pdf_link: str
    acts_page_link: str

class LawPayload(BaseModel):
    pdf_link: str

@app.post("/act", dependencies=[Depends(verify_password)])
async def create_act_processing_job(payload: ActPayload):
    """
    Accepts an act PDF for processing.
    The processing is done synchronously.
    """
    try:
        result = process_acts(payload.pdf_link, payload.acts_page_link)
        return {"message": "Act PDF processed successfully.", "details": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during processing: {str(e)}")


@app.post("/laws", dependencies=[Depends(verify_password)])
async def create_law_processing_job(payload: LawPayload):
    """
    Accepts a law PDF for processing.
    The processing is done synchronously.
    """
    try:
        result = process_laws(payload.pdf_link)
        return {"message": "Law PDF processed successfully.", "details": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during processing: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
