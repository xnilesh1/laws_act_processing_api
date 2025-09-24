import logging
import time
import uvicorn
from langsmith.run_trees import RunTree
from fastapi import FastAPI, Depends, HTTPException, status, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from src.processing.main import process_laws, process_acts
from src.config import API_PASSWORD
from src.query.chat import ChatRequest, get_chat_session, ChatResponse
from google.genai import types as genai_types
from google.genai import protos
from src.query.tools import AVAILABLE_TOOLS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Legal Documents API",
    description="An API for processing legal documents and chatting.",
    version="1.0.0",
)


# --- Pydantic Models for Request Bodies ---
class LawsProcessRequest(BaseModel):
    pdf_link: str


class ActsProcessRequest(BaseModel):
    pdf_link: str
    acts_page_link: str


# --- Authentication ---
async def verify_password(Authorization: str = Header(None)):
    """FastAPI dependency to verify the password in the Authorization header."""
    if Authorization != API_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required: Include Authorization header with the correct password.",
        )


# --- API Endpoints ---

@app.get('/health')
def health_check():
    """Health check endpoint."""
    return {
        'status': 'healthy',
        'message': 'Legal Documents API is running',
        'timestamp': time.time()
    }


@app.post('/process/laws', dependencies=[Depends(verify_password)])
def process_laws_endpoint(request_data: LawsProcessRequest):
    """
    Process a laws PDF document.
    """
    try:
        pdf_link = request_data.pdf_link
        logger.info(f"Starting laws processing for: {pdf_link}")
        start_time = time.time()

        # Process the PDF (this will wait until completion)
        result = process_laws(pdf_link)

        processing_time = time.time() - start_time
        logger.info(f"Laws processing completed in {processing_time:.2f} seconds")

        return {
            'success': True,
            'message': result,
            'pdf_link': pdf_link,
            'processing_time_seconds': round(processing_time, 2)
        }

    except Exception as e:
        logger.error(f"Error processing laws PDF: {str(e)}")
        # Using JSONResponse to set status code for exceptions
        return JSONResponse(
            status_code=500,
            content={
                'success': False,
                'error': 'Processing failed',
                'message': str(e),
                'pdf_link': request_data.pdf_link if 'request_data' in locals() else None
            }
        )


@app.post('/process/acts', dependencies=[Depends(verify_password)])
def process_acts_endpoint(request_data: ActsProcessRequest):
    """
    Process an acts PDF document.
    """
    try:
        pdf_link = request_data.pdf_link
        acts_page_link = request_data.acts_page_link

        logger.info(f"Starting acts processing for: {pdf_link}")
        start_time = time.time()

        # Process the PDF (this will wait until completion)
        result = process_acts(pdf_link, acts_page_link)

        processing_time = time.time() - start_time
        logger.info(f"Acts processing completed in {processing_time:.2f} seconds")

        return {
            'success': True,
            'message': result,
            'pdf_link': pdf_link,
            'acts_page_link': acts_page_link,
            'processing_time_seconds': round(processing_time, 2)
        }

    except Exception as e:
        logger.error(f"Error processing acts PDF: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                'success': False,
                'error': 'Processing failed',
                'message': str(e),
                'pdf_link': request_data.pdf_link if 'request_data' in locals() else None,
                'acts_page_link': request_data.acts_page_link if 'request_data' in locals() else None
            }
        )


@app.post('/query/chat', response_model=ChatResponse, dependencies=[Depends(verify_password)])
async def chat_with_bot(request: ChatRequest):
    """
    Handles a chat request with the bot, maintaining session history.
    """
    top_p = 3  # Example value, you can adjust as needed

    parent_run = RunTree(
        name="Chat Session",
        run_type="chain",
        inputs={"question": request.question},
        extra={"metadata": {"session_id": request.session_id}},
    )
    total_prompt_tokens = 0
    total_candidates_tokens = 0

    try:
        chat = get_chat_session(request.session_id, request.model)

        # Truncate history based on top_p before sending the new message
        if top_p is not None:
            if top_p > 0:
                # A "chat" is a user message and a model response, so 2 history entries.
                messages_to_keep = top_p * 2
                if len(chat.history) > messages_to_keep:
                    chat.history = chat.history[-messages_to_keep:]
            elif top_p == 0:
                # Clear history if top_p is 0
                chat.history = []

        generation_config = genai_types.GenerationConfig(
            max_output_tokens=request.max_token
        )

        llm_run = parent_run.create_child(
            name="Gemini Call",
            run_type="llm",
            inputs={"question": request.question, "history_length": len(chat.history)},
        )
        response = chat.send_message(
            request.question,
            generation_config=generation_config
        )
        prompt_tokens = response.usage_metadata.prompt_token_count
        candidates_tokens = response.usage_metadata.candidates_token_count
        total_prompt_tokens += prompt_tokens
        total_candidates_tokens += candidates_tokens
        llm_run.end(outputs=response, metadata={
            "prompt_token_count": prompt_tokens,
            "candidates_token_count": candidates_tokens,
            "total_token_count": response.usage_metadata.total_token_count
        })

        # Handle function calls
        if response.candidates and response.candidates[0].content.parts and any(
                p.function_call for p in response.candidates[0].content.parts):
            tool_responses = []
            tool_run = parent_run.create_child(name="Tool Calling", run_type="tool")
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    function_call = part.function_call
                    function_name = function_call.name
                    if function_name in AVAILABLE_TOOLS:
                        function_to_call = AVAILABLE_TOOLS[function_name]
                        function_args = dict(function_call.args)
                        function_response = function_to_call(**function_args)

                        tool_responses.append(protos.Part(
                            function_response=protos.FunctionResponse(
                                name=function_name,
                                response=function_response
                            )
                        ))
            tool_run.end(outputs={"tool_responses": tool_responses})

            if tool_responses:
                tool_llm_run = parent_run.create_child(
                    name="Gemini Call with Tools",
                    run_type="llm",
                    inputs={"tool_responses": tool_responses},
                )
                response = chat.send_message(tool_responses)
                prompt_tokens_tool = response.usage_metadata.prompt_token_count
                candidates_tokens_tool = response.usage_metadata.candidates_token_count
                total_prompt_tokens += prompt_tokens_tool
                total_candidates_tokens += candidates_tokens_tool
                tool_llm_run.end(outputs=response, metadata={
                    "prompt_token_count": prompt_tokens_tool,
                    "candidates_token_count": candidates_tokens_tool,
                    "total_token_count": response.usage_metadata.total_token_count
                })

        final_response = response.text

        # Clean history for future calls by removing tool-related messages
        clean_history = []
        for message in chat.history:
            text_parts = [part for part in message.parts if part.text]
            if text_parts:
                clean_message = protos.Content(role=message.role, parts=text_parts)
                clean_history.append(clean_message)

        chat.history = clean_history

        parent_run.end(outputs={"response": final_response}, metadata={
            "total_prompt_tokens": total_prompt_tokens,
            "total_candidates_tokens": total_candidates_tokens,
            "total_tokens": total_prompt_tokens + total_candidates_tokens,
        })
        return ChatResponse(response=final_response)

    except Exception as e:
        parent_run.end(error=str(e))
        # Re-raise as HTTPException to be handled by FastAPI
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
    finally:
        # Ensure the run is always posted
        parent_run.post()


if __name__ == '__main__':
    uvicorn.run(app,
                host="0.0.0.0",
                port=8000)
