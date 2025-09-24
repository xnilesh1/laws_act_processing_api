import os
import uuid
from typing import Dict, Any, Optional
import google.generativeai as genai
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google.generativeai import types as genai_types



# Assuming 'query' is a module in the same project root
from src.query.tools import query_acts_schema, query_laws_schema
from src.query.prompts import system_prompt
from src.config import CHAT_MODEL, CHAT_TOP_P


# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- LangSmith Configuration ---
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "Caseone AI Chatbot")

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique session ID for the chat. A new one is created if not provided.",
    )
    question: str = Field(..., description="The question from the user.")
    model: str = Field(
        CHAT_MODEL,
        description="The model to use for the chat.",
    )
    max_token: int = Field(2048, description="Maximum tokens for the response.")
    top_p: Optional[int] = Field(
        None,
        description="Number of recent chat exchanges to keep in history (e.g., 3 for the last 3 Q&As). If 0, history is cleared. If not provided, all history is kept.",
    )

class ChatResponse(BaseModel):
    response: str

# --- Gemini Model and Chat Management ---
legal_tools = [
    genai_types.Tool(function_declarations=[
        query_acts_schema,
        query_laws_schema,
    ])
]

generation_config = genai_types.GenerationConfig(
    temperature=0.2,
    top_p=CHAT_TOP_P
    # max_output_tokens is the parameter for controlling response length
    # but we will use the one from request
)

model = genai.GenerativeModel(
    model_name=CHAT_MODEL,  # Using 1.5 flash as it's generally better
    system_instruction=system_prompt,
    tools=legal_tools,
    generation_config=generation_config,
)

# In-memory store for chat sessions
chat_sessions: Dict[str, Any] = {}

def get_chat_session(session_id: str, model_name: str = CHAT_MODEL):
    if session_id not in chat_sessions:
        # New session, create model and chat
        active_model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt,
            tools=legal_tools
        )
        chat_sessions[session_id] = {"model_name": model_name, "chat": active_model.start_chat(history=[])}

    session = chat_sessions[session_id]

    if session["model_name"] != model_name:
        # Model has changed. Create new chat, history from previous model is lost for this session.
        # We can take the history from old chat and pass to new one.
        old_history = session["chat"].history
        active_model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt,
            tools=legal_tools
        )
        chat_sessions[session_id] = {"model_name": model_name, "chat": active_model.start_chat(history=old_history)}

    return chat_sessions[session_id]["chat"]
