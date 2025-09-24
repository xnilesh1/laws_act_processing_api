from src.query.embeddings_chat import execute_query_acts, execute_query_laws
# Function schemas for the Gemini model, defined as dictionaries
query_acts_schema = {
    "name": "query_acts",
    "description": "Search the comprehensive Indian Acts and Statutes vector database containing all Indian legal acts, amendments, provisions, and legislative documents.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search the comprehensive Indian Acts and Statutes vector database containing all Indian legal acts, amendments, provisions, and legislative documents.",
            }
        },
        "required": ["query"],
    },
}

query_laws_schema = {
    "name": "query_laws",
    "description": "Search the comprehensive Indian Acts and Statutes vector database containing all Indian legal acts, amendments, provisions, and legislative documents.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search the comprehensive Indian Acts and Statutes vector database containing all Indian legal acts, amendments, provisions, and legislative documents.",
            },
        },
        "required": ["query"],
    },
}



# A dictionary to map function names to the actual functions
AVAILABLE_TOOLS = {
    "query_acts": execute_query_acts,
    "query_laws": execute_query_laws,
} 