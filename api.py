import logging
import os
from functools import wraps

from dotenv import load_dotenv
from flask import Flask, request
from flask_restx import Api, Resource, reqparse, fields
from pydantic import Field
from pydantic_settings import BaseSettings
from werkzeug.exceptions import Unauthorized, InternalServerError

# Import your existing processing functions
from acts import process_acts
from laws_processing import process_laws

# --- 1. Basic Setup ---

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- 2. Centralized Configuration (Same as before) ---

class Settings(BaseSettings):
    """Manages application-wide settings."""
    api_password: str = Field(alias="API_PASSWORD")

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()


# --- 3. Flask & Flask-RESTX Application Initialization ---

app = Flask(__name__)
api = Api(app,
          version='1.0',
          title='PDF Processing API',
          description='An API to process Act and Law PDF documents from URLs and store them as vector embeddings.',
          doc='/')  # Serve Swagger UI at the root URL

# Define a namespace for the API endpoints
ns = api.namespace('processing', description='PDF Processing operations')


# --- 4. Security / Authentication ---

def require_api_key(f):
    """Decorator to protect endpoints with an API key."""
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('x-api-password')
        if not settings.api_password:
            logger.error("API_PASSWORD environment variable is not set on the server.")
            raise InternalServerError("Server configuration error.")
        if not api_key or api_key != settings.api_password:
            logger.warning("Invalid API password attempt.")
            raise Unauthorized("Invalid or missing API Password")
        return f(*args, **kwargs)
    return decorated


# --- 5. API Request and Response Models ---

# Request parsers for input validation
act_parser = reqparse.RequestParser()
act_parser.add_argument('pdf_link', type=str, required=True, help='A direct public URL to the Act PDF file.', location='json')
act_parser.add_argument('acts_page_link', type=str, required=True, help='The source URL of the page where the PDF was found.', location='json')

law_parser = reqparse.RequestParser()
law_parser.add_argument('pdf_link', type=str, required=True, help='A direct public URL to the Law PDF file.', location='json')

# Response model for consistent output structure
processing_response_model = api.model('ProcessingResponse', {
    'message': fields.String(required=True, description='Status message of the processing job.'),
    'details': fields.String(required=True, description='Details returned from the processing job.')
})


# --- 6. API Endpoints ---

@ns.route('/act')
class ActResource(Resource):
    @ns.doc('process_act', description="Accepts a URL to an Act PDF, processes it, and upserts the vectorized content.")
    @ns.expect(act_parser)
    @ns.marshal_with(processing_response_model)
    @require_api_key
    def post(self):
        """Process an Act PDF"""
        args = act_parser.parse_args()
        pdf_link = args['pdf_link']
        acts_page_link = args['acts_page_link']
        try:
            logger.info(f"Starting processing for Act PDF: {pdf_link}")
            result = process_acts(pdf_link, acts_page_link)
            logger.info(f"Successfully processed Act PDF: {pdf_link}")
            return {'message': 'Act PDF processed successfully.', 'details': result}, 200
        except Exception as e:
            logger.error(f"Error processing Act PDF {pdf_link}: {e}", exc_info=True)
            api.abort(500, f"An error occurred during processing: {str(e)}")


@ns.route('/laws')
class LawResource(Resource):
    @ns.doc('process_law', description="Accepts a URL to a Law PDF, processes it, and upserts the vectorized content.")
    @ns.expect(law_parser)
    @ns.marshal_with(processing_response_model)
    @require_api_key
    def post(self):
        """Process a Law PDF"""
        args = law_parser.parse_args()
        pdf_link = args['pdf_link']
        try:
            logger.info(f"Starting processing for Law PDF: {pdf_link}")
            result = process_laws(pdf_link)
            logger.info(f"Successfully processed Law PDF: {pdf_link}")
            return {'message': 'Law PDF processed successfully.', 'details': result}, 200
        except Exception as e:
            logger.error(f"Error processing Law PDF {pdf_link}: {e}", exc_info=True)
            api.abort(500, f"An error occurred during processing: {str(e)}")


# --- 7. Main Execution Block ---

if __name__ == "__main__":
    # Read the port from the environment variable, with a default for local development
    port = int(os.environ.get("PORT", 5000))
    # Note: Flask's development server is not for production.
    # Use a production-ready WSGI server like Gunicorn or Waitress.
    app.run(host="0.0.0.0", port=port, debug=False)
