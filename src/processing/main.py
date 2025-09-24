import gc
import json
import logging
import multiprocessing
import os
import tempfile
import time
from abc import ABC, abstractmethod
from typing import Dict
from urllib.parse import urlparse

from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Handle Pinecone imports with fallback
try:
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone
    USE_LANGCHAIN_PINECONE = True
except ImportError as e:
    print(f"Warning: langchain-pinecone not available ({e}). Using direct Pinecone client.")
    from pinecone import Pinecone
    USE_LANGCHAIN_PINECONE = False
from dotenv import load_dotenv

# Import your existing functions
from src.processing.safe_download import download_pdf
from src.processing.verify_pdf import verify_pdf_file
from src.config import LAWS_INDEX, LAWS_NAMESPACE, ACTS_INDEX, ACTS_NAMESPACE


load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Configuration constants
BATCH_SIZE = 100
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
MAX_UPSERT_RETRIES = 3
INITIAL_UPSERT_DELAY = 1.0


class BaseLegalDocumentProcessor(ABC):
    """Base class for processing legal documents into vector embeddings."""
    
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.environ["GOOGLE_API_KEY"],
        )
        self.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        
    @property
    @abstractmethod
    def index_name(self) -> str:
        """Return the Pinecone index name for this processor type."""
        pass
    
    @property
    @abstractmethod
    def namespace(self) -> str:
        """Return the namespace for this processor type."""
        pass
    
    @abstractmethod
    def prepare_metadata(self, split_doc, **kwargs) -> None:
        """Prepare document metadata before upserting."""
        pass
    
    def safe_upsert(self, vector_store_or_index, batch):
        """Attempt to upsert a batch with retries and exponential backoff."""
        for attempt in range(MAX_UPSERT_RETRIES):
            try:
                if USE_LANGCHAIN_PINECONE:
                    vector_store_or_index.add_documents(documents=batch)
                else:
                    # Direct Pinecone upsert - convert documents to vectors
                    vectors = []
                    for doc in batch:
                        # Generate embedding for the document
                        embedding = self.embeddings.embed_query(doc.page_content)
                        vectors.append({
                            'id': f"{doc.metadata.get('document_name', 'doc')}_{len(vectors)}",
                            'values': embedding,
                            'metadata': {
                                **doc.metadata,
                                'text': doc.page_content
                            }
                        })
                    vector_store_or_index.upsert(vectors=vectors, namespace=self.namespace)
                return
            except Exception as e:
                logger.warning(
                    f"Upsert attempt {attempt+1}/{MAX_UPSERT_RETRIES} failed: {e}"
                )
                time.sleep(INITIAL_UPSERT_DELAY * (2**attempt))
        raise RuntimeError("Failed to upsert batch after retries")

    def process_pdf_safely(self, loader):
        """Safely process PDF pages with proper resource management."""
        try:
            pages = loader.load()
            # Add page numbers to metadata (starting from 1, not 0)
            for page in pages:
                page.metadata["page"] = page.metadata.get("page", 0) + 1
            return pages
        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            raise
        finally:
            # Ensure the loader's resources are cleaned up
            if hasattr(loader, "pdf_reader") and hasattr(loader.pdf_reader, "stream"):
                try:
                    loader.pdf_reader.stream.close()
                except Exception as e:
                    logger.warning(f"Error closing PDF reader stream: {e}")

    def pdf_processor_subprocess(self, pdf_path: str, params: Dict, result_file: str):
        """Process PDF in a separate subprocess and save results to temp file."""
        try:
            # Set up vector store or direct index
            index = self.pc.Index(self.index_name)
            
            if USE_LANGCHAIN_PINECONE:
                vector_store_or_index = PineconeVectorStore(
                    embedding=self.embeddings, 
                    index=index, 
                    namespace=self.namespace
                )
            else:
                vector_store_or_index = index
            
            # Load and process PDF
            loader = PyPDFLoader(file_path=pdf_path)
            docs = self.process_pdf_safely(loader)
            
            # Configure text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                add_start_index=True,
            )
            
            # Batch-split and upsert documents for memory efficiency
            batch_size = int(os.getenv("PINECONE_BATCH_SIZE", str(BATCH_SIZE)))
            batch = []
            total_splits = 0
            
            for page in docs:
                splits = text_splitter.split_documents([page])
                total_splits += len(splits)
                for split in splits:
                    # Apply processor-specific metadata preparation
                    self.prepare_metadata(split, **params)
                    batch.append(split)
                    if len(batch) >= batch_size:
                        self.safe_upsert(vector_store_or_index, batch)
                        batch = []
                        gc.collect()
            
            # Flush remaining splits
            if batch:
                self.safe_upsert(vector_store_or_index, batch)
                batch = []
                gc.collect()
            
            # Write success result to temp file
            with open(result_file, 'w') as f:
                json.dump({
                    "success": True, 
                    "total_splits": total_splits, 
                    "pages": len(docs)
                }, f)
                
        except Exception as e:
            # Write error to temp file
            with open(result_file, 'w') as f:
                json.dump({"success": False, "error": str(e)}, f)
            logger.error(f"Subprocess error processing PDF: {e}")

    def process_document(self, pdf_link: str, **kwargs) -> str:
        """Main method to process a PDF document from a URL."""
        try:
            # Download PDF safely
            temp_pdf_path = None
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                temp_pdf_path = temp_file.name
            
            try:
                # Download the PDF
                download_pdf(pdf_link, output_path=temp_pdf_path)
                
                # Verify PDF integrity
                is_valid, message = verify_pdf_file(temp_pdf_path)
                if not is_valid:
                    raise ValueError(f"PDF validation failed: {message}")
                
                # Extract filename from URL for metadata
                original_filename = self._extract_filename_from_url(pdf_link)
                
                # Create temporary file for subprocess results
                with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as result_file:
                    result_path = result_file.name
                
                # Parameters to pass to the subprocess
                params = {
                    'original_filename': original_filename,
                    **kwargs
                }
                
                # Start the subprocess
                process = multiprocessing.Process(
                    target=self.pdf_processor_subprocess,
                    args=(temp_pdf_path, params, result_path)
                )
                
                logger.info(f"Starting PDF processing subprocess for {self.__class__.__name__}")
                process.start()
                process.join()  # Wait for process to complete
                
                # Check the results from the subprocess
                with open(result_path, 'r') as f:
                    result = json.load(f)
                
                # Clean up the temporary result file
                try:
                    os.unlink(result_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary result file: {e}")
                
                if not result.get('success', False):
                    raise RuntimeError(f"Subprocess failed: {result.get('error', 'Unknown error')}")
                
                # Log processing summary
                logger.info(
                    f"Processed {result.get('pages', 0)} pages into {result.get('total_splits', 0)} "
                    f"chunks using {self.__class__.__name__}"
                )
                
                return f"Successfully processed PDF into {self.namespace} namespace"
                
            finally:
                # Clean up downloaded PDF
                if temp_pdf_path and os.path.exists(temp_pdf_path):
                    try:
                        os.unlink(temp_pdf_path)
                    except Exception as e:
                        logger.warning(f"Failed to delete temporary PDF file: {e}")
                        
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise

    def _extract_filename_from_url(self, url: str) -> str:
        """Extract filename from URL for metadata."""
        try:
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            return filename if filename else "Unknown_Document"
        except Exception:
            return "Unknown_Document"


class LawsProcessor(BaseLegalDocumentProcessor):
    """Processor for legal laws documents."""
    
    @property
    def index_name(self) -> str:
        return LAWS_INDEX
    
    @property
    def namespace(self) -> str:
        return LAWS_NAMESPACE
    
    def prepare_metadata(self, split_doc, **kwargs):
        """Prepare metadata for laws documents."""
        # Add original filename if provided
        if 'original_filename' in kwargs:
            split_doc.metadata['document_name'] = kwargs['original_filename']
        
        # Page number is already added in process_pdf_safely method
        # Ensure it exists and starts from 1
        if 'page' not in split_doc.metadata:
            split_doc.metadata['page'] = 1


class ActsProcessor(BaseLegalDocumentProcessor):
    """Processor for legal acts documents."""
    
    @property
    def index_name(self) -> str:
        return ACTS_INDEX
    
    @property
    def namespace(self) -> str:
        return ACTS_NAMESPACE
    
    def prepare_metadata(self, split_doc, **kwargs):
        """Prepare metadata for acts documents."""
        # Add original filename if provided
        if 'original_filename' in kwargs:
            split_doc.metadata['document_name'] = kwargs['original_filename']
        
        # Add acts page link (required for acts)
        if 'acts_page_link' in kwargs:
            split_doc.metadata['acts_page_link'] = kwargs['acts_page_link']
        else:
            raise ValueError("acts_page_link is required for ActsProcessor")
        
        # Page number is already added in process_pdf_safely method
        # Ensure it exists and starts from 1
        if 'page' not in split_doc.metadata:
            split_doc.metadata['page'] = 1


class LegalDocumentProcessorFactory:
    """Factory class to create appropriate processors."""
    
    @staticmethod
    def create_laws_processor() -> LawsProcessor:
        """Create a laws processor instance."""
        return LawsProcessor()
    
    @staticmethod
    def create_acts_processor() -> ActsProcessor:
        """Create an acts processor instance."""
        return ActsProcessor()

def process_acts(pdf_link: str, acts_page_link: str) -> str:
    """Convenience function to process acts documents."""
    processor = LegalDocumentProcessorFactory.create_acts_processor()
    return processor.process_document(pdf_link, acts_page_link=acts_page_link)


def process_laws(pdf_link: str) -> str:
    """Convenience function to process laws documents."""
    processor = LegalDocumentProcessorFactory.create_laws_processor()
    return processor.process_document(pdf_link)
