import structlog
from typing import List, Dict, Tuple
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_transformers.openai_functions import create_metadata_tagger
from langchain_groq import ChatGroq
from langchain.schema import Document
from config import settings
import hashlib

logger = structlog.get_logger()

class DocumentProcessor:    
    def __init__(self, llm):
        self.llm = llm
        self.logger = logger.bind(component="document_processor")
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n\n", "\n\n", "\n", ". ", " "]
        )
        
        # Metadata taggers
        try:
            self.doc_tagger = self._create_document_tagger()
            self.chunk_tagger = self._create_chunk_tagger()
            self.logger.info("metadata_taggers_initialized")
        except Exception as e:
            self.logger.warning("metadata_tagger_init_failed", error=str(e))
            self.doc_tagger = None
            self.chunk_tagger = None
    
    def _create_document_tagger(self):
        #document-level metadata tagger
        schema = {
            "properties": {
                "case_name": {"type": "string"},
                "document_type": {"type": "string"},
                "court": {"type": "string"}
            }
        }
        return create_metadata_tagger(schema, self.llm)
    
    def _create_chunk_tagger(self):
        #Create chunk-level metadata tagger
        schema = {
            "properties": {
                "section_type": {
                    "type": "string",
                    "enum": ["introduction", "facts", "argument", "conclusion", "other"]
                },
                "stance": {
                    "type": "string",
                    "enum": ["plaintiff", "defendant", "neutral", "unknown"]
                },
                "importance_score": {"type": "number", "minimum": 0, "maximum": 1},
                "legal_concepts": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["section_type", "stance", "importance_score"]
        }
        return create_metadata_tagger(schema, self.llm)
    
    def process_document(self, pdf_path: str) -> Tuple[List[Document], Dict, str]:
        self.logger.info("processing_document", path=pdf_path)
        
        # Load PDF
        loader = PDFPlumberLoader(pdf_path)
        pages = loader.load()
        
        # Generate document ID
        doc_id = hashlib.md5(pdf_path.encode()).hexdigest()[:12]
        
        # Extract document-level metadata from first 3 pages
        doc_metadata = {'total_pages': len(pages)}
        if self.doc_tagger:
            try:
                first_pages = pages[:3]
                enriched_pages = self.doc_tagger.transform_documents(first_pages)
                doc_metadata.update(enriched_pages[0].metadata)
                self.logger.info("document_metadata_extracted")
            except Exception as e:
                self.logger.warning("document_metadata_extraction_failed", error=str(e))
        
        # Chunk documents
        chunks = self.text_splitter.split_documents(pages)
        
        # Add document metadata to all chunks
        for idx, chunk in enumerate(chunks):
            chunk.metadata.update({
                'document_id': doc_id,
                'chunk_id': f"{doc_id}_chunk_{idx:04d}",
                'chunk_index': idx
            })
            chunk.metadata.update(doc_metadata)
        
        # Extract chunk-level metadata in batches
        enriched_chunks = self._extract_chunk_metadata_batched(chunks)
        
        self.logger.info("document_processed", chunks=len(enriched_chunks))
        
        return enriched_chunks, doc_metadata, doc_id
    
    def _extract_chunk_metadata_batched(self, chunks: List[Document]) -> List[Document]:
        if not self.chunk_tagger:
            self.logger.warning("chunk_tagger_not_available_skipping")
            return chunks
        
        batch_size = 50
        
        # Small documents: process all at once
        if len(chunks) <= batch_size:
            try:
                return self.chunk_tagger.transform_documents(chunks)
            except Exception as e:
                self.logger.warning("chunk_metadata_extraction_failed", error=str(e))
                return chunks
        
        # Large documents: batch processing
        enriched = []
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        for i in range(0, len(chunks), batch_size):
            batch_num = i // batch_size + 1
            batch = chunks[i:i+batch_size]
            
            self.logger.info(f"processing_batch", batch=batch_num, total=total_batches)
            
            try:
                enriched_batch = self.chunk_tagger.transform_documents(batch)
                enriched.extend(enriched_batch)
            except Exception as e:
                self.logger.warning(f"batch_metadata_failed", batch=batch_num, error=str(e))
                enriched.extend(batch) 
        
        return enriched
