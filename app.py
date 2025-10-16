import structlog
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import time
from datetime import datetime

from config import settings
from utils import DocumentProcessor, VectorOperations, LLMServices, PostProcessor
from models import AnalysisResult
from langchain_groq import ChatGroq

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev-secret-key-change-in-production'
app.config['UPLOAD_FOLDER'] = settings.upload_folder
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

services = {}

def init_services():
    """Initialize services once"""
    if not services:
        logger.info("initializing_services")
        
        llm = ChatGroq(
            model=settings.llm_model,
            temperature=0,
            groq_api_key=settings.groq_api_key
        )
        
        services['doc_processor'] = DocumentProcessor(llm)
        services['vector_ops'] = VectorOperations()
        services['llm'] = LLMServices()
        services['post_processor'] = PostProcessor(services['vector_ops'])
        
        logger.info("services_initialized")

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Main analysis endpoint"""
    start_time = time.time()
    
    try:
        init_services()
        
        if 'file' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(url_for('index'))
        
        file = request.files['file']
        if not file.filename or not file.filename.endswith('.pdf'):
            flash('Only PDF files are allowed', 'error')
            return redirect(url_for('index'))
        
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{filename}")
        file.save(filepath)
        
        logger.info("file_uploaded", filename=filename)
        
        query = request.form.get('query', 'key legal arguments and important points')
        logger.info("user_query", query=query)
        
        # STEP 1-3: Process document (generates doc_id)
        logger.info("step_1_3_processing_document")
        enriched_chunks, doc_metadata, doc_id = services['doc_processor'].process_document(filepath)
        
        # STEP 3.5: Initialize document-specific collection
        logger.info("step_3_5_init_document_collection", doc_id=doc_id)
        services['vector_ops'].init_collection_for_document(doc_id)
        
        # STEP 3.6: Cleanup old collections (keep only 5 most recent)
        logger.info("step_3_6_cleanup_old_collections")
        services['vector_ops'].cleanup_old_collections(keep_latest=5)
        
        # STEP 4: Embed chunks
        logger.info("step_4_embedding_chunks")
        texts = [c.page_content for c in enriched_chunks]
        embeddings = services['vector_ops'].embed_texts(texts)
        
        # STEP 5: Store in document-specific collection
        logger.info("step_5_storing_in_vector_db")
        services['vector_ops'].store_chunks(enriched_chunks, embeddings)
        
        # STEP 6: Multi-query retrieval
        logger.info("step_6_generating_query_variations")
        query_variations = services['llm'].generate_query_variations(query)
        logger.info("query_variations", count=len(query_variations))
        
        logger.info("step_6_multi_query_retrieval")
        retrieved = services['vector_ops'].multi_query_retrieve(
            queries=query_variations,
            top_k=settings.top_k_retrieval
        )
        
        if not retrieved:
            flash('No relevant content found', 'warning')
            return redirect(url_for('index'))
        
        # STEP 7: Rerank
        logger.info("step_7_reranking")
        reranked = services['llm'].rerank(query, retrieved, top_k=settings.top_k_reranked)
        
        # STEP 8: LLM Analysis
        logger.info("step_8_llm_analysis")
        llm_output = services['llm'].analyze(reranked, query)
        
        # STEP 9: Quote mapping
        logger.info("step_9_quote_mapping")
        enriched_points = []
        for point in llm_output.extracted_points:
            quote_info = services['post_processor'].map_quote(point.supporting_quote, reranked)
            
            enriched_points.append({
                "summary": point.summary,
                "importance": point.importance,
                "importance_score": point.importance_score,
                "stance": point.stance,
                "supporting_quote": point.supporting_quote,
                "legal_concepts": point.legal_concepts,
                "page_number": quote_info.get("page_number"),
                "section_type": quote_info.get("section_type"),
                "retrieval_score": 0.0,
                "reranker_score": 0.0
            })
        
        # STEP 9B: Refinement
        logger.info("step_9b_llm_refinement")
        enriched_points = services['llm'].refine_extraction(enriched_points, query)
        
        # STEP 10: Diversity ranking
        logger.info("step_10_diversity_ranking")
        final_points = services['post_processor'].rank_with_diversity(
            enriched_points,
            top_k=settings.final_output_count
        )
        
        processing_time = round(time.time() - start_time, 2)
        
        result = AnalysisResult(
            case_name=doc_metadata.get('case_name'),
            document_type=doc_metadata.get('document_type'),
            key_points=final_points,
            total_pages=doc_metadata.get('total_pages', 0),
            total_chunks=len(enriched_chunks),
            processing_time=processing_time
        )
        
        logger.info("analysis_complete", time=processing_time, points=len(final_points))
        
        # Optional: Delete uploaded file after processing
        try:
            os.unlink(filepath)
            logger.info("temp_file_deleted", filepath=filepath)
        except:
            pass
        
        return render_template('results.html', result=result, query=query)
        
    except Exception as e:
        logger.error("analysis_failed", error=str(e), exc_info=True)
        flash(f'Analysis failed: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/health')
def health():
    """Health check"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000, debug=True)
