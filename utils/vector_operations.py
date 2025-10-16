import structlog
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from langchain.schema import Document
from config import settings
import uuid
import os
import time
import requests
import numpy as np
import hashlib

logger = structlog.get_logger()


class HuggingFaceEmbeddings:
    
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.model_name = model_name
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not self.api_key:
            raise ValueError("HUGGINGFACE_API_KEY required")
        
        self.api_url = (
            f"https://router.huggingface.co/hf-inference/models/"
            f"{model_name}/pipeline/feature-extraction"
        )
        
        self.fallback_url = self.api_url
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    def _request_embeddings(self, batch, url, normalize=True, truncate=True, wait_for_model=True, timeout=60):
        payload = {
            "inputs": batch,
            "parameters": {"normalize": normalize, "truncate": truncate},
            "options": {"wait_for_model": wait_for_model},
        }
        return requests.post(url, headers=self.headers, json=payload, timeout=timeout)
    
    def encode(self, texts, batch_size=8, show_progress=False, normalize=True):
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        for batch_idx, batch in enumerate(batches):
            if show_progress:
                print(f"Embedding batch {batch_idx + 1}/{len(batches)}")
            
            success = False
            for attempt in range(3):
                try:
                    response = self._request_embeddings(batch, self.api_url, normalize=normalize)
                    if response.status_code != 200:
                        response = self._request_embeddings(batch, self.fallback_url, normalize=normalize, wait_for_model=True)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if isinstance(data, list) and data and isinstance(data[0], float):
                            data = [data]
                        all_embeddings.extend(data)
                        success = True
                        break
                    else:
                        print(f"API Error {response.status_code}, retrying in 5s...")
                        time.sleep(5)
                except Exception as e:
                    print(f"Request error: {e}, retrying in 5s...")
                    time.sleep(5)
            
            if not success:
                print("Failed to get embeddings, using dummy vectors")
                dim = 768
                dummy_embedding = [0.0] * dim
                all_embeddings.extend([dummy_embedding] * len(batch))
        
        embeddings = np.array(all_embeddings, dtype=np.float32)
        
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms
        
        return embeddings


class VectorOperations:
    
    def __init__(self):
        self.logger = logger.bind(component="vector_operations")
        
        # HuggingFace embeddings
        self.embedding_model = HuggingFaceEmbeddings(settings.embedding_model)
        
        # Qdrant client with increased timeout
        self.qdrant = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=120
        )
        
        # Current collection name (set per document)
        self.collection = None
        
        self.logger.info("vector_operations_initialized")
    
    def get_collection_name(self, doc_id: str) -> str:
        # Use first 8 chars of doc_id for readability
        return f"legal_doc_{doc_id[:8]}"
    
    def init_collection_for_document(self, doc_id: str):
        """Create a fresh Qdrant collection for a specific document."""
        self.collection = self.get_collection_name(doc_id)
        
        try:
            # Check if collection exists
            collections = self.qdrant.get_collections().collections
            collection_exists = any(c.name == self.collection for c in collections)
            
            if collection_exists:
                self.logger.info("deleting_existing_collection", collection=self.collection)
                self.qdrant.delete_collection(collection_name=self.collection)
                time.sleep(0.5)
            
            # Create new collection
            self.qdrant.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
            # Allow Qdrant a moment to register the collection
            time.sleep(0.5)
            self.logger.info("collection_created", collection=self.collection)
            
        except Exception as e:
            self.logger.error("collection_init_failed", error=str(e))
            raise
    
    def cleanup_old_collections(self, keep_latest: int = 5):
        """Keep only a few recent collections; never delete the active one."""
        try:
            collections = self.qdrant.get_collections().collections
            
            # Filter only legal document collections
            legal_collections = [
                c for c in collections 
                if c.name.startswith("legal_doc_")
            ]
            
            if len(legal_collections) <= keep_latest:
                self.logger.info("no_cleanup_needed", total=len(legal_collections))
                return
            
            # Sort alphabetically (approx chronological if doc_id has timestamp)
            legal_collections.sort(key=lambda c: c.name)
            to_delete = legal_collections[:-keep_latest]
            
            # Ensure current (just-created) collection is not deleted
            if self.collection:
                to_delete = [c for c in to_delete if c.name != self.collection]
            
            for collection in to_delete:
                self.logger.info("deleting_old_collection", name=collection.name)
                self.qdrant.delete_collection(collection_name=collection.name)
            
            self.logger.info("cleanup_complete", deleted=len(to_delete), kept=keep_latest)
            
        except Exception as e:
            self.logger.warning("cleanup_failed", error=str(e))
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using HuggingFace model."""
        self.logger.info("embedding_texts", count=len(texts))
        embeddings = self.embedding_model.encode(texts, batch_size=8, show_progress=True)
        embeddings_list = embeddings.tolist()
        self.logger.info("embeddings_complete", total=len(embeddings_list))
        return embeddings_list
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""
        return self.embed_texts([query])[0]
    
    def store_chunks(self, chunks: List[Document], embeddings: List[List[float]]):
        """Store document chunks and their embeddings in Qdrant."""
        if not self.collection:
            raise ValueError("Collection not initialized. Call init_collection_for_document first.")
        
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={"text": chunk.page_content, **chunk.metadata}
            ))
        
        batch_size = 50
        total_batches = (len(points) + batch_size - 1) // batch_size
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            try:
                self.qdrant.upsert(
                    collection_name=self.collection,
                    points=batch,
                    wait=False
                )
                self.logger.info("batch_stored", batch=batch_num, total=total_batches, count=len(batch))
            except Exception as e:
                self.logger.error("batch_storage_failed", batch=batch_num, error=str(e))
        
        self.logger.info("chunks_stored", total=len(points), collection=self.collection)
    
    def retrieve(self, query: str, top_k: int = 60, filters: Dict = None) -> List[Dict]:
        """Retrieve top relevant chunks for a query."""
        if not self.collection:
            raise ValueError("Collection not initialized")
        
        query_vector = self.embed_query(query)
        
        query_filter = None
        if filters:
            conditions = [FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()]
            query_filter = Filter(must=conditions)
        
        results = self.qdrant.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter
        )
        
        formatted = []
        for r in results:
            formatted.append({
                "text": r.payload.get("text", ""),
                "score": r.score,
                "metadata": {k: v for k, v in r.payload.items() if k != "text"}
            })
        
        self.logger.info("retrieval_complete", results=len(formatted), collection=self.collection)
        return formatted
    
    def multi_query_retrieve(self, queries: List[str], top_k: int = 60) -> List[Dict]:
        """Perform retrieval using multiple query variations."""
        self.logger.info("multi_query_retrieval", num_queries=len(queries))
        
        all_results = {}
        
        for idx, query in enumerate(queries):
            self.logger.info(f"query_{idx+1}", query=query[:100])
            results = self.retrieve(query, top_k=top_k // len(queries) + 10)
            
            for r in results:
                text_key = r["text"][:200]
                if text_key not in all_results or r["score"] > all_results[text_key]["score"]:
                    all_results[text_key] = r
        
        combined = list(all_results.values())
        combined.sort(key=lambda x: x["score"], reverse=True)
        
        self.logger.info("multi_query_complete", total_unique=len(combined))
        return combined[:top_k]

