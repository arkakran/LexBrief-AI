import structlog
from typing import List, Dict
from rapidfuzz import fuzz
from models.schemas import FinalKeyPoint
from config import settings
import numpy as np

logger = structlog.get_logger()


class PostProcessor:
    #Final processing: quote mapping + category-aware diversity ranking
    
    def __init__(self, vector_ops):
        self.vector_ops = vector_ops  # Has access to HF API embeddings
        self.logger = logger.bind(component="post_processor")
    
    def map_quote(self, quote: str, chunks: List[Dict]) -> Dict:
        best_match = None
        best_score = 0.0
        
        for chunk in chunks:
            score = fuzz.partial_ratio(quote, chunk["text"])
            if score > best_score:
                best_score = score
                best_match = chunk
        
        if best_match and best_score >= 80:
            return {
                "page_number": best_match["metadata"].get("page_number"),
                "section_type": best_match["metadata"].get("section_type"),
                "matched": True
            }
        
        return {"page_number": None, "section_type": None, "matched": False}
    
    def categorize_argument(self, point: Dict) -> str:
        #Categorize argument: 'statutory', 'constitutional', 'regulatory', 'case_law', 'procedural', 'other'
        legal_concepts = [c.lower() for c in point.get('legal_concepts', [])]
        summary = point.get('summary', '').lower()
        
        # Check for statutory (USC sections)
        if any('u.s.c.' in c or 'usc' in c or 'comstock' in c for c in legal_concepts):
            return 'statutory'
        if any('18 u.s.c.' in s or 'comstock act' in s for s in [summary]):
            return 'statutory'
        
        # Check for regulatory (CFR sections)
        if any('c.f.r.' in c or 'cfr' in c or 'subpart' in c or 'rems' in c for c in legal_concepts):
            return 'regulatory'
        if any('21 c.f.r.' in s or 'subpart h' in s or 'rems' in s for s in [summary]):
            return 'regulatory'
        
        # Check for constitutional
        if any('amendment' in c or 'constitution' in c or 'const.' in c for c in legal_concepts):
            return 'constitutional'
        if 'amendment' in summary or 'constitutional' in summary:
            return 'constitutional'
        
        # Check for case law (has v. or case names)
        if any('v.' in c or 'dobbs' in c or 'jackson' in c for c in legal_concepts):
            return 'case_law'
        if ' v. ' in summary:
            return 'case_law'
        
        # Check for procedural
        if any('procedural' in c or 'approval process' in c or 'administrative' in c for c in legal_concepts):
            return 'procedural'
        
        return 'other'
    
    def rank_with_diversity(self, points: List[Dict], top_k: int = 10) -> List[FinalKeyPoint]:
        if len(points) <= top_k:
            return self._create_final_points(points)
        
        self.logger.info("starting_category_aware_ranking", total_points=len(points))
        
        # Step 1: Categorize all points
        for p in points:
            p['category'] = self.categorize_argument(p)
        
        # Log category distribution
        categories = {}
        for p in points:
            cat = p['category']
            categories[cat] = categories.get(cat, 0) + 1
        self.logger.info("category_distribution", categories=categories)
        
        # Step 2: Embed summaries for similarity comparison
        summaries = [p["summary"] for p in points]
        embeddings = self.vector_ops.embed_texts(summaries)
        embeddings = np.array(embeddings)
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        # Step 3: Deduplicate within each category
        deduplicated_by_category = {}
        
        for category in ['statutory', 'regulatory', 'constitutional', 'case_law', 'procedural', 'other']:
            category_points = [p for p in points if p['category'] == category]
            if not category_points:
                continue
            
            category_indices = [i for i, p in enumerate(points) if p['category'] == category]
            
            keep_indices = []
            for idx in category_indices:
                is_duplicate = False
                for kept_idx in keep_indices:
                    similarity = np.dot(embeddings[idx], embeddings[kept_idx])
                    if similarity > 0.75:  # Similar within same category
                        # Keep the one with higher importance
                        if points[idx]["importance_score"] <= points[kept_idx]["importance_score"]:
                            is_duplicate = True
                            break
                        else:
                            # Replace with higher scoring one
                            keep_indices.remove(kept_idx)
                            break
                
                if not is_duplicate:
                    keep_indices.append(idx)
            
            deduplicated_by_category[category] = [points[i] for i in keep_indices]
            self.logger.info(f"deduplicated_{category}", original=len(category_points), kept=len(keep_indices))
        
        # Step 4: Category-aware selection
        # Prioritize getting at least one from each major category
        selected = []
        
        # First pass: Get top from each major category
        priority_categories = ['statutory', 'regulatory', 'constitutional', 'case_law']
        for category in priority_categories:
            if category in deduplicated_by_category and deduplicated_by_category[category]:
                # Get top point from this category
                category_points = deduplicated_by_category[category]
                category_points.sort(key=lambda x: x['importance_score'], reverse=True)
                if category_points and len(selected) < top_k:
                    selected.append(category_points[0])
                    self.logger.info(f"selected_top_{category}", summary=category_points[0]['summary'][:100])
        
        # Second pass: Fill remaining slots with highest scoring points
        remaining_slots = top_k - len(selected)
        if remaining_slots > 0:
            # Get all remaining points
            selected_indices = {id(p) for p in selected}
            remaining = []
            for cat_points in deduplicated_by_category.values():
                for p in cat_points:
                    if id(p) not in selected_indices:
                        remaining.append(p)
            
            # Sort by combined score
            for p in remaining:
                p["combined_score"] = (
                    0.5 * p["importance_score"] +
                    0.3 * p.get("reranker_score", 0) +
                    0.2 * p.get("retrieval_score", 0)
                )
            
            remaining.sort(key=lambda x: x["combined_score"], reverse=True)
            selected.extend(remaining[:remaining_slots])
        
        # Sort final selection by importance
        selected.sort(key=lambda x: x["importance_score"], reverse=True)
        
        self.logger.info("category_aware_ranking_complete", 
                        final_count=len(selected),
                        categories={cat: sum(1 for p in selected if p['category'] == cat) 
                                   for cat in priority_categories})
        
        return self._create_final_points(selected[:top_k])
    
    def _create_final_points(self, points: List[Dict]) -> List[FinalKeyPoint]:
        return [
            FinalKeyPoint(
                summary=p["summary"],
                importance=p["importance"],
                importance_score=p["importance_score"],
                stance=p["stance"],
                supporting_quote=p["supporting_quote"],
                legal_concepts=p.get("legal_concepts", []),
                page_number=p.get("page_number"),
                section_type=p.get("section_type"),
                retrieval_score=p.get("retrieval_score", 0),
                reranker_score=p.get("reranker_score", 0),
                final_rank=idx + 1
            )
            for idx, p in enumerate(points)
        ]
