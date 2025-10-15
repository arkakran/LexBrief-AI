import structlog
from typing import List, Dict
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from models.schemas import LLMAnalysisOutput
from config import settings
import json

logger = structlog.get_logger()

class LLMServices:    
    def __init__(self, huggingface_api_key: str = None):
        self.logger = logger.bind(component="llm_services")
        
        # LLM
        self.llm = ChatGroq(
            model=settings.llm_model,
            temperature=0.1,
            groq_api_key=settings.groq_api_key,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        self.logger.info("llm_initialized")
    
    def generate_query_variations(self, user_query: str) -> List[str]:        
        variation_prompt = ChatPromptTemplate.from_template("""
            You are a legal search expert. Generate 4 different search queries to find ALL types of legal arguments.

            Original Query: {user_query}

            Generate 4 query variations that target:
            1. Specific statutory violations (18 USC 1461 1462 Comstock Act federal criminal law)
            2. Specific regulatory failures (21 CFR 314.500 Subpart H REMS requirements FDA approval)
            3. Constitutional and case law arguments (Dobbs 10th Amendment federalism precedents)
            4. Procedural and harm claims (resource diversion enforcement state laws)

            Return ONLY a JSON array of 4 query strings, no explanation.

            EXAMPLE:
            Original: "key legal arguments"
            Output: [
                "18 USC 1461 1462 Comstock Act federal criminal law mail distribution nonmailable abortion drugs statutory violations",
                "21 CFR 314.500 Subpart H REMS requirements FDA approval regulatory compliance pregnancy illness",
                "Dobbs Jackson 10th Amendment federalism constitutional claims state sovereignty case law precedents",
                "resource diversion state enforcement investigation prosecution harm violations state laws procedural"
            ]

            Now generate for:
            Original: {user_query}
            Output:""")
        
        try:
            response = self.llm.invoke(variation_prompt.format_messages(user_query=user_query))
            content = response.content.strip()
            
            # Clean markdown
            if content.startswith("```"):
                content = content.split("```")[1].replace("json", "").strip()
            
            variations = json.loads(content)
            
            # Ensure we have a list
            if isinstance(variations, list) and len(variations) >= 4:
                self.logger.info("query_variations_generated", count=len(variations))
                return variations[:4]
            else:
                # Fallback
                return [user_query]
        
        except Exception as e:
            self.logger.warning("query_variation_failed", error=str(e))
            # Fallback variations targeting ALL argument types
            return [
                "18 USC 1461 1462 Comstock Act federal criminal law statutory violations mail distribution",
                "21 CFR 314.500 Subpart H REMS FDA approval regulatory failures",
                "Dobbs 10th Amendment constitutional federalism case law",
                user_query
            ]
    
    def rerank(self, query: str, candidates: List[Dict], top_k: int = 25) -> List[Dict]:
        #Rerank using retrieval scores
        if not candidates:
            return []
        
        self.logger.info("reranking_with_retrieval_scores", count=len(candidates))
        
        # Sort by retrieval score
        sorted_candidates = sorted(
            candidates, 
            key=lambda x: x.get("score", 0.0), 
            reverse=True
        )[:top_k]
        
        # Set reranker_score = retrieval score
        for candidate in sorted_candidates:
            candidate["reranker_score"] = candidate.get("score", 0.0)
        
        self.logger.info("reranking_complete", results=len(sorted_candidates))
        return sorted_candidates
    
    def analyze(self, chunks: List[Dict], query: str) -> LLMAnalysisOutput:
        #Extract key points using LLM with enhanced legal analysis
        # Prepare context
        context = "\n\n---\n\n".join([
            f"[Page {c['metadata'].get('page_number', '?')}]\n{c['text']}"
            for c in chunks[:20]
        ])
        
        prompt = ChatPromptTemplate.from_template("""
            You are an expert legal analyst. Extract 12-15 DIVERSE legal arguments from DIFFERENT categories.

            Query: {query}

            Document Excerpts:
            {context}

            CRITICAL: You MUST extract arguments from ALL these categories:
            1. STATUTORY violations (18 U.S.C., Comstock Act)
            2. REGULATORY failures (21 C.F.R., Subpart H, REMS)
            3. CONSTITUTIONAL claims (10th Amendment, federalism)
            4. CASE LAW precedents (Dobbs, Supreme Court cases)
            5. PROCEDURAL/HARM claims (resource diversion, enforcement)

            SCAN THE TEXT FOR THESE SPECIFIC MARKERS:
            - "18 U.S.C. § 1461" or "§ 1462" or "Comstock" → STATUTORY ARGUMENT (MUST EXTRACT)
            - "21 C.F.R. § 314.500" or "Subpart H" or "REMS" → REGULATORY ARGUMENT (MUST EXTRACT)
            - "10th Amendment" or "U.S. Const. amend. X" → CONSTITUTIONAL ARGUMENT
            - "Dobbs" or "v. Jackson" → CASE LAW ARGUMENT
            - "resource" or "divert" or "investigate" → HARM/PROCEDURAL ARGUMENT

            PRIORITIZE DIVERSITY:
            - Extract at least 2-3 arguments from EACH category above
            - Do NOT extract 10 similar federalism arguments
            - BALANCE constitutional, statutory, and regulatory arguments

            CONCRETE EXAMPLES:

            STATUTORY (MUST HAVE 2-3 of these):
            {{
                "summary": "FDA violated Comstock Act (18 U.S.C. §§ 1461, 1462) by authorizing mail distribution of mifepristone, which federal criminal law classifies as nonmailable matter",
                "importance": "critical",
                "importance_score": 0.95,
                "stance": "plaintiff",
                "supporting_quote": "18 U.S.C. § 1461 declares articles designed for producing abortion to be nonmailable matter",
                "legal_concepts": ["Comstock Act", "18 U.S.C. § 1461", "18 U.S.C. § 1462", "federal criminal law"],
                "page_reference": 10
            }}

            REGULATORY (MUST HAVE 2-3 of these):
            {{
                "summary": "FDA improperly invoked Subpart H (21 C.F.R. § 314.500) by treating pregnancy as a life-threatening illness when pregnancy is a natural condition, not a disease",
                "importance": "critical",
                "importance_score": 0.90,
                "stance": "plaintiff",
                "supporting_quote": "Subpart H at 21 C.F.R. § 314.500 requires a serious or immediately life-threatening illness",
                "legal_concepts": ["Subpart H", "21 C.F.R. § 314.500", "FDA approval", "regulatory compliance"],
                "page_reference": 8
            }}

            CONSTITUTIONAL (HAVE 2-3 of these):
            {{
                "summary": "Post-Dobbs v. Jackson (142 S. Ct. 2228), abortion regulation is reserved to states under 10th Amendment; FDA's federal preemption violates state sovereignty",
                "importance": "critical",
                "importance_score": 0.92,
                "stance": "plaintiff",
                "supporting_quote": "Dobbs held that regulation of abortion is returned to the people and their elected representatives",
                "legal_concepts": ["Dobbs v. Jackson", "10th Amendment", "federalism", "state sovereignty"],
                "page_reference": 12
            }}

            PROCEDURAL/HARM (HAVE 1-2 of these):
            {{
                "summary": "States forced to divert scarce resources to investigate and prosecute violations caused by FDA's abandonment of REMS safety requirements",
                "importance": "high",
                "importance_score": 0.80,
                "stance": "plaintiff",
                "supporting_quote": "States must divert resources to investigate and prosecute violations of their laws",
                "legal_concepts": ["resource diversion", "state enforcement", "harm"],
                "page_reference": 15
            }}

            Return JSON:
            {{
                "extracted_points": [
                    // 2-3 STATUTORY arguments
                    // 2-3 REGULATORY arguments
                    // 2-3 CONSTITUTIONAL arguments
                    // 2-3 CASE LAW arguments
                    // 1-2 HARM/PROCEDURAL arguments
                ],
                "confidence": 0.0-1.0
            }}

            REQUIREMENTS:
            - Extract 12-15 points total
            - MUST include statutory (USC), regulatory (CFR), constitutional, and case law arguments
            - Each summary must cite specific law
            - Include ALL statute/case numbers in legal_concepts
            - Return ONLY valid JSON
            """)
        
        try:
            response = self.llm.invoke(prompt.format_messages(query=query, context=context))
            content = response.content.strip()
            
            # Clean markdown
            if content.startswith("```"):
                lines = content.split("```")
                if len(lines) >= 2:
                    content = lines[1].replace("json", "").strip()
            
            data = json.loads(content)
            validated = LLMAnalysisOutput(**data)
            
            self.logger.info("analysis_complete", points=len(validated.extracted_points))
            return validated
            
        except Exception as e:
            self.logger.error("analysis_failed", error=str(e))
            raise
    
    def refine_extraction(self, initial_points: List[Dict], query: str) -> List[Dict]:
        #Use LLM to refine extracted points for better specificity
        
        points_text = "\n\n".join([
            f"{i+1}. Summary: {p['summary']}\n   Quote: {p['supporting_quote']}\n   Legal Concepts: {', '.join(p.get('legal_concepts', []))}"
            for i, p in enumerate(initial_points)
        ])
        
        refinement_prompt = ChatPromptTemplate.from_template("""
                Review and refine these legal arguments to add exact statutory/regulatory citations.

                Original Query: {query}

                Extracted Arguments:
                {points_text}

                For each argument, enhance the summary by adding:
                - Exact USC section numbers (18 U.S.C. § 1461)
                - Exact CFR section numbers (21 C.F.R. § 314.500)
                - Full case citations (Dobbs v. Jackson, 142 S. Ct. 2228)
                - Specific legal concepts in the array

                Return JSON array with same structure but improved summaries and expanded legal_concepts.
                Return ONLY valid JSON, no markdown.
                """)
        
        try:
            response = self.llm.invoke(refinement_prompt.format_messages(
                query=query,
                points_text=points_text
            ))
            content = response.content.strip()
            
            if content.startswith("```"):
                lines = content.split("```")
                if len(lines) >= 2:
                    content = lines[1].replace("json", "").strip()
            
            refined = json.loads(content)
            
            for i, point in enumerate(initial_points):
                if i < len(refined):
                    if 'summary' in refined[i]:
                        point['summary'] = refined[i]['summary']
                    if 'legal_concepts' in refined[i]:
                        point['legal_concepts'] = refined[i]['legal_concepts']
            
            self.logger.info("extraction_refined", count=len(initial_points))
            return initial_points
            
        except Exception as e:
            self.logger.warning("refinement_failed_using_original", error=str(e))
            return initial_points
