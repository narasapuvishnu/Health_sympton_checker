import json
from typing import Dict, Any, Tuple
from models.schemas import RAGResponse
from safety.safety_checker import SafetyChecker
from retrieval.retriever import get_retriever
from llm.groq_client import get_groq_client
from llm.prompts import RAG_SYSTEM_PROMPT, construct_rag_prompt
from utils.logger import get_logger

logger = get_logger(__name__)

class RAGPipeline:
    def __init__(self):
        self.retriever = get_retriever()
        self.llm_client = get_groq_client()
        
    def process_query(self, query: str) -> Tuple[bool, Any]:
        """
        Process the user query end-to-end.
        Returns a tuple: (success, result)
        where result is either a RAGResponse object or an error/warning message string.
        """
        logger.info(f"Processing user query: '{query[:50]}...'")
        
        # 1. Validation
        is_valid, validation_msg = SafetyChecker.is_valid_query(query)
        if not is_valid:
            logger.warning(f"Query validation failed: {validation_msg}")
            return False, validation_msg
            
        # 2. Emergency Check
        is_emergency, emergency_msg = SafetyChecker.check_emergency(query)
        if is_emergency:
            logger.warning("Emergency keywords detected.")
            # We can still proceed but we must emphasize the emergency.
            # For this pipeline, we will return the emergency message immediately 
            # to prioritize safety, effectively bypassing standard retrieval.
            return False, emergency_msg
            
        # 3. Retrieve Context
        try:
            retrieved_docs = self.retriever.retrieve_context(query)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return False, "Failed to connect to the knowledge base. Please try again later."
            
        if not retrieved_docs:
            return False, "No relevant medical information found for these symptoms in the knowledge base."
            
        # 4. Construct Context String
        context_parts = []
        sources_list = []
        
        for idx, doc in enumerate(retrieved_docs):
            title = doc.metadata.get("title", "Unknown Source")
            source_file = doc.metadata.get("source", "Unknown File")
            context_parts.append(f"--- SOURCE {idx+1}: {title} ---\n{doc.text}\n")
            
            # Deduplicate sources
            source_entry = {"title": title, "source": source_file}
            if source_entry not in sources_list:
                sources_list.append(source_entry)
                
        context_str = "\n".join(context_parts)
        
        # 5. Generate LLM Response
        prompt = construct_rag_prompt(query, context_str)
        try:
            raw_response = self.llm_client.generate_response(prompt, RAG_SYSTEM_PROMPT)
            logger.info("Successfully received LLM response.")
            
            # 6. Parse JSON response
            # Sometimes LLMs wrap JSON in markdown blocks despite instructions
            raw_response = raw_response.strip()
            if raw_response.startswith("```json"):
                raw_response = raw_response[7:]
            if raw_response.endswith("```"):
                raw_response = raw_response[:-3]
                
            response_dict = json.loads(raw_response)
            
            # 7. Construct final typed response
            rag_response = RAGResponse(
                symptoms_summary=response_dict.get("symptoms_summary", ""),
                possible_conditions=response_dict.get("possible_conditions", ""),
                general_information=response_dict.get("general_information", ""),
                warning_signs=response_dict.get("warning_signs", ""),
                when_to_seek_care=response_dict.get("when_to_seek_care", ""),
                sources=sources_list
            )
            return True, rag_response
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {raw_response}. Error: {e}")
            return False, "Error processing the response format. Please try again."
        except Exception as e:
            logger.error(f"Error in LLM generation: {e}")
            return False, "An error occurred while generating the response. Please try again later."
