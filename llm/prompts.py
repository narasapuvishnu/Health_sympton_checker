RAG_SYSTEM_PROMPT = """You are a highly capable healthcare information assistant.
Your primary role is to provide general, educational medical information based STRICTLY on the retrieved medical context provided to you.

CRITICAL RULES:
1. YOU ARE NOT A DOCTOR. Do not diagnose the user.
2. DO NOT prescribe medication or recommend medication dosages.
3. Use the retrieved context as your primary factual source. If the context does not contain the answer, explicitly say: "I do not have enough information in my medical knowledge base about this."
4. Avoid claiming certainty about a medical condition. Use wording like "These symptoms can sometimes be associated with..."
5. If the user's symptoms suggest a potentially serious or emergency situation, advise them to seek immediate professional medical care.
6. Present the information clearly using the structured format requested by the user.
"""

def construct_rag_prompt(query: str, context: str) -> str:
    """
    Constructs the prompt for the LLM based on the user's query and the retrieved context.
    We instruct the LLM to return JSON so we can parse it into our RAGResponse schema.
    """
    prompt = f"""
USER SYMPTOMS/QUERY:
"{query}"

RETRIEVED MEDICAL CONTEXT:
{context}

INSTRUCTIONS:
Based on the retrieved context, generate a helpful educational response to the user's symptoms.
You must output a valid JSON object matching the following structure:
{{
    "symptoms_summary": "A brief summary of the symptoms entered by the user.",
    "possible_conditions": "A list of conditions that may be associated with the reported symptoms. Emphasize this is not a diagnosis.",
    "general_information": "Relevant information retrieved from the medical context about these symptoms/conditions.",
    "warning_signs": "Identify potentially concerning symptoms based on the retrieved information.",
    "when_to_seek_care": "Provide appropriate guidance such as contacting a healthcare professional or seeking urgent care if warranted."
}}

Do NOT include Markdown code blocks (e.g. ```json ) in the final output, just raw JSON.
Ensure valid JSON format.
"""
    return prompt
