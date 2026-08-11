import re
from typing import Tuple

# A basic heuristic list of emergency keywords.
# In a real medical app, this would be far more comprehensive.
EMERGENCY_KEYWORDS = [
    "severe difficulty breathing",
    "cannot breathe",
    "severe chest pain",
    "chest pressure",
    "loss of consciousness",
    "passed out",
    "fainted",
    "signs of stroke",
    "face drooping",
    "arm weakness",
    "slurred speech",
    "severe allergic reaction",
    "anaphylaxis",
    "uncontrolled bleeding",
    "bleeding heavily",
    "sudden severe neurological symptoms",
    "sudden severe headache",
    "suicide",
    "kill myself"
]

class SafetyChecker:
    @staticmethod
    def is_valid_query(query: str) -> Tuple[bool, str]:
        """
        Validates the user query to prevent empty, extremely short, or overly long inputs.
        Returns a tuple: (is_valid, error_message)
        """
        if not query or not query.strip():
            return False, "Query cannot be empty."
            
        cleaned_query = query.strip()
        if len(cleaned_query) < 5:
            return False, "Query is too short. Please provide more details about your symptoms."
            
        if len(cleaned_query) > 1000:
            return False, "Query is too long. Please summarize your symptoms in under 1000 characters."
            
        return True, ""

    @staticmethod
    def check_emergency(query: str) -> Tuple[bool, str]:
        """
        Checks if the query contains potential emergency keywords.
        Returns a tuple: (is_emergency, warning_message)
        """
        lower_query = query.lower()
        
        for keyword in EMERGENCY_KEYWORDS:
            if re.search(r'\b' + re.escape(keyword) + r'\b', lower_query):
                return True, (
                    f"WARNING: Your symptoms mention '{keyword}', which could indicate a medical emergency. "
                    "Please seek immediate professional medical care or call your local emergency number. "
                    "This system cannot diagnose emergencies."
                )
                
        return False, ""
