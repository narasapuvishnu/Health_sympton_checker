import streamlit as st
from pipeline.rag_pipeline import RAGPipeline
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# Initialize pipeline once per session
@st.cache_resource
def get_pipeline():
    logger.info("Initializing RAG Pipeline for Streamlit app...")
    return RAGPipeline()

def main():
    st.set_page_config(
        page_title="Healthcare Symptom Checker",
        page_icon="🩺",
        layout="centered"
    )

    # Sidebar
    with st.sidebar:
        st.title("About")
        st.info(
            "This project is an AI-powered medical information retrieval system "
            "using Retrieval-Augmented Generation (RAG)."
        )
        st.markdown(
            """
            **How it works:**
            1. Enter your symptoms.
            2. System searches the medical knowledge base.
            3. AI generates an educational summary based *only* on the retrieved context.
            """
        )
        
        st.markdown("### Settings")
        st.text(f"LLM: {settings.GROQ_MODEL}")
        st.text(f"Embeddings: {settings.EMBEDDING_MODEL}")
        
        st.markdown("---")
        st.warning(
            "**DISCLAIMER**: This application is for educational purposes only. "
            "It is not a substitute for professional medical advice, diagnosis, or treatment."
        )

    # Main area
    st.title("Healthcare Symptom Checker")
    st.subheader("AI-powered medical information retrieval using Retrieval-Augmented Generation")
    
    st.markdown("---")

    # Example queries
    st.markdown("**Example Symptoms:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Fever and sore throat for two days"):
            st.session_state.symptoms = "I have a fever and sore throat for two days."
    with col2:
        if st.button("Cough and difficulty breathing"):
            st.session_state.symptoms = "I have a cough and difficulty breathing."
    with col3:
        if st.button("Headache and nausea"):
            st.session_state.symptoms = "I have a headache and nausea."

    # Text area
    if "symptoms" not in st.session_state:
        st.session_state.symptoms = ""

    user_query = st.text_area(
        "Describe your symptoms...",
        value=st.session_state.symptoms,
        height=150,
        placeholder="E.g., I have been experiencing a mild headache and fatigue since yesterday..."
    )

    # Check button
    if st.button("Check Symptoms", type="primary"):
        if not user_query:
            st.error("Please describe your symptoms first.")
            return
            
        pipeline = get_pipeline()
        
        with st.spinner("Analyzing symptoms and searching medical knowledge base..."):
            success, result = pipeline.process_query(user_query)
            
        if not success:
            # Result contains the error or emergency message string
            if "WARNING" in result:
                st.error(result, icon="🚨")
            else:
                st.warning(result, icon="⚠️")
        else:
            # Result is a RAGResponse object
            st.markdown("---")
            st.header("Results")
            
            # Symptoms Summary
            st.subheader("📝 Understanding Your Symptoms")
            st.write(result.symptoms_summary)
            
            # Possible Conditions
            st.subheader("🔍 Possible Related Conditions")
            st.write(result.possible_conditions)
            
            # General Info
            st.subheader("ℹ️ General Information")
            st.write(result.general_information)
            
            # Warning Signs & Care
            col_warn, col_care = st.columns(2)
            with col_warn:
                st.subheader("⚠️ Warning Signs")
                st.write(result.warning_signs)
            with col_care:
                st.subheader("🏥 When to Seek Medical Care")
                st.write(result.when_to_seek_care)
                
            # Sources
            st.markdown("---")
            st.subheader("📚 Sources")
            if result.sources:
                for source in result.sources:
                    title = source.get("title", "Unknown Source")
                    file_name = source.get("source", "")
                    st.markdown(f"- **{title}** (File: `{file_name}`)")
            else:
                st.write("No specific sources retrieved.")
                
            # Final Disclaimer
            st.info(result.disclaimer, icon="🛑")

if __name__ == "__main__":
    main()
