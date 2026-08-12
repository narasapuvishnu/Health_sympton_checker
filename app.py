import streamlit as st
from pipeline.rag_pipeline import RAGPipeline
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

        /* Global typography & Outer background */
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }
        
        .stApp {
            background-color: #F8FAFC;
        }

        /* Hide Streamlit branding & default header */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}

        /* Full Width Hero */
        .hero-banner {
            width: 100%;
            background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
            padding: 4rem 2rem;
            text-align: center;
            border-radius: 0 0 40px 40px;
            margin-top: -60px;
            margin-bottom: 3rem;
            box-shadow: 0 10px 30px -10px rgba(37, 99, 235, 0.4);
            color: white;
        }

        .hero-title {
            font-size: 3.5rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
            line-height: 1.2;
            letter-spacing: -1px;
        }
        
        .hero-subtitle {
            font-size: 1.25rem;
            font-weight: 400;
            opacity: 0.9;
            max-width: 700px;
            margin: 0 auto;
        }

        /* Input Container */
        .input-card {
            background: #FFFFFF;
            padding: 30px;
            border-radius: 24px;
            box-shadow: 0 20px 40px -15px rgba(0,0,0,0.05);
            border: 1px solid #F1F5F9;
            margin-bottom: 2rem;
        }

        /* Text area styling */
        .stTextArea textarea {
            border-radius: 16px !important;
            border: 2px solid #E2E8F0 !important;
            padding: 20px !important;
            font-size: 1.1rem !important;
            background: #F8FAFC !important;
            color: #0F172A !important;
            transition: all 0.3s ease !important;
        }
        
        .stTextArea textarea:focus {
            border-color: #3B82F6 !important;
            background: #FFFFFF !important;
            box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.1) !important;
        }

        /* Primary Button (Full Width) */
        div.stButton > button:first-child {
            border-radius: 14px !important;
            font-weight: 600 !important;
            padding: 0.75rem 2rem !important;
            transition: all 0.3s ease !important;
            border: none !important;
            width: 100%;
            background: linear-gradient(135deg, #2563EB 0%, #3B82F6 100%) !important;
            color: white !important;
            box-shadow: 0 10px 20px -10px rgba(37, 99, 235, 0.5) !important;
        }
        
        div.stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 15px 25px -10px rgba(37, 99, 235, 0.6) !important;
        }
        
        /* Secondary Suggestion Buttons */
        div[data-testid="stVerticalBlock"] div.stButton > button[kind="secondary"] {
            background: #F8FAFC !important;
            color: #64748B !important;
            border: 1px solid #E2E8F0 !important;
            font-size: 1rem !important;
        }
        
        div[data-testid="stVerticalBlock"] div.stButton > button[kind="secondary"]:hover {
            border-color: #3B82F6 !important;
            color: #3B82F6 !important;
            background: #EFF6FF !important;
        }

        /* Dashboard Cards for Results */
        .dashboard-card {
            background: #FFFFFF;
            border-radius: 20px;
            border: 1px solid #F1F5F9;
            box-shadow: 0 10px 30px -10px rgba(0,0,0,0.05);
            padding: 24px;
            margin-bottom: 24px;
            height: 100%;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .dashboard-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 20px 40px -10px rgba(37,99,235,0.1);
            border-color: #E0E7FF;
        }

        .card-header {
            font-size: 1.2rem;
            font-weight: 700;
            color: #0F172A;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
        }

        .card-icon {
            background: #EFF6FF;
            width: 44px;
            height: 44px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 12px;
            margin-right: 14px;
            font-size: 1.25rem;
        }

        .card-content {
            color: #475569;
            line-height: 1.7;
            font-size: 1.05rem;
            border-top: 2px solid #F8FAFC;
            padding-top: 16px;
        }

        /* Warning styling */
        .warning-card {
            border-left: 6px solid #EF4444;
            background: #FEF2F2;
            border-color: #FEE2E2;
        }
        .warning-card .card-icon {
            background: #FEE2E2;
        }
        .warning-card .card-content {
            border-top-color: #FECACA;
            color: #991B1B;
        }
        .warning-card:hover {
            box-shadow: 0 20px 40px -10px rgba(239, 68, 68, 0.2);
            border-color: #FECACA;
        }

        /* Disclaimer */
        .disclaimer-box {
            background: white;
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            padding: 20px;
            margin-top: 30px;
            display: flex;
            gap: 16px;
            font-size: 1rem;
            color: #64748B;
            box-shadow: 0 4px 6px rgba(0,0,0,0.02);
            align-items: center;
        }
        </style>
    """, unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def _load_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)

@st.cache_resource(show_spinner="⚡ Initializing Core AI Systems...")
def get_pipeline():
    import embeddings.embedding_service as emb_module
    model = _load_embedding_model(settings.EMBEDDING_MODEL)
    service = emb_module.EmbeddingService.__new__(emb_module.EmbeddingService)
    service.provider = settings.EMBEDDING_PROVIDER
    service.model_name = settings.EMBEDDING_MODEL
    service.model = model
    emb_module._embedding_service = service
    logger.info("Injected cached embedding model into EmbeddingService.")
    return RAGPipeline()

def render_card(icon, title, content, is_warning=False):
    """Render a clean dashboard card"""
    card_class = "dashboard-card warning-card" if is_warning else "dashboard-card"
    st.markdown(f"""
        <div class="{card_class}">
            <div class="card-header">
                <div class="card-icon">{icon}</div>
                {title}
            </div>
            <div class="card-content">
                {content.replace(chr(10), '<br>')}
            </div>
        </div>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Aura Health | Desktop Dashboard",
        page_icon="✨",
        layout="wide", # CRITICAL: Full width layout
        initial_sidebar_state="collapsed"
    )
    
    inject_custom_css()
    pipeline = get_pipeline()

    # Full Width Hero Banner
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">✨ Aura Health</div>
            <div class="hero-subtitle">Instantly analyze your symptoms against millions of medical literature records. Get educational insights, warning signs, and care recommendations.</div>
        </div>
    """, unsafe_allow_html=True)

    # Initialize State
    if "symptoms" not in st.session_state:
        st.session_state.symptoms = ""
    
    # Input area, centered using columns to prevent it from being TOO wide on ultra-wide monitors
    _, col_input, _ = st.columns([1, 2, 1])
    
    with col_input:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        user_query = st.text_area(
            "What are you experiencing?",
            value=st.session_state.symptoms,
            height=140,
            placeholder="E.g., I have been experiencing a mild headache, nausea, and fatigue since yesterday..."
        )
        
        analyze_btn = st.button("🔍 Analyze Symptoms", type="primary")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # --- SUGGESTIONS ---
        if not user_query and not analyze_btn:
            st.markdown("<h4 style='text-align: center; color: #64748B; font-weight: 500; margin-bottom: 1.5rem;'>Try an example scenario:</h4>", unsafe_allow_html=True)
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                if st.button("🌡️ Fever & severe sore throat", use_container_width=True):
                    st.session_state.symptoms = "I have a high fever and a very severe sore throat for two days."
                    st.rerun()
            with sc2:
                if st.button("🫁 Persistent cough & wheezing", use_container_width=True):
                    st.session_state.symptoms = "I have a persistent cough and wheezing when breathing."
                    st.rerun()
            with sc3:
                if st.button("🤕 Migraine & nausea", use_container_width=True):
                    st.session_state.symptoms = "I am experiencing an intense migraine accompanied by nausea."
                    st.rerun()

    # --- RESULTS DASHBOARD (GRID LAYOUT) ---
    if analyze_btn:
        if not user_query:
            with col_input:
                st.error("Please describe your symptoms first.")
        else:
            # We don't constrain the results to col_input, we let them span wider
            _, col_results, _ = st.columns([1, 6, 1])
            with col_results:
                st.markdown("<br>", unsafe_allow_html=True)
                with st.spinner("🧠 Synthesizing medical knowledge..."):
                    success, result = pipeline.process_query(user_query)
                    
                if not success:
                    st.error(result, icon="🚨")
                else:
                    st.markdown("<h2 style='color: #0F172A; margin-bottom: 2rem; font-weight: 800;'>Analysis Results</h2>", unsafe_allow_html=True)
                    
                    # Top wide card for overview
                    render_card("📝", "Clinical Overview", result.symptoms_summary)
                    
                    # 2-column grid for specific details
                    grid_col1, grid_col2 = st.columns(2)
                    with grid_col1:
                        render_card("🔍", "Possible Conditions", result.possible_conditions)
                    with grid_col2:
                        render_card("🏥", "Recommended Care", result.when_to_seek_care)
                    
                    # Full width for warnings
                    if result.warning_signs and result.warning_signs.strip() and result.warning_signs.lower() != "none":
                        render_card("⚠️", "Critical Warning Signs", result.warning_signs, is_warning=True)
                        
                    # General Info
                    render_card("ℹ️", "General Information", result.general_information)
                        
                    # Sources Expander
                    st.markdown("### 📚 Medical Literature Sources")
                    st.markdown("<p style='color: #64748B; margin-bottom: 1rem;'>Aura generated this analysis based on the following verified documents:</p>", unsafe_allow_html=True)
                    if result.sources:
                        for idx, source in enumerate(result.sources):
                            with st.expander(f"Source {idx+1}: {source.get('title', 'Unknown')}"):
                                st.code(source.get('source', 'Unknown path'))
                    else:
                        st.info("No specific sources retrieved.")
                                
                    st.markdown(f"""
                        <div class="disclaimer-box">
                            <div style="font-size: 1.5rem;">🛑</div>
                            <div>
                                {result.disclaimer}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
    # Footer
    st.markdown("""
        <div style="text-align: center; margin-top: 5rem; padding: 2rem; color: #94A3B8; font-size: 1rem;">
            Aura Health Engine • Powered by RAG & Vector Embeddings
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
