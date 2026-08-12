import streamlit as st
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pipeline.rag_pipeline import RAGPipeline
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Pre-warm: fire a background thread the instant this module is imported ──
# Streamlit imports app.py before it renders any UI, so by the time the user
# sees the first frame the heavy work (torch + SentenceTransformer + Qdrant)
# is already well underway or completely finished.
_prewarm_done   = threading.Event()
_prewarm_error  = None
_prewarm_result = {}   # shared dict filled by the background threads

def _prewarm_worker():
    global _prewarm_error
    try:
        import embeddings.embedding_service as emb_module
        from sentence_transformers import SentenceTransformer
        from vectorstore.qdrant_client import get_qdrant_client
        from llm.groq_client import get_groq_client

        def _load_model():
            model = SentenceTransformer(settings.EMBEDDING_MODEL)
            svc = emb_module.EmbeddingService.__new__(emb_module.EmbeddingService)
            svc.provider   = settings.EMBEDDING_PROVIDER
            svc.model_name = settings.EMBEDDING_MODEL
            svc.model      = model
            emb_module._embedding_service = svc
            return model

        def _load_qdrant():
            return get_qdrant_client()

        def _load_groq():
            return get_groq_client()

        with ThreadPoolExecutor(max_workers=3) as pool:
            futs = {
                pool.submit(_load_model):  "model",
                pool.submit(_load_qdrant): "qdrant",
                pool.submit(_load_groq):   "groq",
            }
            for fut in as_completed(futs):
                _prewarm_result[futs[fut]] = fut.result()

        _prewarm_result["pipeline"] = RAGPipeline()
    except Exception as exc:
        _prewarm_error = exc
    finally:
        _prewarm_done.set()

# Daemon=True so it never blocks process exit
_prewarm_thread = threading.Thread(target=_prewarm_worker, daemon=True, name="prewarm")
_prewarm_thread.start()

def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* Global typography & Outer background */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #F3F4F6;
        }

        /* Sidebar Styling - White background */
        [data-testid="stSidebar"],
        [data-testid="stSidebarContent"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #E5E7EB !important;
        }

        /* Sidebar nav links */
        .nav-item {
            display: block;
            color: #374151 !important;
            padding: 10px 15px;
            border-radius: 6px;
            margin-bottom: 4px;
            font-weight: 500;
            font-size: 1rem;
            cursor: pointer;
            transition: background-color 0.2s;
            text-decoration: none !important;
        }
        .nav-item:hover {
            background-color: #EFF6FF;
            color: #1D4ED8 !important;
            text-decoration: none !important;
        }
        .nav-item.active {
            background-color: #EFF6FF;
            color: #1D4ED8 !important;
            font-weight: 600;
        }
        .nav-item.disabled {
            color: #9CA3AF !important;
            cursor: default;
        }

        /* Main Content Background */
        .stApp {
            background-color: #F9FAFB;
        }

        /* Header styling */
        h1, h2, h3, h4 {
            color: #111827 !important;
            font-weight: 600 !important;
        }

        /* Metrics */
        div[data-testid="stMetricValue"] {
            font-size: 1.5rem !important;
            color: #0369A1 !important;
            font-weight: 600 !important;
        }
        div[data-testid="stMetricLabel"] {
            font-weight: 500 !important;
            color: #6B7280 !important;
        }

        /* Input Card */
        .clinical-card {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 8px;
            padding: 24px;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
            margin-bottom: 24px;
        }

        /* Text Area */
        .stTextArea textarea {
            border-radius: 6px !important;
            border: 1px solid #D1D5DB !important;
            padding: 12px !important;
            font-size: 1rem !important;
            background: #FFFFFF !important;
            color: #111827 !important;
        }
        .stTextArea textarea:focus {
            border-color: #0369A1 !important;
            box-shadow: 0 0 0 1px #0369A1 !important;
        }

        /* Primary Button */
        div.stButton > button:first-child {
            border-radius: 6px !important;
            font-weight: 500 !important;
            padding: 0.5rem 1.5rem !important;
            border: none !important;
            background-color: #0369A1 !important;
            color: white !important;
        }
        div.stButton > button:hover {
            background-color: #0284C7 !important;
        }
        
        /* Secondary Action Buttons (Suggestions) */
        div[data-testid="stVerticalBlock"] div.stButton > button[kind="secondary"] {
            background-color: #F3F4F6 !important;
            color: #374151 !important;
            border: 1px solid #D1D5DB !important;
            font-size: 0.9rem !important;
        }
        div[data-testid="stVerticalBlock"] div.stButton > button[kind="secondary"]:hover {
            background-color: #E5E7EB !important;
        }

        /* Data Table (Knowledge Sources) - Force dark text */
        [data-testid="stTable"] table {
            color: #111827 !important;
            background-color: #FFFFFF !important;
        }
        [data-testid="stTable"] th {
            background-color: #F3F4F6 !important;
            color: #374151 !important;
            font-weight: 600 !important;
            border-bottom: 2px solid #E5E7EB !important;
        }
        [data-testid="stTable"] td {
            color: #111827 !important;
            border-bottom: 1px solid #E5E7EB !important;
        }
        [data-testid="stTable"] tr:hover td {
            background-color: #F9FAFB !important;
        }

        /* Results Data Headers */
        .data-header {
            background-color: #F3F4F6;
            border-bottom: 1px solid #E5E7EB;
            padding: 10px 16px;
            font-weight: 600;
            color: #374151;
            border-radius: 8px 8px 0 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .data-content {
            padding: 16px;
            color: #111827;
            font-size: 0.95rem;
            line-height: 1.6;
            white-space: pre-line;
        }
        
        .alert-header {
            background-color: #FEF2F2 !important;
            border-bottom: 1px solid #FEE2E2 !important;
            color: #991B1B !important;
        }
        .alert-content {
            background-color: #FEF2F2 !important;
            color: #991B1B !important;
            border-top: none !important;
        }

        /* Hide Streamlit components */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_pipeline():
    """Wait for the pre-warm thread, then return the already-built pipeline."""
    _prewarm_done.wait()          # blocks only if still loading (usually near-instant)
    if _prewarm_error:
        raise _prewarm_error
    return _prewarm_result["pipeline"]

def render_data_panel(icon, title, content, is_alert=False):
    """Render an enterprise data panel"""
    header_class = "data-header alert-header" if is_alert else "data-header"
    content_class = "data-content alert-content" if is_alert else "data-content"
    
    st.markdown(f"""
        <div class="clinical-card" style="padding: 0;">
            <div class="{header_class}">
                <span>{icon}</span> {title}
            </div>
            <div class="{content_class}">
                {content.replace(chr(10), '<br>')}
            </div>
        </div>
    """, unsafe_allow_html=True)

def page_symptom_analyzer(pipeline):
    st.title("Clinical Decision Support")
    st.markdown("<p style='color: #6B7280; margin-bottom: 20px;'>Aura Health Diagnostic Intelligence System</p>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("System Status", "Online", delta="Stable", delta_color="normal")
    m2.metric("Active Models", "Llama-3.1-8B", delta="Groq Cloud", delta_color="off")
    m3.metric("Vector Store", "Qdrant Cloud", delta="Connected", delta_color="normal")
    m4.metric("Knowledge Records", "93 Chunks", delta="Indexed", delta_color="off")
    st.markdown("<hr style='border: 1px solid #E5E7EB; margin-bottom: 30px;'>", unsafe_allow_html=True)

    st.markdown("### Patient Presentation", unsafe_allow_html=True)
    user_query = st.text_area(
        "Enter Chief Complaint / Symptoms",
        value=st.session_state.get("symptoms", ""),
        height=120,
        placeholder="Document patient symptoms here..."
    )
    analyze_btn = st.button("Run Diagnostic Analysis", type="primary")

    if not user_query and not analyze_btn:
        st.markdown("<p style='color: #6B7280; font-size: 0.9rem; margin-top: 15px;'>Test Scenarios:</p>", unsafe_allow_html=True)
        sc1, sc2, sc3, sc4 = st.columns(4)
        if sc1.button("Fever & Sore Throat", use_container_width=True):
            st.session_state.symptoms = "Patient presents with a high fever and very severe sore throat for two days."
            st.rerun()
        if sc2.button("Migraine w/ Nausea", use_container_width=True):
            st.session_state.symptoms = "Patient reports an intense, throbbing migraine accompanied by nausea."
            st.rerun()
        if sc3.button("Chest & Reflux", use_container_width=True):
            st.session_state.symptoms = "Patient has severe heartburn after eating, and sour liquid keeps coming up."
            st.rerun()
        if sc4.button("Itchy Arm Rash", use_container_width=True):
            st.session_state.symptoms = "Patient developed a very itchy, red, cracked rash with some blisters on the arm."
            st.rerun()

    if analyze_btn:
        if not user_query:
            st.error("Please enter patient symptoms.")
        else:
            with st.spinner("Executing semantic search and LLM synthesis..."):
                success, result = pipeline.process_query(user_query)

            if not success:
                st.error(f"System Error: {result}", icon="🚨")
            else:
                # Save to history
                if "history" not in st.session_state:
                    st.session_state.history = []
                import datetime
                st.session_state.history.append({
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "query": user_query,
                    "conditions": result.possible_conditions.split("\n")[0][:80],
                    "result": result
                })

                st.markdown("### Diagnostic Report")
                if result.warning_signs and result.warning_signs.strip() and result.warning_signs.lower() != "none":
                    render_data_panel("🚨", "CRITICAL CLINICAL WARNINGS", result.warning_signs, is_alert=True)
                render_data_panel("📝", "Clinical Overview", result.symptoms_summary)
                c1, c2 = st.columns(2)
                with c1:
                    render_data_panel("🔍", "Possible Differentials", result.possible_conditions)
                with c2:
                    render_data_panel("🏥", "Recommended Action Plan", result.when_to_seek_care)
                render_data_panel("ℹ️", "General Medical Context", result.general_information)

                st.markdown("#### Retrieved Knowledge Sources")
                if result.sources:
                    st.table([{"Document Title": s.get('title', 'Unknown'), "Source Path": s.get('source', 'Unknown')} for s in result.sources])
                else:
                    st.info("No sources matched in vector database.")

                st.markdown(f"""
                    <div style="background-color: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 8px; padding: 16px 20px; margin-top: 16px; display: flex; align-items: flex-start; gap: 14px;">
                        <span style="font-size: 1.3rem;">ℹ️</span>
                        <span style="color: #1E40AF; font-size: 0.95rem; line-height: 1.6; font-weight: 500;">{result.disclaimer}</span>
                    </div>
                """, unsafe_allow_html=True)


def page_patient_history():
    st.title("Patient History")
    st.markdown("<p style='color: #6B7280; margin-bottom: 20px;'>Session-based log of all diagnostic queries and results</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 1px solid #E5E7EB; margin-bottom: 30px;'>", unsafe_allow_html=True)

    history = st.session_state.get("history", [])
    if not history:
        st.markdown("""
            <div class="clinical-card" style="text-align: center; padding: 60px; color: #9CA3AF;">
                <div style="font-size: 3rem; margin-bottom: 16px;">📋</div>
                <h3 style="color: #6B7280;">No History Yet</h3>
                <p>Run a diagnostic analysis in the Symptom Analyzer to see records here.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"**{len(history)} record(s) in this session**")
        st.markdown("")
        for i, record in enumerate(reversed(history)):
            with st.expander(f"🕐 {record['timestamp']}  —  {record['query'][:80]}...", expanded=(i == 0)):
                r1, r2 = st.columns(2)
                with r1:
                    render_data_panel("📝", "Clinical Overview", record["result"].symptoms_summary)
                    render_data_panel("🔍", "Possible Differentials", record["result"].possible_conditions)
                with r2:
                    render_data_panel("🏥", "Recommended Action Plan", record["result"].when_to_seek_care)
                    if record["result"].warning_signs and record["result"].warning_signs.strip():
                        render_data_panel("🚨", "Warning Signs", record["result"].warning_signs, is_alert=True)

        if st.button("🗑️ Clear Session History", type="primary"):
            st.session_state.history = []
            st.rerun()


def page_knowledge_base():
    import os, glob
    st.title("Knowledge Base")
    st.markdown("<p style='color: #6B7280; margin-bottom: 20px;'>Browse the medical literature indexed in the Qdrant vector store</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 1px solid #E5E7EB; margin-bottom: 30px;'>", unsafe_allow_html=True)

    kb_dir = os.path.join(os.path.dirname(__file__), "data", "raw")
    files = glob.glob(os.path.join(kb_dir, "*.md")) + glob.glob(os.path.join(kb_dir, "*.txt")) + glob.glob(os.path.join(kb_dir, "*.pdf"))

    # ── Centered Qdrant label (plain text, no background) ──
    st.markdown("""
        <div style="text-align: center; margin-bottom: 32px;">
            <div style="color: #111827; font-size: 1.4rem; font-weight: 700;">Qdrant Cloud — medical_knowledge</div>
        </div>
    """, unsafe_allow_html=True)

    if not files:
        st.markdown("""
            <div style="background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 10px; text-align: center; padding: 60px; color: #9CA3AF;">
                <div style="font-size: 3rem; margin-bottom: 16px;">📂</div>
                <h3 style="color: #6B7280;">No Documents Found</h3>
                <p style="color: #9CA3AF;">Add <code>.md</code>, <code>.txt</code>, or <code>.pdf</code> files to <code>data/raw/</code> to get started.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        for idx, fpath in enumerate(files):
            fname = os.path.basename(fpath)
            fsize = os.path.getsize(fpath)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()

                sections = [l.strip() for l in content.splitlines() if l.startswith("### ")]
                word_count = len(content.split())

                # Session state key for toggling this document
                toggle_key = f"kb_doc_{idx}"

                # Styled card as a clickable button
                if st.button(
                    f"📄  {fname}   ({fsize:,} bytes  ·  ~{word_count:,} words  ·  {len(sections)} sections)",
                    key=f"kb_btn_{idx}",
                    use_container_width=True
                ):
                    st.session_state[toggle_key] = not st.session_state.get(toggle_key, False)

                # Show content below the card when toggled open
                if st.session_state.get(toggle_key, False):
                    safe_content = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    st.markdown(f"""
                        <div style="
                            background: #000000;
                            color: #EF4444;
                            border-radius: 0 0 10px 10px;
                            padding: 24px;
                            margin-top: -16px;
                            margin-bottom: 24px;
                            font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
                            font-size: 0.85rem;
                            line-height: 1.7;
                            max-height: 500px;
                            overflow-y: auto;
                            white-space: pre-wrap;
                            word-wrap: break-word;
                        ">{safe_content}</div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Could not read file: {e}")


def page_system_settings():
    st.title("System Settings")
    st.markdown("<p style='color: #6B7280; margin-bottom: 20px;'>Current Aura Health system configuration and environment</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 1px solid #E5E7EB; margin-bottom: 30px;'>", unsafe_allow_html=True)

    s1, s2 = st.columns(2)
    with s1:
        st.markdown("""<div style="background:#FFFFFF; border:1px solid #E5E7EB; border-radius:8px; padding:24px; box-shadow:0 1px 3px rgba(0,0,0,0.1); margin-bottom:24px;">
            <h4 style="color:#111827; margin-top:0;">🤖 LLM Configuration</h4>
            <table style="width:100%; border-collapse:collapse; color:#111827;">
                <tr style="border-bottom:1px solid #E5E7EB;"><td style="padding:10px 0; font-weight:500; color:#6B7280;">Provider</td><td style="padding:10px 0; text-align:right;">Groq Cloud</td></tr>
                <tr style="border-bottom:1px solid #E5E7EB;"><td style="padding:10px 0; font-weight:500; color:#6B7280;">Model</td><td style="padding:10px 0; text-align:right;">""" + settings.GROQ_MODEL + """</td></tr>
                <tr><td style="padding:10px 0; font-weight:500; color:#6B7280;">API Key</td><td style="padding:10px 0; text-align:right;">""" + ("●●●●●●●●" if settings.GROQ_API_KEY else "Not Set") + """</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div style="background:#FFFFFF; border:1px solid #E5E7EB; border-radius:8px; padding:24px; box-shadow:0 1px 3px rgba(0,0,0,0.1); margin-bottom:24px;">
            <h4 style="color:#111827; margin-top:0;">🔢 Embedding Configuration</h4>
            <table style="width:100%; border-collapse:collapse; color:#111827;">
                <tr style="border-bottom:1px solid #E5E7EB;"><td style="padding:10px 0; font-weight:500; color:#6B7280;">Provider</td><td style="padding:10px 0; text-align:right;">""" + settings.EMBEDDING_PROVIDER + """</td></tr>
                <tr style="border-bottom:1px solid #E5E7EB;"><td style="padding:10px 0; font-weight:500; color:#6B7280;">Model</td><td style="padding:10px 0; text-align:right;">""" + settings.EMBEDDING_MODEL + """</td></tr>
                <tr><td style="padding:10px 0; font-weight:500; color:#6B7280;">Vector Dimensions</td><td style="padding:10px 0; text-align:right;">384</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)

    with s2:
        st.markdown("""<div style="background:#FFFFFF; border:1px solid #E5E7EB; border-radius:8px; padding:24px; box-shadow:0 1px 3px rgba(0,0,0,0.1); margin-bottom:24px;">
            <h4 style="color:#111827; margin-top:0;">🗄️ Vector Database Configuration</h4>
            <table style="width:100%; border-collapse:collapse; color:#111827;">
                <tr style="border-bottom:1px solid #E5E7EB;"><td style="padding:10px 0; font-weight:500; color:#6B7280;">Provider</td><td style="padding:10px 0; text-align:right;">Qdrant</td></tr>
                <tr style="border-bottom:1px solid #E5E7EB;"><td style="padding:10px 0; font-weight:500; color:#6B7280;">Connection</td><td style="padding:10px 0; text-align:right;">""" + ("Cloud" if settings.QDRANT_URL else "Local") + """</td></tr>
                <tr style="border-bottom:1px solid #E5E7EB;"><td style="padding:10px 0; font-weight:500; color:#6B7280;">Collection</td><td style="padding:10px 0; text-align:right;">medical_knowledge</td></tr>
                <tr style="border-bottom:1px solid #E5E7EB;"><td style="padding:10px 0; font-weight:500; color:#6B7280;">Top-K Retrieval</td><td style="padding:10px 0; text-align:right;">""" + str(settings.TOP_K) + """</td></tr>
                <tr><td style="padding:10px 0; font-weight:500; color:#6B7280;">API Key</td><td style="padding:10px 0; text-align:right;">""" + ("●●●●●●●●" if settings.QDRANT_API_KEY else "Not Set") + """</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div style="background:#FFFFFF; border:1px solid #E5E7EB; border-radius:8px; padding:24px; box-shadow:0 1px 3px rgba(0,0,0,0.1); margin-bottom:24px;">
            <h4 style="color:#111827; margin-top:0;">📦 Application Info</h4>
            <table style="width:100%; border-collapse:collapse; color:#111827;">
                <tr style="border-bottom:1px solid #E5E7EB;"><td style="padding:10px 0; font-weight:500; color:#6B7280;">Version</td><td style="padding:10px 0; text-align:right;">v2.4.1 Enterprise</td></tr>
                <tr style="border-bottom:1px solid #E5E7EB;"><td style="padding:10px 0; font-weight:500; color:#6B7280;">Framework</td><td style="padding:10px 0; text-align:right;">Streamlit</td></tr>
                <tr><td style="padding:10px 0; font-weight:500; color:#6B7280;">Pipeline</td><td style="padding:10px 0; text-align:right;">RAG (Retrieval-Augmented Generation)</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="Aura Health | Clinical System",
        page_icon="⚕️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    inject_custom_css()

    # Show animated progress only if the pre-warm thread is still running.
    # On fast machines / warm restarts, _prewarm_done is already set → instant load.
    if "pipeline_ready" not in st.session_state:
        if not _prewarm_done.is_set():
            bar = st.progress(0, text="⚕️ Starting Clinical Decision Support Engine...")
            steps = [
                (20, "🔢 Loading embedding model (PyTorch)..."),
                (45, "🗄️  Connecting to Qdrant vector store..."),
                (65, "🤖 Initialising Groq LLM client..."),
                (85, "🔗 Assembling RAG pipeline..."),
            ]
            for pct, msg in steps:
                if _prewarm_done.is_set():
                    break
                bar.progress(pct, text=msg)
            bar.progress(100, text="✅ Engine ready!")
            bar.empty()
        pipeline = get_pipeline()
        st.session_state.pipeline_ready = True
    else:
        pipeline = get_pipeline()

    # --- Session State Init ---
    if "page" not in st.session_state:
        st.session_state.page = "analyzer"
    if "symptoms" not in st.session_state:
        st.session_state.symptoms = ""
    if "history" not in st.session_state:
        st.session_state.history = []

    # --- SIDEBAR NAV ---
    with st.sidebar:
        st.markdown("<div style='padding: 10px 0 20px 0;'><h2 style='color: #111827; font-size: 1.4rem; font-weight: 700; margin-bottom: 24px;'>⚕️ Aura Health</h2></div>", unsafe_allow_html=True)

        pages = [
            ("analyzer",  "🏥", "Symptom Analyzer"),
            ("history",   "📋", "Patient History"),
            ("knowledge", "📚", "Knowledge Base"),
            ("settings",  "⚙️", "System Settings"),
        ]
        for key, icon, label in pages:
            is_active = st.session_state.page == key
            btn_style = "background-color: #EFF6FF; color: #1D4ED8; font-weight: 600;" if is_active else "color: #374151;"
            if st.sidebar.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True):
                st.session_state.page = key
                st.rerun()

        history_count = len(st.session_state.get("history", []))
        st.sidebar.markdown(f"<div style='margin-top: 40px; color: #9CA3AF; font-size: 0.8rem; padding: 0 15px;'>Session Records: {history_count}<br>v2.4.1 Enterprise Build</div>", unsafe_allow_html=True)

    # --- PAGE ROUTING ---
    page = st.session_state.page
    if page == "analyzer":
        page_symptom_analyzer(pipeline)
    elif page == "history":
        page_patient_history()
    elif page == "knowledge":
        page_knowledge_base()
    elif page == "settings":
        page_system_settings()


if __name__ == "__main__":
    main()
