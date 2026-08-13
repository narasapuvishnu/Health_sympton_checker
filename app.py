import streamlit as st
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pipeline.rag_pipeline import RAGPipeline
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

        /* Global typography & Outer background */
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
            background-color: #f0f4f8;
        }

        /* Sidebar Styling - Light Glassmorphism */
        section[data-testid="stSidebar"] {
            background-color: #ffffff !important;
            border-right: 1px solid #e2e8f0 !important;
        }
        
        /* Force all sidebar text to be visible (dark) */
        section[data-testid="stSidebar"] * {
            color: #0f172a !important;
        }

        /* Ensure table text is dark for visibility */
        [data-testid="stTable"] table {
            color: #0f172a !important;
            background-color: #ffffff !important;
        }
        [data-testid="stTable"] th {
            color: #0f172a !important;
            background-color: #f3f4f6 !important;
        }
        [data-testid="stTable"] td {
            color: #0f172a !important;
            background-color: #ffffff !important;
        }

        /* Sidebar nav links - using Streamlit buttons styled via CSS */
        section[data-testid="stSidebar"] button[kind="secondary"] {
            background-color: #f8fafc !important;
            color: #0f172a !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 8px !important;
            padding: 12px !important;
            text-align: left !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
            margin-bottom: 8px !important;
        }
        section[data-testid="stSidebar"] button[kind="secondary"]:hover {
            background-color: #e2e8f0 !important;
            transform: translateX(4px);
        }
        section[data-testid="stSidebar"] button[kind="secondary"] p {
            font-size: 1.05rem !important;
            color: #0f172a !important;
        }

        /* Main Content Background */
        .stApp {
            background: #f8fafc;
            background-image: radial-gradient(at 0% 0%, hsla(217,100%,97%,1) 0, transparent 50%), radial-gradient(at 100% 0%, hsla(210,100%,95%,1) 0, transparent 50%);
        }

        /* Header styling */
        h1, h2, h3, h4 {
            color: #0f172a !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em;
        }

        /* Metrics */
        div[data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            color: #2563eb !important;
            font-weight: 700 !important;
        }
        div[data-testid="stMetricLabel"] {
            font-weight: 500 !important;
            color: #64748b !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-size: 0.8rem !important;
        }

        /* Input Card - Glassmorphism */
        .clinical-card {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.5);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.01);
            margin-bottom: 24px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        /* Text Area */
        .stTextArea textarea {
            border-radius: 12px !important;
            border: 2px solid #e2e8f0 !important;
            padding: 16px !important;
            font-size: 1.05rem !important;
            background: #ffffff !important;
            color: #0f172a !important;
            transition: all 0.3s ease;
            box-shadow: inset 0 2px 4px 0 rgba(0,0,0,0.02);
        }
        .stTextArea textarea:focus {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.1) !important;
        }

        /* Primary Button (Analyze) */
        button[kind="primary"] {
            border-radius: 10px !important;
            font-weight: 600 !important;
            padding: 0.6rem 1.8rem !important;
            border: none !important;
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
            color: white !important;
            box-shadow: 0 4px 14px 0 rgba(37, 99, 235, 0.39);
            transition: all 0.3s ease;
        }
        button[kind="primary"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4);
        }
        
        /* Secondary Action Buttons (Main area) */
        div[data-testid="stVerticalBlock"] div.stButton > button[kind="secondary"] {
            background-color: #ffffff !important;
            color: #475569 !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 10px !important;
            font-size: 0.95rem !important;
            transition: all 0.2s ease;
        }
        div[data-testid="stVerticalBlock"] div.stButton > button[kind="secondary"]:hover {
            background-color: #f8fafc !important;
            border-color: #94a3b8 !important;
            color: #0f172a !important;
        }

        /* Table Styling */
        [data-testid="stTable"] table {
            color: #0f172a !important;
            background-color: #ffffff !important;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            width: 100%;
        }
        [data-testid="stTable"] th {
            background-color: #f1f5f9 !important;
            color: #334155 !important;
            font-weight: 700 !important;
            border-bottom: 2px solid #cbd5e1 !important;
        }
        [data-testid="stTable"] td {
            color: #1e293b !important;
            border-bottom: 1px solid #f1f5f9 !important;
            background-color: #ffffff !important;
        }

        /* Results Data Headers */
        .data-header {
            background: linear-gradient(90deg, #f8fafc 0%, #f1f5f9 100%);
            border-bottom: 1px solid #e2e8f0;
            padding: 14px 20px;
            font-weight: 700;
            color: #0f172a;
            border-radius: 16px 16px 0 0;
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 1.1rem;
        }
        .data-content {
            padding: 20px;
            color: #334155;
            font-size: 1rem;
            line-height: 1.7;
            white-space: pre-line;
        }
        
        .alert-header {
            background: linear-gradient(90deg, #fef2f2 0%, #fee2e2 100%) !important;
            border-bottom: 1px solid #fecaca !important;
            color: #b91c1c !important;
        }
        .alert-content {
            background-color: #fef2f2 !important;
            color: #991b1b !important;
            border-top: none !important;
        }

        /* Hide Streamlit components */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}

        /* Spinner Text Color */
        div[data-testid="stSpinner"] > div > div {
            color: #2563eb !important;
            font-weight: 600 !important;
        }
        </style>
    """, unsafe_allow_html=True)

@st.cache_resource(show_spinner="Starting Clinical Decision Support Engine...")
def get_pipeline():
    """Initializes and returns the RAG pipeline."""
    return RAGPipeline()

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

def page_symptom_analyzer():
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
        key="symptoms",
        height=120,
        placeholder="Document patient symptoms here..."
    )
    analyze_btn = st.button("Run Diagnostic Analysis", type="primary")

    if not user_query and not analyze_btn:
        st.markdown("<p style='color: #6B7280; font-size: 0.9rem; margin-top: 15px;'>Test Scenarios:</p>", unsafe_allow_html=True)
        
        def set_symptoms(text):
            st.session_state.symptoms = text

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.button("Fever & Sore Throat", use_container_width=True, on_click=set_symptoms, args=("Patient presents with a high fever and very severe sore throat for two days.",))
        sc2.button("Migraine w/ Nausea", use_container_width=True, on_click=set_symptoms, args=("Patient reports an intense, throbbing migraine accompanied by nausea.",))
        sc3.button("Chest & Reflux", use_container_width=True, on_click=set_symptoms, args=("Patient has severe heartburn after eating, and sour liquid keeps coming up.",))
        sc4.button("Itchy Arm Rash", use_container_width=True, on_click=set_symptoms, args=("Patient developed a very itchy, red, cracked rash with some blisters on the arm.",))

    if analyze_btn:
        if not user_query:
            st.error("Please enter patient symptoms.")
        else:
            with st.spinner("Executing semantic search and LLM synthesis..."):
                try:
                    pipeline = get_pipeline()
                    success, result = pipeline.process_query(user_query)
                except Exception as e:
                    success = False
                    result = str(e)

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

    # --- Session State Init ---
    if "page" not in st.session_state:
        st.session_state.page = "analyzer"
    if "symptoms" not in st.session_state:
        st.session_state.symptoms = ""
    if "history" not in st.session_state:
        st.session_state.history = []

    # --- SIDEBAR NAV ---
    with st.sidebar:
        st.markdown("""
            <div style='padding: 20px 0 30px 0; text-align: center;'>
                <div style='font-size: 3rem; margin-bottom: 10px;'>⚕️</div>
                <h2 style='color: #ffffff; font-size: 1.8rem; font-weight: 700; margin: 0; letter-spacing: 1px;'>Aura Health</h2>
                <p style='color: #94a3b8; font-size: 0.9rem; margin-top: 5px;'>Diagnostic Intelligence</p>
            </div>
        """, unsafe_allow_html=True)

        pages = [
            ("analyzer",  "🏥", "Symptom Analyzer"),
            ("history",   "📋", "Patient History"),
            ("knowledge", "📚", "Knowledge Base"),
            ("settings",  "⚙️", "System Settings"),
        ]
        
        st.markdown("<div style='padding: 0 10px;'>", unsafe_allow_html=True)
        for key, icon, label in pages:
            if st.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True):
                st.session_state.page = key
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        history_count = len(st.session_state.get("history", []))
        st.markdown(f"""
            <div style='margin-top: 60px; padding: 20px; background: rgba(255,255,255,0.05); border-radius: 12px; text-align: center;'>
                <div style='color: #cbd5e1; font-weight: 600; font-size: 1.1rem;'>{history_count}</div>
                <div style='color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;'>Session Records</div>
                <div style='margin-top: 15px; color: #64748b; font-size: 0.75rem;'>v2.4.1 Premium Edition</div>
            </div>
        """, unsafe_allow_html=True)

    # --- PAGE ROUTING ---
    page = st.session_state.page
    if page == "analyzer":
        page_symptom_analyzer()
    elif page == "history":
        page_patient_history()
    elif page == "knowledge":
        page_knowledge_base()
    elif page == "settings":
        page_system_settings()
        
    # Eagerly initialize pipeline in the background so it doesn't block the initial interface load
    if "pipeline_warmed_up" not in st.session_state:
        st.session_state.pipeline_warmed_up = True
        try:
            from streamlit.runtime.scriptrunner import add_script_run_ctx
            t = threading.Thread(target=get_pipeline)
            add_script_run_ctx(t)
            t.start()
        except Exception as e:
            logger.error(f"Failed to start background warmup: {e}")
            get_pipeline() # Fallback to blocking if thread fails


if __name__ == "__main__":
    main()
