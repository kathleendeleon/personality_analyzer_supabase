import os
import re
import uuid
from typing import TypedDict, List, Literal, Dict, Optional
from datetime import datetime, timezone

import streamlit as st

# LangChain / LLMs
# Prefer modern integration; fallback for older LangChain if needed.
from langchain_openai import ChatOpenAI  # modern path
#from langchain.chat_models import ChatOpenAI  # fallback
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangGraph
from langgraph.graph import StateGraph, END

# Supabase
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except Exception:
    SUPABASE_AVAILABLE = False

# ----------------------- Secrets & Keys -----------------------
# --- OpenAI key (Streamlit Cloud: set in Settings → Secrets) ---
api_key = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))
if not api_key:
    st.error("🔐 Missing OpenAI API key. Add OPENAI_API_KEY in Streamlit → Settings → Secrets.")
    st.stop()

# Ensure downstream libs see it (tiktoken / OpenAI SDK)
os.environ["OPENAI_API_KEY"] = api_key

# --- Supabase (optional; enable in sidebar) ---
SUPABASE_URL = os.getenv("SUPABASE_URL", st.secrets.get("SUPABASE_URL", ""))
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", st.secrets.get("SUPABASE_ANON_KEY", ""))

# ----------------------- UI -----------------------
st.set_page_config(page_title="Personality Analyzer — LangGraph + Supabase", page_icon="🧭", layout="wide")
st.title("👩🏻‍💻 Kath's Personality Analyzer — LangGraph Edition")
st.caption("LangGraph orchestration • MBTI category routing • Optional Supabase persistence")

with st.sidebar:
    st.header("Input")
    mode = st.radio("Content source", ["Paste Text", "Web URL"])
    model_name = st.selectbox("LLM model", ["gpt-5-nano", "gpt-4o-mini"], index=1)

    if model_name == "gpt-5-nano":
        temperature = 1.0
        st.sidebar.caption("`gpt-5-nano` uses fixed temperature = 1.0")
    else:
        temperature = st.sidebar.slider("Temperature", 0.0, 1.2, 0.7, 0.1)
        
    #temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    max_chars = st.number_input("Max characters", min_value=500, max_value=20000, value=4000, step=500)
    chunk_size = st.number_input("Chunk size", min_value=600, max_value=2400, value=1200, step=100)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=600, value=150, step=10)

    st.divider()
    st.subheader("Supabase (optional)")
    save_to_supabase = st.checkbox("Save results to Supabase", value=False,
                                   help="Toggle on to persist each run. Configure secrets first.")
    table_name = st.text_input("Table name", value="personality_runs")

if mode == "Paste Text":
    input_text = st.text_area("Paste writing samples:", height=240, placeholder="Paste up to several paragraphs…")
    source_url: Optional[str] = None
else:
    source_url = st.text_input("Web URL to analyze", placeholder="https://example.com/article")
    input_text = ""

run = st.button("Analyze")
st.markdown("---")

# ----------------------- State -----------------------
class AppState(TypedDict):
    text: str
    chunks: List[str]
    partials: List[str]
    report: str
    mbti_guess: str
    mbti_category: Literal["Analyst","Diplomat","Sentinel","Explorer","Unknown"]
    vacation_plan: str

# ----------------------- LLM + Prompts -----------------------
def make_llm(model_name: str, temperature: float):
    # Pass api_key explicitly to avoid relying on global env if desired.
    return ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)

analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a senior psychologist and data scientist. Be precise, evidence-aware, and concise."),
    ("user",
     """Analyze the following writing for personality signals.
Return:
1) Emotional tone (1-2 lines)
2) Big Five (O,C,E,A,N) with 1-2 sentence justification each
3) Likely MBTI type (best guess + 2nd guess) with rationale
4) Strengths & watchouts (bullets)
5) Two book or career recommendations
Text:
{passage}
""")
])

synthesis_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an executive editor. Merge multiple analyses into a single, coherent report under 500 words."),
    ("user", "Combine the following partial analyses into ONE final report:\n\n{partials}")
])

mbti_prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify MBTI from writing. If unsure, make your best-justified guess."),
    ("user",
     """From this text, guess a likely MBTI type and map it to a category set: 
Analyst (INTJ, INTP, ENTJ, ENTP), Diplomat (INFJ, INFP, ENFJ, ENFP), Sentinel (ISTJ, ISFJ, ESTJ, ESFJ), Explorer (ISTP, ISFP, ESTP, ESFP).
Return exactly two lines:
MBTI: <type>
Category: <Analyst|Diplomat|Sentinel|Explorer|Unknown>
Text:
{text}
""")
])

parser = StrOutputParser()

# ----------------------- Helpers -----------------------
def load_text_from_url(url: str, max_chars: int) -> str:
    loader = WebBaseLoader(url)
    docs = loader.load()
    return "\n\n".join([d.page_content for d in docs])[:max_chars]

def chunk_text(text: str, size: int, overlap: int, max_chars: int) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return splitter.split_text(text[:max_chars])

def extract_mbti_and_category(output: str) -> (str, str):
    mbti = "Unknown"
    cat = "Unknown"
    for line in output.splitlines():
        if line.strip().lower().startswith("mbti:"):
            mbti = line.split(":",1)[1].strip().upper()
        if line.strip().lower().startswith("category:"):
            cat = line.split(":",1)[1].strip().title()
    if cat not in {"Analyst","Diplomat","Sentinel","Explorer"}:
        cat = "Unknown"
    return mbti, cat

VACATION_PLANS: Dict[str, str] = {
    "Analyst": (
        "• Kyoto, Japan — culture + quiet planning time\n"
        "• Zurich & Interlaken — alpine hikes, efficient transit, data museum stop\n"
        "• Iceland road trip — solitude, photography, geothermal spas\n"
        "• Work-cation: Lisbon co-working + weekend Sintra/Évora analysis walks"
    ),
    "Diplomat": (
        "• Bali retreat — yoga, community workshops, mindful cuisine\n"
        "• Costa Rica — eco-lodges, wildlife volunteering, surf & sunsets\n"
        "• Tuscany — small group cooking classes, vineyard chats, slow travel\n"
        "• Cultural immersion: Oaxaca homestay + Spanish classes"
    ),
    "Sentinel": (
        "• London/Edinburgh — museums, orderly itineraries, classic tea rooms\n"
        "• National Parks (Utah Mighty 5) — well-marked trails, ranger programs\n"
        "• Switzerland rail loop — punctual journeys, scenic but structured\n"
        "• Heritage tours: Washington D.C. civics + Smithsonian deep dives"
    ),
    "Explorer": (
        "• Vietnam motorbike loop — street food, spontaneous detours\n"
        "• New Zealand campervan — bungee, glacier hikes, blackwater rafting\n"
        "• Morocco — souks, Sahara glamping, improvised photo walks\n"
        "• Festival hop: Barcelona → Porto → coastal surf days"
    ),
    "Unknown": "• City sampler with flexible passes (e.g., Tokyo/NYC) + day-by-day picks based on vibe."
}

# ----------------------- Node implementations -----------------------
def node_prepare(state: AppState) -> AppState:
    text = state["text"]
    _chunks = chunk_text(
        text,
        size=state.get("chunk_size", 1200),
        overlap=state.get("chunk_overlap", 150),
        max_chars=state.get("max_chars", 4000)
    )
    return {**state, "chunks": _chunks}

def node_analyze(state: AppState, llm: ChatOpenAI) -> AppState:
    partials: List[str] = []
    for ch in state["chunks"][:3]:
        out = (analysis_prompt | llm | parser).invoke({"passage": ch})
        partials.append(out)
    return {**state, "partials": partials}

def node_synthesize(state: AppState, llm: ChatOpenAI) -> AppState:
    merged = (synthesis_prompt | llm | parser).invoke({"partials": "\n\n---\n\n".join(state["partials"])})
    return {**state, "report": merged}

def node_mbti(state: AppState, llm: ChatOpenAI) -> AppState:
    res = (mbti_prompt | llm | parser).invoke({"text": state["report"] or state["text"]})
    mbti, cat = extract_mbti_and_category(res)
    return {**state, "mbti_guess": mbti, "mbti_category": cat}

# Conditional router
def route_by_category(state: AppState) -> Literal["Analyst","Diplomat","Sentinel","Explorer","Unknown"]:
    return state.get("mbti_category","Unknown") or "Unknown"

def node_plan_analyst(state: AppState) -> AppState:
    return {**state, "vacation_plan": VACATION_PLANS["Analyst"]}

def node_plan_diplomat(state: AppState) -> AppState:
    return {**state, "vacation_plan": VACATION_PLANS["Diplomat"]}

def node_plan_sentinel(state: AppState) -> AppState:
    return {**state, "vacation_plan": VACATION_PLANS["Sentinel"]}

def node_plan_explorer(state: AppState) -> AppState:
    return {**state, "vacation_plan": VACATION_PLANS["Explorer"]}

def node_plan_unknown(state: AppState) -> AppState:
    return {**state, "vacation_plan": VACATION_PLANS["Unknown"]}

# ----------------------- Build Graph -----------------------
def build_graph(llm: ChatOpenAI):
    g = StateGraph(AppState)
    # Register nodes
    g.add_node("prepare", node_prepare)
    g.add_node("analyze", lambda s: node_analyze(s, llm))
    g.add_node("synthesize", lambda s: node_synthesize(s, llm))
    g.add_node("mbti", lambda s: node_mbti(s, llm))

    g.add_node("plan_analyst", node_plan_analyst)
    g.add_node("plan_diplomat", node_plan_diplomat)
    g.add_node("plan_sentinel", node_plan_sentinel)
    g.add_node("plan_explorer", node_plan_explorer)
    g.add_node("plan_unknown", node_plan_unknown)

    # Edges
    g.add_edge("prepare", "analyze")
    g.add_edge("analyze", "synthesize")
    g.add_edge("synthesize", "mbti")
    g.add_conditional_edges("mbti", route_by_category, {
        "Analyst": "plan_analyst",
        "Diplomat": "plan_diplomat",
        "Sentinel": "plan_sentinel",
        "Explorer": "plan_explorer",
        "Unknown": "plan_unknown",
    })
    g.add_edge("plan_analyst", END)
    g.add_edge("plan_diplomat", END)
    g.add_edge("plan_sentinel", END)
    g.add_edge("plan_explorer", END)
    g.add_edge("plan_unknown", END)

    g.set_entry_point("prepare")
    return g.compile()

# ----------------------- Supabase -----------------------
def make_supabase(url: str, key: str) -> Optional[Client]:
    if not SUPABASE_AVAILABLE:
        return None
    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception:
        return None

def save_run_to_supabase(
    sb: Client,
    table: str,
    *,
    source_mode: str,
    source_url: Optional[str],
    model_name: str,
    mbti: str,
    category: str,
    report: str,
    vacation_plan: str,
    chunks_count: int,
    partials: List[str],
    temperature: float,
    max_chars: int,
    chunk_size: int,
    chunk_overlap: int,
) -> Optional[dict]:
    payload = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_mode": source_mode,
        "source_url": source_url,
        "model": model_name,
        "temperature": temperature,
        "max_chars": max_chars,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunks_count": chunks_count,
        "mbti_guess": mbti,
        "mbti_category": category,
        "report": report,
        "vacation_plan": vacation_plan,
        "partials": partials,  # JSONB
    }
    try:
        resp = sb.table(table).insert(payload).execute()
        return {"status": "ok", "count": getattr(resp, "count", None)}
    except Exception as e:
        return {"status": "error", "error": str(e)}

# ----------------------- App Run -----------------------
if run:
    # Validate text
    if mode == "Web URL":
        if not source_url:
            st.error("Please provide a URL.")
            st.stop()
        try:
            text = load_text_from_url(source_url, int(max_chars))
        except Exception as e:
            st.error(f"Failed to load URL: {e}")
            st.stop()
    else:
        text = input_text

    if not text or len(text.strip()) < 50:
        st.error("Please provide at least ~50 characters of text.")
        st.stop()

    # Initialize state
    state: AppState = {
        "text": text[: int(max_chars)],
        "chunks": [],
        "partials": [],
        "report": "",
        "mbti_guess": "Unknown",
        "mbti_category": "Unknown",
        "vacation_plan": "",
        "chunk_size": int(chunk_size),
        "chunk_overlap": int(chunk_overlap),
        "max_chars": int(max_chars),
    }  # type: ignore

    # LLM + Graph
    llm = make_llm(model_name, float(temperature))
    app = build_graph(llm)

    with st.status("Running LangGraph pipeline...", expanded=False):
        final_state = app.invoke(state)

    # UI output
    st.subheader("Personality Report (Merged)")
    st.write(final_state.get("report","(no report)"))

    st.subheader("MBTI Guess & Category")
    st.write(f"**MBTI:** {final_state.get('mbti_guess','Unknown')}")
    st.write(f"**Category:** {final_state.get('mbti_category','Unknown')}")

    st.subheader("Vacation Plan Recommendation")
    st.text(final_state.get("vacation_plan","(no plan)"))

    st.download_button(
        "Download Report (.txt)",
        data=(final_state.get("report","") + "\n\n---\nMBTI: " + final_state.get("mbti_guess","Unknown") +
              "\nCategory: " + final_state.get("mbti_category","Unknown") +
              "\n\nVacation Plan:\n" + final_state.get("vacation_plan","")),
        file_name=f"langgraph_personality_{uuid.uuid4().hex[:8]}.txt"
    )

    # Save to Supabase if enabled
    if save_to_supabase:
        if not SUPABASE_AVAILABLE:
            st.error("supabase-py is not installed. Add `supabase` to requirements.txt.")
        elif not SUPABASE_URL or not SUPABASE_ANON_KEY:
            st.error("Missing Supabase secrets. Add SUPABASE_URL and SUPABASE_ANON_KEY in Secrets.")
        else:
            sb = make_supabase(SUPABASE_URL, SUPABASE_ANON_KEY)
            if not sb:
                st.error("Failed to initialize Supabase client. Check URL/key.")
            else:
                with st.status("Saving run to Supabase…", expanded=False):
                    result = save_run_to_supabase(
                        sb, table_name,
                        source_mode=mode,
                        source_url=source_url,
                        model_name=model_name,
                        mbti=final_state.get("mbti_guess","Unknown"),
                        category=final_state.get("mbti_category","Unknown"),
                        report=final_state.get("report",""),
                        vacation_plan=final_state.get("vacation_plan",""),
                        chunks_count=len(final_state.get("chunks", [])),
                        partials=final_state.get("partials", []),
                        temperature=float(temperature),
                        max_chars=int(max_chars),
                        chunk_size=int(chunk_size),
                        chunk_overlap=int(chunk_overlap),
                    )
                if result and result.get("status") == "ok":
                    st.success("Saved to Supabase ✅")
                else:
                    st.error(f"Supabase insert error: {result.get('error') if result else 'Unknown error'}")

st.markdown("---")
st.markdown("**Notes**")
st.markdown("""
- **LangGraph** orchestrates conditional routing by MBTI category.
- **LangChain** handles chunking, prompts, and URL loading.
- **MBTI Categories:** \n
     **Analyst** - INTJ/INTP/ENTJ/ENTP \n
     **Diplomat** - INFJ/INFP/ENFJ/ENFP \n
     **Sentinel** - ISTJ/ISFJ/ESTJ/ESFJ \n
     **Explorer** - ISTP/ISFP/ESTP/ESFP
""")
