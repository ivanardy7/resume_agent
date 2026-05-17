"""
Architecture:
1. Supervisor Agent      -> decides route: resume_search / chat_history / out_of_scope
2. Chat History Agent    -> answers questions about previous conversation
3. Query Understanding   -> creates better semantic query for Qdrant
4. Resume Search Agent   -> retrieves resume snippets from Qdrant
5. Resume Analyst Agent  -> analyzes retrieved resume evidence
6. Candidate Ranking     -> ranks candidate recommendations
7. Final Response Agent  -> writes final chatbot answer
"""

from __future__ import annotations
from typing import Any, Dict, List, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from config import CHAT_MODEL, IDR_PER_USD, OPENAI_API_KEY, TOP_K, validate_required_settings
from rag_tools import search_resume_tool
from prompts import (
    FINAL_RESPONSE_PROMPT,
    QUERY_UNDERSTANDING_PROMPT,
    RANKING_PROMPT,
    RESUME_ANALYST_PROMPT,
)

class AgentState(TypedDict, total=False):
    question: str
    chat_history: List[Dict[str, str]]
    # Routing
    route: str
    route_reason: str
    # RAG workflow
    search_query: str
    retrieved_docs: List[Dict[str, Any]]
    analysis: str
    ranking: str
    sources: List[Dict[str, Any]]
    # Final output
    answer: str
    # Token usage
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    estimated_cost_idr: float


def _get_llm() -> ChatOpenAI:
    validate_required_settings()
    return ChatOpenAI(
        model=CHAT_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0.2,
    )

def _history_to_text(chat_history: List[Dict[str, str]] | None) -> str:
    if not chat_history:
        return "No previous conversation."

    lines = []
    for msg in chat_history[-15:]:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")

    return "\n".join(lines)

def _add_usage(state: AgentState, response: Any) -> AgentState:
    usage = getattr(response, "usage_metadata", None) or {}

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    response_metadata = getattr(response, "response_metadata", {}) or {}
    token_usage = response_metadata.get("token_usage", {}) or {}

    input_tokens = input_tokens or token_usage.get("prompt_tokens", 0)
    output_tokens = output_tokens or token_usage.get("completion_tokens", 0)

    current_input = state.get("total_input_tokens", 0)
    current_output = state.get("total_output_tokens", 0)

    state["total_input_tokens"] = current_input + int(input_tokens)
    state["total_output_tokens"] = current_output + int(output_tokens)
    state["total_tokens"] = state["total_input_tokens"] + state["total_output_tokens"]
    state["estimated_cost_idr"] = IDR_PER_USD * (
        (state["total_input_tokens"] * 0.15 + state["total_output_tokens"] * 0.60)
        / 1_000_000
    )

    return state

# Supervisor / Router Agent
def supervisor_agent(state: AgentState) -> AgentState:
    llm = _get_llm()
    question = state["question"]
    history_text = _history_to_text(state.get("chat_history", []))

    response = llm.invoke(
        [
            SystemMessage(
                content="""
You are a Supervisor Agent for an HR Resume Screening Assistant.

Your task is to classify the user's question into exactly one route:

1. resume_search
Use this if the user asks about:
- resume
- candidate
- CV
- skills
- job role
- applicant
- work experience
- ranking candidates
- finding suitable candidates
- comparing candidates from the resume database

2. chat_history
Use this if the user asks about previous conversation or asks to compare/rank results already shown, for example:
- "tadi aku tanya apa?"
- "sebelumnya aku minta apa?"
- "kandidat nomor 1 tadi siapa?"
- "bandingkan dengan jawaban sebelumnya"
- "apa yang kamu jawab tadi?"
- "mana yang lebih bagus diantara mereka?"
- "siapa yang paling cocok dari daftar tadi?"
- "pilihkan satu yang terbaik"

3. greeting
Use this if the user says hello, hi, or simple greetings, for example:
- "halo"
- "hai"
- "selamat pagi"
- "pagi"
- "halo, apa kabar?"

4. out_of_scope
Use this if the question is not related to resume screening, chat history, or greeting.

Return only one of these labels:
resume_search
chat_history
greeting
out_of_scope
"""
            ),
            HumanMessage(
                content=(
                    f"Recent chat history:\n{history_text}\n\n"
                    f"Current user question:\n{question}\n\n"
                    "Route:"
                )
            ),
        ]
    )

    raw_route = response.content.strip().lower()

    if "chat_history" in raw_route:
        route = "chat_history"
    elif "greeting" in raw_route:
        route = "greeting"
    elif "out_of_scope" in raw_route:
        route = "out_of_scope"
    else:
        route = "resume_search"

    state["route"] = route
    state["route_reason"] = response.content.strip()

    return _add_usage(state, response)

def route_after_supervisor(state: AgentState) -> str:
    return state.get("route", "resume_search")

# Chat History Agent
def chat_history_agent(state: AgentState) -> AgentState:
    llm = _get_llm()
    question = state["question"]
    history_text = _history_to_text(state.get("chat_history", []))
    response = llm.invoke(
        [
            SystemMessage(
                content="""
You are a Chat History Agent for an HR Resume Screening Assistant.

Your job is to answer the user's question using the previous conversation history.
Do not search the resume database.

Guidelines:
- If the chat history contains candidate information (names, IDs, skills, rankings), use that information to answer the question.
- If the user asks to compare, rank, or pick the best from previously shown candidates, make your best recommendation based on the information available in the chat history.
- Be helpful and make a concrete recommendation even if the information is limited. Explain your reasoning based on what was previously discussed.
- Only say you don't have enough information if the chat history is truly empty or completely unrelated to the question.
- Always respond in the same language the user used to ask the question.
"""
            ),
            HumanMessage(
                content=(
                    f"Chat history:\n{history_text}\n\n"
                    f"Current user question:\n{question}\n\n"
                    "Answer based only on chat history."
                )
            ),
        ]
    )

    state["answer"] = response.content
    state["sources"] = []
    state["search_query"] = ""
    state["retrieved_docs"] = []
    state["analysis"] = ""
    state["ranking"] = ""

    return _add_usage(state, response)

# Greeting Agent
def greeting_agent(state: AgentState) -> AgentState:
    llm = _get_llm()
    question = state["question"]

    response = llm.invoke(
        [
            SystemMessage(
                content="""
You are a friendly HR Assistant. 
The user said hello. Respond warmly in the same language the user used.
Briefly mention that you can help with searching and analyzing resumes.
Keep it concise and professional.
"""
            ),
            HumanMessage(content=question),
        ]
    )

    state["answer"] = response.content
    state["sources"] = []
    state["search_query"] = ""
    state["retrieved_docs"] = []
    state["analysis"] = ""
    state["ranking"] = ""

    return _add_usage(state, response)

# Fallback Agent
def out_of_scope_agent(state: AgentState) -> AgentState:
    state["answer"] = (
        "Maaf, aplikasi ini difokuskan untuk membantu proses HR dalam mencari, "
        "menganalisis, dan meranking kandidat berdasarkan database resume. "
        "Silakan ajukan pertanyaan yang berkaitan dengan kandidat, resume, skill, "
        "pengalaman kerja, atau kebutuhan posisi tertentu. "
        "Contoh: 'Cari kandidat IT Manager' atau "
        "'Tampilkan kandidat dengan pengalaman Python dan SQL'."
    )
    state["sources"] = []
    state["search_query"] = ""
    state["retrieved_docs"] = []
    state["analysis"] = ""
    state["ranking"] = ""

    return state

# Query Understanding Agent
def query_understanding_agent(state: AgentState) -> AgentState:
    llm = _get_llm()
    question = state["question"]
    history_text = _history_to_text(state.get("chat_history", []))

    response = llm.invoke(
        [
            SystemMessage(content=QUERY_UNDERSTANDING_PROMPT),
            HumanMessage(
                content=(
                    f"Recent chat history:\n{history_text}\n\n"
                    f"User question:\n{question}\n\n"
                    "Create the best semantic search query for Qdrant."
                )
            ),
        ]
    )

    state["search_query"] = response.content.strip()
    return _add_usage(state, response)

# Resume Search Agent
def resume_search_agent(state: AgentState) -> AgentState:
    search_query = state.get("search_query") or state["question"]

    retrieved_docs = search_resume_tool.invoke(
        {
            "query": search_query,
            "k": TOP_K,
        }
    )

    state["retrieved_docs"] = retrieved_docs
    state["sources"] = retrieved_docs
    return state

# Resume Analyst Agent
def resume_analyst_agent(state: AgentState) -> AgentState:
    llm = _get_llm()
    question = state["question"]
    retrieved_docs = state.get("retrieved_docs", [])

    response = llm.invoke(
        [
            SystemMessage(content=RESUME_ANALYST_PROMPT),
            HumanMessage(
                content=(
                    f"User question:\n{question}\n\n"
                    f"Retrieved resume snippets:\n{retrieved_docs}\n\n"
                    "Analyze the candidate evidence."
                )
            ),
        ]
    )

    state["analysis"] = response.content
    return _add_usage(state, response)

# Candidate Ranking Agent
def candidate_ranking_agent(state: AgentState) -> AgentState:
    llm = _get_llm()
    question = state["question"]
    retrieved_docs = state.get("retrieved_docs", [])
    analysis = state.get("analysis", "")

    response = llm.invoke(
        [
            SystemMessage(content=RANKING_PROMPT),
            HumanMessage(
                content=(
                    f"User question:\n{question}\n\n"
                    f"Retrieved resume snippets:\n{retrieved_docs}\n\n"
                    f"Resume analysis:\n{analysis}\n\n"
                    "Create a ranked candidate recommendation."
                )
            ),
        ]
    )

    state["ranking"] = response.content
    return _add_usage(state, response)

# Final Response Agent
def final_response_agent(state: AgentState) -> AgentState:
    llm = _get_llm()
    question = state["question"]
    analysis = state.get("analysis", "")
    ranking = state.get("ranking", "")
    sources = state.get("sources", [])

    response = llm.invoke(
        [
            SystemMessage(content=FINAL_RESPONSE_PROMPT),
            HumanMessage(
                content=(
                    f"User question:\n{question}\n\n"
                    f"Analysis:\n{analysis}\n\n"
                    f"Ranking:\n{ranking}\n\n"
                    f"Sources:\n{sources}\n\n"
                    "Write the final answer in the same language the user used. "
                    "Make it natural, clear, and helpful for an HR user."
                )
            ),
        ]
    )

    state["answer"] = response.content
    return _add_usage(state, response)

# LangGraph Workflow
def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor_agent", supervisor_agent)
    workflow.add_node("chat_history_agent", chat_history_agent)
    workflow.add_node("greeting_agent", greeting_agent)
    workflow.add_node("out_of_scope_agent", out_of_scope_agent)
    workflow.add_node("query_understanding_agent", query_understanding_agent)
    workflow.add_node("resume_search_agent", resume_search_agent)
    workflow.add_node("resume_analyst_agent", resume_analyst_agent)
    workflow.add_node("candidate_ranking_agent", candidate_ranking_agent)
    workflow.add_node("final_response_agent", final_response_agent)

    workflow.add_edge(START, "supervisor_agent")

    workflow.add_conditional_edges(
        "supervisor_agent",
        route_after_supervisor,
        {
            "resume_search": "query_understanding_agent",
            "chat_history": "chat_history_agent",
            "greeting": "greeting_agent",
            "out_of_scope": "out_of_scope_agent",
        },
    )

    # Resume RAG path
    workflow.add_edge("query_understanding_agent", "resume_search_agent")
    workflow.add_edge("resume_search_agent", "resume_analyst_agent")
    workflow.add_edge("resume_analyst_agent", "candidate_ranking_agent")
    workflow.add_edge("candidate_ranking_agent", "final_response_agent")
    workflow.add_edge("final_response_agent", END)

    # Chat history path
    workflow.add_edge("chat_history_agent", END)

    # Greeting path
    workflow.add_edge("greeting_agent", END)

    # Fallback path
    workflow.add_edge("out_of_scope_agent", END)

    return workflow.compile()

_agent_graph = None

def run_hr_agent(question: str, chat_history: List[Dict[str, str]] | None = None) -> AgentState:
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_graph()
    initial_state: AgentState = {
        "question": question,
        "chat_history": chat_history or [],
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_idr": 0.0,
    }

    return _agent_graph.invoke(initial_state)