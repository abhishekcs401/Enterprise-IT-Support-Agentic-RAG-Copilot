import logging
from typing import Literal
# from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from app.core.config import get_settings
from app.rag.state import AgentState, RouteDecision, EvidenceGrade
from app.rag.vectorstore import get_retriever

logger = logging.getLogger(__name__)
settings = get_settings()

_llm = None
_web_search = None

# def llm():
#     global _llm
#     if _llm is None:
#         if not settings.openai_api_key:
#             raise RuntimeError("OPENAI_API_KEY is missing")
#         _llm = ChatOpenAI(
#             model=settings.openai_model,
#             temperature=0,
#             api_key=settings.openai_api_key,
#         )
#     return _llm


def llm():
    global _llm
    if _llm is None:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is missing")
        _llm = ChatGroq(
            model=settings.groq_model,
            temperature=0,
            api_key=settings.groq_api_key,
        )
    return _llm


def format_history(history: list[dict] | None, max_chars: int = 600) -> str:
    """Format chat history for prompts. Long messages are trimmed to save tokens."""
    if not history:
        return "No previous conversation."

    lines = []
    for item in history:
        content = item["content"]
        if len(content) > max_chars:
            content = content[:max_chars] + "..."
        lines.append(f"{item['role'].upper()}: {content}")
    return "\n".join(lines)


def web_search_tool():
    global _web_search
    if _web_search is None:
        if not settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is missing")
        _web_search = TavilySearch(
            tavily_api_key=settings.tavily_api_key,
            max_results=5,
            topic="general",
            include_answer=True,
            include_raw_content=False,
        )
    return _web_search


def add_trace(state: AgentState, message: str):
    return [*state.get("trace", []), message]


# ---------------------------------------------------------------- routing
def route_question(state: AgentState):
    router = llm().with_structured_output(RouteDecision, method="json_mode")
    decision = router.invoke(f"""
You route messages for an enterprise IT support assistant.
Use kb for questions about company IT policies, VPN, password reset, MFA, laptop setup,
software access, security, email, devices, troubleshooting, or technology support.
Also use kb for follow-up questions that continue an earlier IT topic.
Use direct for greetings, thanks, casual chat, personal statements (e.g. "my name is X"),
or questions about the conversation itself (e.g. "what did I ask?", "what is my name?").

Previous conversation:
{format_history(state.get('history', []))}

Question: {state['question']}
Return valid JSON like {{"route":"kb"}}.
""")
    return {
        "source_used": decision.route,
        "trace": add_trace(state, f"Router → {decision.route.upper()}"),
    }


def route_after_router(state: AgentState) -> Literal["contextualize_question", "direct_answer"]:
    return "contextualize_question" if state["source_used"] == "kb" else "direct_answer"


# ------------------------------------------------- follow-up -> standalone
def contextualize_question(state: AgentState):
    """Turn a short follow-up ("uska step batao") into a standalone question."""
    history = state.get("history", [])
    if not history:
        return {
            "current_query": state["question"],
            "trace": add_trace(state, "No history → using question as-is"),
        }

    standalone = llm().invoke(f"""
Given the conversation and a follow-up question, rewrite the follow-up as a
standalone question. Resolve pronouns like "it", "that", "uska", "isko".
Do NOT answer it. If it is already standalone, return it unchanged.
Return only the question.

Conversation:
{format_history(history[-6:])}

Follow-up: {state['question']}
""").content.strip()

    return {
        "current_query": standalone or state["question"],
        "trace": add_trace(state, f"Contextualized → {standalone}"),
    }


# -------------------------------------------------------------- retrieval
def retrieve_kb(state: AgentState):
    docs = get_retriever().invoke(state["current_query"])
    return {
        "kb_docs": docs,
        "trace": add_trace(state, f"Private KB retrieval → {len(docs)} chunks"),
    }


def grade_kb(state: AgentState):
    grader = llm().with_structured_output(EvidenceGrade, method="json_mode")
    context = "\n\n".join(
        f"Source: {d.metadata.get('source', 'unknown')}\n{d.page_content}"
        for d in state["kb_docs"]
    )
    grade = grader.invoke(f"""
You grade evidence for an enterprise IT support assistant.

Previous conversation:
{format_history(state.get('history', []))}

Question: {state['current_query']}
Private company KB evidence:
{context}

Return good only if the evidence is sufficient to answer confidently and specifically.
Otherwise return weak. JSON: {{"grade":"good"}} or {{"grade":"weak"}}.
""")
    return {
        "kb_grade": grade.grade,
        "trace": add_trace(state, f"KB evidence grade → {grade.grade.upper()}"),
    }


def after_kb(state: AgentState) -> Literal["generate_from_kb", "search_web"]:
    return "generate_from_kb" if state["kb_grade"] == "good" else "search_web"


# -------------------------------------------------------------------- web
def search_web(state: AgentState):
    result = web_search_tool().invoke({"query": state["current_query"]})
    lines, citations = [], []
    if isinstance(result, dict):
        if result.get("answer"):
            lines.append("Search answer: " + result["answer"])
        for item in result.get("results", []):
            title, url, content = item.get("title", ""), item.get("url", ""), item.get("content", "")
            lines.append(f"Title: {title}\nURL: {url}\nContent: {content}")
            citations.append({"title": title or url, "url": url, "type": "web"})
    else:
        lines.append(str(result))
    return {
        "web_results": "\n\n".join(lines),
        "citations": citations,
        "source_used": "web",
        "trace": add_trace(state, "Web fallback → Tavily search"),
    }


def grade_web(state: AgentState):
    grader = llm().with_structured_output(EvidenceGrade, method="json_mode")
    grade = grader.invoke(f"""
Previous conversation:
{format_history(state.get('history', []))}

Question: {state['current_query']}
Web evidence:
{state['web_results']}

Return good if the evidence is sufficient and directly relevant; otherwise weak.
Return valid JSON like {{"grade":"good"}} or {{"grade":"weak"}}.
""")
    return {
        "web_grade": grade.grade,
        "trace": add_trace(state, f"Web evidence grade → {grade.grade.upper()}"),
    }


def after_web(state: AgentState) -> Literal["generate_from_web", "rewrite_query", "insufficient"]:
    if state["web_grade"] == "good":
        return "generate_from_web"
    if state["retry_count"] < settings.max_retries:
        return "rewrite_query"
    return "insufficient"


def rewrite_query(state: AgentState):
    rewritten = llm().invoke(f"""
Rewrite this IT support question for better private knowledge retrieval and vendor web search.
Preserve intent, add useful technical keywords, do not answer, return only the query.

Conversation (for context only):
{format_history(state.get('history', [])[-6:])}

Question: {state['current_query']}
""").content.strip()
    return {
        "current_query": rewritten or state["current_query"],
        "retry_count": state["retry_count"] + 1,
        "trace": add_trace(state, f"Query rewrite → {rewritten}"),
    }


# ------------------------------------------------------------- generation
def generate_from_kb(state: AgentState):
    history = state.get("history", [])
    context = "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in state["kb_docs"]
    )

    answer = llm().invoke(f"""
You are an enterprise IT support copilot.
Answer the technical content ONLY from the private company KB below.
Use the previous conversation only to understand context and follow-up questions
(for example pronouns or what the user said earlier).
Be concise, actionable, and safe.
Do not invent policy details.

Previous conversation:
{format_history(history)}

Current question: {state['question']}
Standalone version: {state['current_query']}

Private KB:
{context}
""").content

    citations = []
    seen = set()
    for d in state["kb_docs"]:
        src = d.metadata.get("source", "Private KB")
        if src not in seen:
            seen.add(src)
            citations.append({
                "title": src.split("/")[-1],
                "url": "",
                "type": "private_kb",
            })

    return {
        "answer": answer,
        "source_used": "private_kb",
        "citations": citations,
        "trace": add_trace(state, "Answer generation → PRIVATE KB"),
    }


def generate_from_web(state: AgentState):
    history = state.get("history", [])
    answer = llm().invoke(f"""
You are an enterprise IT support copilot.
Answer the technical content ONLY from the web evidence below.
Use the previous conversation only to understand context and follow-up questions.
Clearly state that this is external web information and may need IT validation.

Previous conversation:
{format_history(history)}

Question: {state['question']}
Standalone version: {state['current_query']}

Web evidence:
{state['web_results']}
""").content

    return {
        "answer": answer,
        "source_used": "web_search",
        "trace": add_trace(state, "Answer generation → WEB SEARCH"),
    }


def direct_answer(state: AgentState):
    history = state.get("history", [])
    answer = llm().invoke(f"""
You are a helpful conversational IT assistant.
Use the previous conversation to answer questions about the user or the chat itself
(for example their name or what they asked earlier). If the information is not in the
conversation, say you don't know it yet and ask.

Previous conversation:
{format_history(history)}

Current question:
{state['question']}

Respond naturally in the same language the user writes in.
""").content

    return {
        "answer": answer,
        "source_used": "direct",
        "trace": add_trace(state, "Direct response → conversation memory"),
    }


def insufficient(state: AgentState):
    return {
        "answer": "I couldn't find enough reliable evidence in the company knowledge base or external search to answer confidently. Please contact the IT help desk or provide more details.",
        "source_used": "insufficient_evidence",
        "trace": add_trace(state, "Stopped → insufficient reliable evidence"),
    }


# ------------------------------------------------------------------ graph
def build_graph():
    graph = StateGraph(AgentState)
    for name, fn in {
        "route_question": route_question,
        "contextualize_question": contextualize_question,
        "retrieve_kb": retrieve_kb,
        "grade_kb": grade_kb,
        "search_web": search_web,
        "grade_web": grade_web,
        "rewrite_query": rewrite_query,
        "generate_from_kb": generate_from_kb,
        "generate_from_web": generate_from_web,
        "direct_answer": direct_answer,
        "insufficient": insufficient,
    }.items():
        graph.add_node(name, fn)

    graph.add_edge(START, "route_question")
    graph.add_conditional_edges("route_question", route_after_router, {
        "contextualize_question": "contextualize_question",
        "direct_answer": "direct_answer",
    })
    graph.add_edge("contextualize_question", "retrieve_kb")
    graph.add_edge("retrieve_kb", "grade_kb")
    graph.add_conditional_edges("grade_kb", after_kb, {
        "generate_from_kb": "generate_from_kb",
        "search_web": "search_web",
    })
    graph.add_edge("search_web", "grade_web")
    graph.add_conditional_edges("grade_web", after_web, {
        "generate_from_web": "generate_from_web",
        "rewrite_query": "rewrite_query",
        "insufficient": "insufficient",
    })
    graph.add_edge("rewrite_query", "retrieve_kb")
    graph.add_edge("generate_from_kb", END)
    graph.add_edge("generate_from_web", END)
    graph.add_edge("direct_answer", END)
    graph.add_edge("insufficient", END)
    return graph.compile()


agent_graph = build_graph()


def ask(question: str, history: list[dict] | None = None):
    initial: AgentState = {
        "question": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
        "trace": [],
        "citations": [],
        "history": history or [],
    }
    return agent_graph.invoke(initial)