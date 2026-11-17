"""LangChain + LangGraph version of the question answering flow.

    retrieve -> build_context -> answer --(found?)--> END
                     ^                      |
                     +---- not found -------+   (second look: twice the passages, once)

The retriever is a LangChain BaseRetriever over FinSight's own indexes, and the answer step is a
LangChain chain (prompt | Groq chat model | JSON parser). Same prompt, model, context budget and
retry rule as the plain Python path, so the evaluation numbers carry over (check with
python run.py check-graph).
"""
from typing import Any, List, Optional, TypedDict

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph

import config

from . import llm, retrieval


class FinSightRetriever(BaseRetriever):
    """Exposes any FinSight search method (plain, table aware, hybrid, visual) as a LangChain retriever."""
    searcher: Any
    ticker: str
    method: str
    k: int = 40

    def _get_relevant_documents(self, query: str, *,
                                run_manager: Optional[CallbackManagerForRetrieverRun] = None) -> List[Document]:
        hits = self.searcher.search(query, self.ticker, self.method, k=self.k)
        return [Document(page_content=h["text"], metadata={k: v for k, v in h.items() if k != "text"})
                for h in hits]


def answer_chain(model: str):
    from langchain_groq import ChatGroq
    extra = {"reasoning_effort": config.REASONING_EFFORT} if config.REASONING_EFFORT and "gpt-oss" in model else {}
    chat = ChatGroq(model=model, api_key=config.GROQ_API_KEY, temperature=0, max_retries=6,
                    model_kwargs={"response_format": {"type": "json_object"}}, **extra)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system}"),
        ("human", "Excerpts:\n{context}\n\nQuestion: {question}"),
    ])
    return prompt | chat | StrOutputParser() | RunnableLambda(llm._parse_json)


class QAState(TypedDict, total=False):
    question: str
    require_value: bool   # numeric questions: an answer with no number counts as not found
    docs: list
    context: list
    result: dict
    attempts: int


def _docs_to_hits(docs: list) -> list:
    return [{**d.metadata, "text": d.page_content} for d in docs]


def build_graph(retriever: FinSightRetriever, model: str = None, max_attempts: int = 2):
    chain = answer_chain(model or config.LLM_MODEL)

    def retrieve(state: QAState) -> QAState:
        return {"docs": retriever.invoke(state["question"]), "attempts": 0}

    def build_context(state: QAState) -> QAState:
        budget = config.CONTEXT_CHARS * (state["attempts"] + 1)  # second look reads twice as much
        return {"context": retrieval.fill_context(_docs_to_hits(state["docs"]), budget=budget)}

    def answer(state: QAState) -> QAState:
        try:
            out = chain.invoke({"system": llm.ANSWER_SYSTEM, "question": state["question"],
                                "context": llm.format_context(state["context"])})
        except Exception as e:
            msg = str(e).lower()
            if "per day" in msg or "(tpd)" in msg or "(rpd)" in msg:
                raise llm.DailyLimit(str(e)[:300]) from None
            raise
        return {"result": out, "attempts": state["attempts"] + 1}

    def found_enough(state: QAState) -> str:
        r = state["result"]
        ok = bool(r.get("found")) and (not state.get("require_value") or r.get("value") not in (None, "", "null"))
        return "done" if ok or state["attempts"] >= max_attempts else "look_again"

    g = StateGraph(QAState)
    g.add_node("retrieve", retrieve)
    g.add_node("build_context", build_context)
    g.add_node("answer", answer)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "build_context")
    g.add_edge("build_context", "answer")
    g.add_conditional_edges("answer", found_enough, {"done": END, "look_again": "build_context"})
    return g.compile()


def ask(searcher, question: str, ticker: str, method: str, model: str = None,
        require_value: bool = False, max_attempts: int = 2) -> dict:
    graph = build_graph(FinSightRetriever(searcher=searcher, ticker=ticker, method=method), model, max_attempts)
    state = graph.invoke({"question": question, "require_value": require_value})
    out = dict(state["result"])
    out["context"] = state["context"]
    out["retrieved_pages"] = retrieval.ranked_pages(_docs_to_hits(state["docs"]))
    out["attempts"] = state["attempts"]
    return out
