"""A small tool-calling LangGraph: model -> optional tool -> model."""

from typing import Annotated, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from agent.prompts import SYSTEM_PROMPT
from rag.tools import search_knowledge_base
from rag.store import get_api_key


class AssistantState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_graph():
    """Build and compile the conversational tool-calling graph."""
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, api_key=get_api_key())
    model_with_tools = model.bind_tools([search_knowledge_base])

    def call_model(state: AssistantState):
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        return {"messages": [model_with_tools.invoke(messages)]}

    graph = StateGraph(AssistantState)
    graph.add_node("assistant", call_model)
    graph.add_node("tools", ToolNode([search_knowledge_base]))
    graph.add_edge(START, "assistant")
    graph.add_conditional_edges("assistant", tools_condition)
    graph.add_edge("tools", "assistant")
    return graph.compile()
