from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3

load_dotenv()

llm = ChatGoogleGenerativeAI(model = 'gemini-3-flash-preview')

class BotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: BotState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn= conn)

graph = StateGraph(BotState)
graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_threads():
    all_threads =set()
    for item in checkpointer.list(None):
        all_threads.add(item.config['configurable']['thread_id'])

    return list(all_threads)

