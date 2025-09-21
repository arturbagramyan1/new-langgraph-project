from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


tools = [TavilySearch(max_results=10)]

# initialize our model and tools (API key is taken from environment)
llm = ChatOpenAI(model="gpt-4o")
prompt = """
    You are a financial assistant. You have access to a web search tool (TavilySearch).
    Use it to gather up-to-date information before answering.
"""

# Compile the builder into an executable graph
graph = create_react_agent(model=llm, prompt=prompt, name="react_agent", tools=tools)
