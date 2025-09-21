from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults
import dotenv

dotenv.load_dotenv()

tools = [TavilySearchResults(max_results=10)]

# initialize our model and tools (API key is taken from environment)
llm = ChatOpenAI(model="gpt-4o")
prompt = """
    Youre a financial assistant. You have access to the following tools:
    - TavilySearchResults
    - You can use the following tools to get information about the stock market:
    - TavilySearchResults
    provide financial information for the user.
"""

# Compile the builder into an executable graph
graph = create_react_agent(model=llm, prompt=prompt, name="react_agent", tools=tools)
