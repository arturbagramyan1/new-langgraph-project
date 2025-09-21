from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

# initialize our model and tools (API key is taken from environment)
llm = ChatOpenAI(model="gpt-4o")
prompt = """
    You are a helpful AI assistant trained in creating engaging social media content!
    you have access to two tools: basic_research and get_todays_date. Please get_todays_date then 
    perform any research if needed, before generating a social media post.
"""

# Compile the builder into an executable graph
graph = create_react_agent(model=llm, prompt=prompt, name="react_agent", tools=[])
