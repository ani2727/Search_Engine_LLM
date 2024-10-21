import streamlit as st
import os
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["PROJECT_NAME"] = "Langchain Project"

# Initialize Arxiv and Wikipedia tools
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv_tool = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

api_wrapper_wikipedia = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=250)
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper_wikipedia)

search = DuckDuckGoSearchRun(name="Search")

st.title("🔍 Langchain - Chat")

"""
In this example, we're using StreamlitCallbackHandler to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more Langchain Streamlit agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent)
"""

# Sidebar settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma-7b-It", streaming=True)
    tools = [arxiv_tool, wikipedia_tool, search]

    # Set handle_parsing_errors to True
    search_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
