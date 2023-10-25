from langchain.agents import initialize_agent
from langchain.agents import Tool
from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.agents.agent_types import AgentType

from dotenv import dotenv_values

from tools.RetrievalTool import retrieval_qa_tool
from tools.SPListTool import CustomPandasTool


ENV = dotenv_values(".env")

retrieval_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0,
                           verbose=True, streaming=True, openai_api_key=ENV['OPENAI_API_KEY'])

agent_llm = OpenAI(model_name="text-davinci-003", temperature=0,
                   verbose=True, streaming=True, openai_api_key=ENV['OPENAI_API_KEY'])

docs_search = Chroma(persist_directory="../vector_store",
                     embedding_function=HuggingFaceEmbeddings())
tools = [
    retrieval_qa_tool(llm=retrieval_llm, retriever=docs_search.as_retriever()),
    CustomPandasTool()
]

agent = initialize_agent(
    tools=tools,
    llm=agent_llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)
