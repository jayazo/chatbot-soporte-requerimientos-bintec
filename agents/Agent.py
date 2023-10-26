from langchain.agents import initialize_agent
from langchain.agents import Tool, LLMSingleActionAgent, AgentExecutor
from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.agents.agent_types import AgentType
from langchain.chains import LLMChain

from dotenv import dotenv_values

from tools.RetrievalTool import retrieval_qa_tool
from tools.SPListTool import CustomPandasTool

from prompt_templates.custom_templates import CustomPromptTemplate, CustomOutputParser, MAIN_AGENT_PROMPT_TEMPLATE

ENV = dotenv_values(".env")

# Define models (llms)
retrieval_llm = OpenAI(model="text-davinci-003", temperature=0,
                           verbose=True, streaming=True, openai_api_key=ENV['OPENAI_API_KEY'])

agent_llm = OpenAI(model_name="text-davinci-003", temperature=0,
                   verbose=True, streaming=True, openai_api_key=ENV['OPENAI_API_KEY'])

# Retriever
docs_search = Chroma(persist_directory="../vector_store",
                     embedding_function=HuggingFaceEmbeddings())

# Tools
tools = [
    retrieval_qa_tool(llm=retrieval_llm, retriever=docs_search.as_retriever()),
    CustomPandasTool()
]

# tool_names = [tool.name for tool in tools]

# # Prompt + OutputParser
# main_prompt = CustomPromptTemplate(
#    template=MAIN_AGENT_PROMPT_TEMPLATE,
#    tools=tools,
#    input_variables=["input", "intermediate_steps"]
# )

# main_output_parser = CustomOutputParser()


# # LLM Chain + Agent
# llm_chain = LLMChain(llm=agent_llm, prompt=main_prompt)

# agent = LLMSingleActionAgent(
#    llm_chain=llm_chain,
#    output_parser=main_output_parser,
#    stop=["\nObservation:"],
#    allowed_tools=tool_names
# )

# agent_exec = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)



# # Agent
agent = initialize_agent(
    tools=tools,
    llm=agent_llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=False
)
