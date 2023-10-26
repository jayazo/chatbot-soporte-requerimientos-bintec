import chainlit as cl
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import DeepLake
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool, AgentType,ZeroShotAgent,AgentExecutor,create_pandas_dataframe_agent
from dotenv import dotenv_values
from langchain.agents.agent_types import AgentType
import os
from langchain.chains import LLMChain
from chainlit.sync import run_sync
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferWindowMemory,ConversationSummaryMemory,ConversationKGMemory,CombinedMemory
import pandas as pd
from typing import Any, List, Optional
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
# os.environ['REQUESTS_CA_BUNDLE'] = 'C:/Users/armadrid/Documents/LangChain/ROOTBANCOLOMBIACA.crt'
# os.environ["ACTIVELOOP_TOKEN"] = "eyJhbGciOiJIUzUxMiIsImlhdCI6MTY5NTc1NzQ3OSwiZXhwIjoxNzI3Mzc5ODcwfQ.eyJpZCI6ImFzZWJhc3RpYW5tdiJ9.PzuolRBBULlFHjYMSpYz729GH13tZbnc5tziazffx3oI-Jtitvv4dilWksmjJOsUjGp7ISGl7l05N99Yx4scCA"
ENV = dotenv_values(".env")


df_proveedores = pd.read_csv("./data/data_test.csv")

@cl.on_chat_start
async def main():               
    vectorStore = Chroma(persist_directory="./vector_store",
                     embedding_function=HuggingFaceEmbeddings())

    llm = OpenAI(model_name="text-davinci-003",temperature=0,verbose=True, openai_api_key=ENV['OPENAI_API_KEY'])        
    chain = RetrievalQA.from_chain_type(
      llm=llm,
      chain_type="stuff",
      retriever=vectorStore.as_retriever()             
    )
    #pandasPower = create_pandas_dataframe_agent(llm, [proveedores,gasto,Contratos], verbose=True)
    
    #Prueba de memory ---------------------------------------------------
    llm = OpenAI(model_name="text-davinci-003",temperature=0,verbose=True, openai_api_key=ENV['OPENAI_API_KEY'])    
    llm_context = ChatOpenAI(temperature=0.5, model_name="gpt-3.5", openai_api_key=ENV['OPENAI_API_KEY']) #gpt-3.5-turbo
    PREFIX = """
    You are working with {num_dfs} pandas dataframes in Python named df1, df2, etc.    
    You should use the tools below to answer the question posed of you:

    Summary of the whole conversation:
    {chat_history_summary}

    Last few messages between you and user:
    {chat_history_buffer}

    Entities that the conversation is about:
    {chat_history_KG}

    """

    chat_history_buffer = ConversationBufferWindowMemory(
    k=5,
    memory_key="chat_history_buffer",
    input_key="input"
    )

    chat_history_summary = ConversationSummaryMemory(
        llm=llm_context, 
        memory_key="chat_history_summary",
        input_key="input"
        )

    chat_history_KG = ConversationKGMemory(
        llm=llm_context, 
        memory_key="chat_history_KG",
        input_key="input",
        )

    memory = CombinedMemory(memories=[chat_history_buffer, chat_history_summary, chat_history_KG])
   #  pandasPower = create_pandas_dataframe_agent(
   #  llm, 
   #  [df_proveedores],
   #  prefix=PREFIX,    
   #  verbose=True, 
   #  agent_executor_kwargs={"memory": memory},
   #  input_variables=['dfs_head','num_dfs', 'input', 'agent_scratchpad', 'chat_history_buffer', 'chat_history_summary', 'chat_history_KG']
   #  )
    #Fin prueba -----------------------------------------------------------

    tools = [
        Tool(
           name="Desestructurados",
           func= chain.run,
           description="""Usar cuando la pregunta esté relacionada a: 
                     - Conformacion de equipo de apoyo
                     - Conocimiento de proveedores
                     - Solicitar acuerdos de confidencialidad
                     - Solicitar propuesta económica o cotizaciones
                     - Negociaciones
                     - Definiciones contables y material
                     - Formalizacion de contratos o compra
                     - Gestionar prerrequisitos de negociacion
                     - Conocimiento de proveedores
                     - Gestionar renovacion
                     - Actualizar informacion del proveedor
                     """            
        ),
      #   Tool(
      #       name="Estructurados",
      #       func= pandasPower.run,
      #       description="Usar cuando el usuario desee conocer el estado de un requerimiento y que no existan en el ConversationBufferMemory"            
      #   ),
    ]
    agent_chain = initialize_agent(
        tools, llm=llm,agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True,handle_parsing_errors=True
    )
    
    cl.user_session.set("agent", agent_chain)
   
  

@cl.on_message
async def main(message):
   agent = cl.user_session.get("agent")      
   res = await agent.arun(message, callbacks=[cl.AsyncLangchainCallbackHandler()])
   await cl.Message(content=res).send()