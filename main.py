import chainlit as cl
from langchain.llms import OpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from dotenv import dotenv_values
from prompt_templates import custom_templates as ct

from agents.Agent import agent_exec as agent
# from agents.Agent import agent

ENV = dotenv_values(".env")

@cl.on_chat_start
def main():
   cl.user_session.set("agent", agent)

@cl.on_message
async def main(message: str):
   agent = cl.user_session.get("agent")
   res = await agent.arun(input=message, callbacks=[cl.AsyncLangchainCallbackHandler()], return_only_outputs=False)
   # res = await agent(inputs={"input":message}, callbacks=[cl.AsyncLangchainCallbackHandler()], return_only_outputs=False)
   print(res)
   print(type(res))
   await cl.Message(content=res).send()
