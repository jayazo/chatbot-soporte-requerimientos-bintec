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

from agents.Agent import agent
ENV = dotenv_values(".env")


# messages = [
#     SystemMessagePromptTemplate.from_template(ct.GENERIC_CHAT_TEMPLATE_ES),
#     HumanMessagePromptTemplate.from_template("{question}")
# ]

# prompt = ChatPromptTemplate.from_messages(messages=messages)
# chain_type_kwargs = {"prompt": prompt}


@cl.on_chat_start
def main():
   cl.user_session.set("agent", agent)


@cl.on_message
async def main(message: str):
   agent = cl.user_session.get("agent")
   res = await agent.arun(message, callbacks=[cl.AsyncLangchainCallbackHandler()])
   await cl.Message(content=res).send()
