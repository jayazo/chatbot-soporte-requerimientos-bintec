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


ENV = dotenv_values(".env")


messages = [
    SystemMessagePromptTemplate.from_template(ct.GENERIC_CHAT_TEMPLATE_ES),
    HumanMessagePromptTemplate.from_template("{question}")
]

prompt = ChatPromptTemplate.from_messages(messages=messages)
chain_type_kwargs = {"prompt": prompt}


@cl.on_chat_start
def main():
   docs_search = Chroma(persist_directory="./vector_store",
                        embedding_function=HuggingFaceEmbeddings())

   chain = RetrievalQAWithSourcesChain.from_chain_type(
       llm=OpenAI(temperature=0, openai_api_key=ENV['OPENAI_API_KEY'],
                  streaming=True, model_name="text-davinci-003", verbose=True),
       chain_type="stuff",
       retriever=docs_search.as_retriever(),
       return_source_documents=True
   )
   cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: str):
   chain = cl.user_session.get("chain")

   callbacks = cl.AsyncLangchainCallbackHandler(
       stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
   )

   callbacks.answer_reached = True

   # Line code w the error
   res = await chain.acall(message, callbacks=[callbacks])

   answer = res["answer"]
   sources = res["source_documents"]

   source_elements = []

   # Add the sources to the message
   if sources:
      found_sources = []

      # Add the sources to the message
      for source in sources:
         source_name = source.metadata["source"]
         page_number = source.metadata["page"]
         page_content = source.page_content

         found_sources.append(source_name)
         source_elements.append(cl.Text(
             content=f"Pagina: {page_number+1}. \nContenido: {page_content}", name=source_name))

      if found_sources:
         answer += "\nFuentes:\n" + "\n".join(found_sources)
      else:
         answer += "\n No se encontraron fuentes de información."

      print(answer)

   if callbacks.has_streamed_final_answer:
      # callbacks.final_stream.elements.content = answer
      callbacks.final_stream.elements = source_elements
      await callbacks.final_stream.update()
   else:
      print("Respuesta:"+answer)
      await cl.Message(content=answer, elements=source_elements).send()
