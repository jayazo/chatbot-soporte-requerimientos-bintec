from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.chains import RetrievalQA
from langchain.tools.base import BaseTool
from langchain.agents import initialize_agent, Tool

def retrieval_qa_tool(llm, retriever) -> BaseTool:
   
   # Crear retrieval QA Chain
   rag = RetrievalQA.from_chain_type(
      llm=llm,
      chain_type="stuff",
      retriever=retriever
   )

   # Crea el tool
   tool = Tool(
      name="Retrieval QA",
      func=rag.run,
      description="",
      return_direct=True 
   )

   return tool