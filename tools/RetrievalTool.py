from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.chains import RetrievalQA
from langchain.tools.base import BaseTool
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_toolkits import create_retriever_tool


def retrieval_qa_tool(llm, retriever) -> BaseTool:
   
   # Crear retrieval QA Chain
   rag = RetrievalQA.from_chain_type(
      llm=llm,
      chain_type="stuff",
      retriever=retriever,
      # return_source_documents=True
   )

   # Crea el tool
   tool = Tool(
      name="Retrieval QA",
      func=rag.run,
      # description="""Usar cuando la pregunta esté relacionada a: 
      #                - Conformacion de equipo de apoyo
      #                """,
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
                     """,
      
      return_direct=True,
   )
   return tool
