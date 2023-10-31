from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.chains import RetrievalQA
from langchain.tools.base import BaseTool
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.prompts import PromptTemplate
from prompt_templates.custom_templates import RETRIEVAL_PROMPT_TEMPLATE

def retrieval_qa_tool(llm, retriever) -> Tool:

   # Crear prompt
   prompt = PromptTemplate(
    template=RETRIEVAL_PROMPT_TEMPLATE, 
    input_variables=["context", "question"]
)

   # Crear retrieval QA Chain
   rag = RetrievalQA.from_chain_type(
      llm=llm,
      chain_type="stuff",
      retriever=retriever,
      chain_type_kwargs={"prompt": prompt},
      verbose=True
      # return_source_documents=True,
   )

   # Crea el tool
   tool = Tool(
      name="Retrieval QA",
      func=rag.run,
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
                     - Gestionar renovación de un contrato
                     """,
      return_direct=True,
      # verbose=True
   )
   return tool

#    return create_retriever_tool(
#        retriever=retriever,
#        name="Retrieval QA",
#        description="""Usar cuando la pregunta esté relacionada a: 
#                      - Conformacion de equipo de apoyo
#                      - Conocimiento de proveedores
#                      - Solicitar acuerdos de confidencialidad
#                      - Solicitar propuesta económica o cotizaciones
#                      - Negociaciones
#                      - Definiciones contables y material
#                      - Formalizacion de contratos o compra
#                      - Gestionar prerrequisitos de negociacion
#                      - Conocimiento de proveedores
#                      - Gestionar renovacion
#                      - Actualizar informacion del proveedor
#                      """
#    )
