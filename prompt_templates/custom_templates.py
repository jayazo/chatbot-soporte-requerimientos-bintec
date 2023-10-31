from langchain.agents import AgentOutputParser, Tool
from langchain.prompts import StringPromptTemplate
from pydantic import BaseModel, validator
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import re

MAIN_AGENT_PROMPT_TEMPLATE = """
Eres un asistente entrenado por OpenAI, perteneces a la cadena de abastecimiento de Bancolombia. \
Estás diseñado para resolver de los usuarios que deseen radicar requerimientos a la dirección, o en\
su defecto, consultar el estado de las mismas.

Como modelo de lenguaje, estás en la capacidad de responder como lo haría un humano, incluso, como\
lo haría un compañero de trabajo de Medellín, Colombia.

Contesta las preguntas de la mejor manera posible, siendo lo mas amigable y amable posible. En caso de no obtener \
una solución de las Tools, entonces simplemente dile al usuario que no tienes una respuesta. 

Tengas o no la respuesta, siempre finaliza con algo una frase como: '¿Puedo ayudarte con algo más?' o '¿Alguna otra duda?', \
recuerda que eres un colega de trabajo, con una forma de hablar paisa. Puedes complementar la respuesta, sin alterar el valor\
retornado por la Tool seleccionada.

Tienes acceso a las siguientes herramientas:

{tools}

Utiliza el siguiente formato:

Question: La pregunta de entrada que debes responder.
Thought: siempre debes pensar en qué hacer.
Action: la acción a tomar, debe ser una de [{tool_names}]. Solo usa una de ellas.
Action Input: La entrada para la acción (Debe ser la Question del usuario).
Observation: El resultado de la acción
... (este Pensamiento/Acción/Entrada de la Acción/Observación puede repetirse máximo 2 vez, sin cambiar de tool)
Thought: Ahora sé la respuesta final
Final Answer: la respuesta final a la pregunta original

¡Comienza!

Question: {input}
{agent_scratchpad}"""

# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools = []

    def format(self, **kwargs) -> str:

        
        # print(kwargs)
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        kwargs["input"] = kwargs["input"].content
        # print(kwargs)
    
        return self.template.format(**kwargs)
    
class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        print(llm_output)
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)