from enum import Enum

GENERIC_GENERATIVE_TEMPLATE = """Question: {question}
                     Answer: Let's think step by step."""

GENERIC_CHAT_TEMPLATE = """Use the following pieces of context to answer the users question.
                     If you don't know the answer, just say that you don't know, don't try to make up an answer.
                     ALWAYS return a "SOURCES" part in your answer.
                     The "SOURCES" part should be a reference to the source of the document from which you got your answer.

                     Example of your response should be:

                     ```
                     The answer is foo
                     SOURCES: xyz
                     ```

                     Begin!
                     ----------------
                     {summaries}"""

# GENERIC_CHAT_TEMPLATE_ES = """Utiliza los siguientes elementos de contexto para responder a la pregunta del usuario.
#                      Si no sabes la respuesta, simplemente dile al usuario que no sabes la respuesta, no intentes
#                      inventar una respuesta.

#                      Incluye SIEMPRE una parte de tus "FUENTES" en tu respuesta.
#                      La parte "FUENTES" debe ser una referencia a la fuente del documento del que has obtenido tu respuesta (agrega el nombre completo del documento del que obtuviste la información).

#                      Un ejemplo de respuesta debería ser:

#                      ```
#                      La respuesta es foo
#                      ```

#                      Comienza!
#                      ----------------
#                      {summaries}"""

GENERIC_CHAT_TEMPLATE_ES = """Utiliza los siguientes elementos de contexto para responder a la pregunta del usuario.
                     Si no sabes la respuesta, simplemente dile al usuario que no sabes la respuesta, no intentes
                     inventar una respuesta.

                     Incluye SIEMPRE una parte de tus "FUENTES" en tu respuesta.

                     Un ejemplo de respuesta debería ser:

                     ```
                     La respuesta es foo
                     ```

                     Comienza!
                     ----------------
                     {summaries}"""