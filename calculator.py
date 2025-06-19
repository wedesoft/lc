import math
import operator
import functools
from langchain.tools import BaseTool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from typing import Union
from langchain_ollama.llms import OllamaLLM


model = OllamaLLM(model="qwen-concise:latest")


class OperatorTool(BaseTool):
    name: str = "operator tool"
    description: str = "Use this tool when you need to perform a calculation." \
                       "The input to this tool should be a Pstring with a Python expression to be evaluated."

    def _run(self, string):
        return eval(string)

    def _arun(self, string):
        raise NotImplementedError("This tool does not support async")


conversational_memory = \
        ConversationBufferWindowMemory(memory_key='chat_history', k=5,
                                       return_messages=True)

agent_type = AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
agent = initialize_agent(tools=[OperatorTool()],
                         llm=model,
                         agent=agent_type,
                         memory=conversational_memory,
                         verbose=True)

try:
    while True:
        result = agent.invoke({'input': input('> ')})
        print(result['output'])
except EOFError:
    pass
