from langchain.tools import BaseTool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from typing import Union
from langchain_ollama.llms import OllamaLLM


model = OllamaLLM(model="qwen-concise:latest")


class SquareTool(BaseTool):
    name: str = "Square tool"
    description: str = "use this tool when you need to square a number"

    def _run(self, a: Union[int, float]):
        return float(a) * float(a)

    def _arun(self, a: Union[int, float]):
        raise NotImplementedError("This tool does not support async")


class ChangeSignTool(BaseTool):
    name: str = "change sign tool"
    description: str = "use this tool to multiply a number by minus 1"

    def _run(self, a: Union[int, float]):
        return -float(a)

    def _arun(self, a: Union[int, float]):
        raise NotImplementedError("This tool does not support async")


class MultiplyTool(BaseTool):
    name: str = "multiply tool"
    description: str = "Use this tool when you need to multiply two numbers. The input to this tool should be a string with the first number, a comma, and then the second number."

    def _run(self, string):
        a, b = string.split(',')
        return float(a) * float(b)

    def _arun(self, string):
        raise NotImplementedError("This tool does not support async")


class ModuloTool(BaseTool):
    name: str = "modulo tool"
    description: str = "Use this tool when you need to compute the modulus of two numbers. The input to this tool should be a string with the first number, a comma, and then the second number."

    def _run(self, string):
        a, b = string.split(',')
        return float(a) % float(b)

    def _arun(self, string):
        raise NotImplementedError("This tool does not support async")


conversational_memory = ConversationBufferWindowMemory(memory_key='chat_history', k=5, return_messages=True)

agent = initialize_agent(tools=[SquareTool(), ChangeSignTool(), MultiplyTool(), ModuloTool()], llm=model, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, memory=conversational_memory, verbose=True)

agent.invoke({'input': 'What is the square of the product of two and three?'})['output']
