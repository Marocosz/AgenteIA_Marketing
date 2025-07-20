from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rich.console import Console
from rich.markdown import Markdown 
import os
import getpass

console = Console()
load_dotenv()

# https://lmarena.ai/leaderboard

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

id_model = "llama3-70b-8192"

llm = ChatGroq(
    model=id_model,
    temperature=0.7,  # Controla a aleatoriedade da saída (qual deterministicas ou criativas são, maior mais criativa, menor mais focadas)
    max_tokens=None,  # Sem limite de tokens
    timeout= None,  # Sem tempo máximo para rsposta
    max_retries=2  # Número de tentativas em caso de falha
)

prompt = "Ola! Quem é você?"

def llm_generate(llm, prompt):
    template = ChatPromptTemplate.from_messages([
        ("system", "Você é um redator profissional"),
        ("human", "{prompt}")
    ])

    chain = template | llm | StrOutputParser()

    result = chain.invoke({"prompt": prompt})
    show_result(result)

def show_result(result):
    print(result)
    
