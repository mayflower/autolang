import os
import faiss
import readline # for better CLI experience
from typing import List
from langchain import FAISS, InMemoryDocstore
from langchain.agents import Tool, load_tools
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings
from langchain.llms.base import BaseLLM 

from .auto import AutoAgent
from dotenv import load_dotenv

# Load default environment variables (.env)
load_dotenv()

# API Keys
LLAMA_MODEL = os.getenv("LLAMA_MODEL", default="alpaca-13b-ggml-q4_0-lora-merged/ggml-model-q4_0.bin")
assert LLAMA_MODEL, "LLAMA_MODEL environment variable is missing from .env"


llm: BaseLLM = LlamaCpp(model_path=LLAMA_MODEL, temperature=0, echo=True, n_ctx=2048)
embeddings = LlamaCppEmbeddings(model_path=LLAMA_MODEL) # type: ignore
print(embeddings)

objective = input('What is my purpose? ')

"""
Customize the tools the agent uses here. Here are some others you can add:

os.environ["WOLFRAM_ALPHA_APPID"] = "<APPID>"
os.environ["SERPER_API_KEY"] = "<KEY>"

tool_names = ["terminal", "requests", "python_repl", "human", "google-serper", "wolfram-alpha"]
"""

tool_names = ["python_repl", "human", "google-serper", "requests"]

tools: List[Tool] = load_tools(tool_names, llm=llm)  # type: ignore

index = faiss.IndexFlatL2(1536)
docstore = InMemoryDocstore({})
vectorstore = FAISS(embeddings.embed_query, index, docstore, {}) 

agent = AutoAgent.from_llm_and_objectives(llm, objective, tools, vectorstore, verbose=True) 

agent.run()
