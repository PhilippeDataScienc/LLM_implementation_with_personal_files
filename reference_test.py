from llama_index.llms.ollama import Ollama

llm = Ollama(model="mixtral", request_timeout=600.0)

prompt = (
  "Qui est LLNCJDK ?"
)

response = llm.complete(prompt)
print(response)