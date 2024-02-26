from llama_index.llms.ollama import Ollama

llm = Ollama(model="mixtral", request_timeout=600.0)

prompt = (
  "Crée une classe de contrôleur REST en Java pour une application Spring Boot 3.2. "
  "Cette classe doit gérer des requêtes GET et POST, et inclure des annotations "
  "de sécurité et de configuration."
)

response = llm.complete(prompt)
print(response)