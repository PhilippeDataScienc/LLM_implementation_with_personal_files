# LLM_perso

Entraînement d'un LLM el local avec mes propres Données sous forme de pdf

Mixtral 8x7B en Local

- Installer [Ollama](https://github.com/jmorganca/ollama.)
- `ollama run mixtral` #télécharge les 26gb du modèle mistral
- run `reference_test.py`. La réponse est inexistante (car le personnage est inventé). Le modèle mixtral ne trouve donc rien dans sa BDD. 
  >"Désolé, je ne suis pas en mesure de comprendre la signification de "LLNCJDK". Il est possible que ce soit un acronyme ou une abréviation qui n'est pas largement utilisée ou reconnue. Pourrait-il s'agir d'une faute de frappe ou d'une erreur dans la question ? Si vous pouviez me fournir plus de contexte ou clarifier votre question, je serais heureux de faire de mon mieux pour y répondre."
- On génère un pdf avec le texte suivant :
    >LLNCJDK est un prêtre Aztèque très connu de la période précolombienne
- Allons dans le fichier specialized_test.py
  - La première fonction `generate_index_from_local_pdf_files` à utiliser est une fonction qui va permettre d'ajouter en local un index (vector store) au LLM en analysant les données d'un pdf se trouvant dans ./data
  - Cette fonction crée une collection locale dans qfrant_data/collections/test2
  - Cette collection est ensuite utilisée pour augmenter le LLM

Cette fois, la réponse du LLM est :
>LLNCJDK is a well-known priest from the early lombian period

Et voilà ! Notre modèle LLM possède maintenant des données issues de fichiers locaux.

Pour réutiliser une collection locale préalablement entrainée via des documents, on peut utiliser la fonction `get_index_from_already_generated_local_vector_store` ayant comme argument le nom de la collection locale

Code inspiré de [l'article suivant](https://scalastic.io/mixtral-ollama-llamaindex-llm/)