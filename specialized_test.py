import qdrant_client
from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.core import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore


def generate_index_from_local_pdf_files() -> VectorStoreIndex:
    """
    Generate a VectorStoreIndex from local PDF files.

    This function performs the following steps:
    1. Loads documents from the './data' directory using the `SimpleDirectoryReader` function.
    2. Initializes a Qdrant vector store with the `QdrantVectorStore` function, using a QdrantClient that points to './qdrant_data'.
    3. Sets up a StorageContext with the initialized vector store.
    4. Initializes a Large Language Model (LLM) with the `Ollama` function, using the 'mixtral' model and a request timeout of 600 seconds.
    5. Sets up a ServiceContext with the initialized LLM and a local embed model.
    6. Creates a VectorStoreIndex from the loaded documents, using the set up ServiceContext and StorageContext.

    Returns:
        VectorStoreIndex: The generated VectorStoreIndex.
    """
    # Loading the documents from the disk
    documents = SimpleDirectoryReader("./data").load_data(show_progress=True)

    # Initializing the vector store with Qdrant
    client = qdrant_client.QdrantClient(path="./qdrant_data")
    vector_store = QdrantVectorStore(client=client, collection_name="test2")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Initializing the Large Language Model (LLM) with Ollama
    # The request_timeout may need to be adjusted depending on the system's performance capabilities
    llm = Ollama(model="mixtral", request_timeout=600.0)
    service_context = ServiceContext.from_defaults(llm=llm, embed_model="local")
    # Creating the index, which includes embedding the documents into the vector store
    index = VectorStoreIndex.from_documents(documents,
                                            show_progress=True,
                                            service_context=service_context,
                                            storage_context=storage_context)
    return index


def get_index_from_already_generated_local_vector_store(vector: str) -> VectorStoreIndex:
    """
    Retrieve a VectorStoreIndex from an already generated local vector store.

    This function performs the following steps:
    1. Initializes a Large Language Model (LLM) with the `Ollama` function, using the 'mixtral' model and a request timeout of 600 seconds.
    2. Sets up a ServiceContext with the initialized LLM and a local embed model.
    3. Initializes a Qdrant vector store with the `QdrantVectorStore` function, using a QdrantClient that points to './qdrant_data' and the provided vector store collection name.
    4. Creates a VectorStoreIndex from the initialized vector store and ServiceContext.

    Args:
        vector (str): The name of the vector store collection.

    Returns:
        VectorStoreIndex: The generated VectorStoreIndex.
    """

    llm = Ollama(model="mixtral", request_timeout=600.0)
    service_context = ServiceContext.from_defaults(llm=llm, embed_model="local")
    client = qdrant_client.QdrantClient(path="./qdrant_data")
    vector_store = QdrantVectorStore(client=client, collection_name=vector)
    index = VectorStoreIndex.from_vector_store(embed_model="local", vector_store=vector_store,
                                               service_context=service_context)
    return index


if __name__ == "__main__":
    # Choose one of the following depending on whether you want to generate a new index or retrieve an existing one
    # index = generate_index_from_local_pdf_files()
    index = get_index_from_already_generated_local_vector_store("test2")
    # Querying the index with a specific question
    query_engine = index.as_query_engine()
    prompt = (
        "Qui est LLNCJDK ?"
    )
    response = query_engine.query(prompt)
    print(response)
