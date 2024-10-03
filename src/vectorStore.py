# create your vector store
import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
import qdrant_client

from langchain_qdrant import QdrantVectorStore


client = qdrant_client.QdrantClient(
        os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

embeddings = OpenAIEmbeddings()

embeddings = OpenAIEmbeddings()

vectorstore = Qdrant(
    client=client,
    collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
    embeddings=embeddings,
)


# Create a Qdrant instance
# vectorstore = QdrantVectorStore(
#     client=client,
#     collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
#     embeddings=OpenAIEmbeddings(),
# )
