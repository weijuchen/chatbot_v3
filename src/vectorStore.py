# create your vector store
import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
import qdrant_client
from langchain_community.document_loaders import PyPDFLoader
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
import openai
# uuid is used to generate unique IDs for the points
import uuid


client = qdrant_client.QdrantClient(
        os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

embeddings = OpenAIEmbeddings()


vectorstore = Qdrant(
    client=client,
    collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
    embeddings=embeddings,
)

# import sys

# print(sys.path)

# load environment variable
load_dotenv()

#  Created function to load multiple PDFs

pdf_folder_path = os.getenv("PDF_FOLDER_PATH")


def load_multiple_pdfs(pdf_folder_path):  # ***
    docs = []
    # 檢查資料夾是否存在
    if not os.path.exists(pdf_folder_path):
        print(f"Error: The directory {pdf_folder_path} does not exist.")
        return []
    for file_name in os.listdir(pdf_folder_path):  # ***
        if file_name.endswith(".pdf"):  # *** Check if file is a PDF
            file_path = os.path.join(
                pdf_folder_path, file_name
            )  # *** Get full path of the file
            loader = PyPDFLoader(file_path)  # *** Load the PDF using PyPDFLoader
            docs.extend(loader.load())  # *** Add loaded documents to docs list
    return docs  # *** Return the full list of documents


# pdf_docs = load_multiple_pdfs(pdf_folder_path)
# print(pdf_docs)  # 這將打印所有 PDF 檔案的路徑


def get_embedding(text_chunks, model_id="text-embedding-ada-002"):
    points = []
    for idx, chunk in enumerate(text_chunks):
        response = openai.Embedding.create(input=chunk, model=model_id)
        embeddings = response["data"][0]["embedding"]
        point_id = str(uuid.uuid4())  # Generate a unique ID for the point
        points.append(
            PointStruct(id=point_id, vector=embeddings, payload={"text": chunk})
        )

    return points

connection = QdrantClient(
    url=os.getenv("QDRANT_HOST"),
)

def insert_data(get_points):
    operation_info = connection.upsert(
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        wait=True,
        points=get_points,
    )


# def get_chunk(docs):
#     # split your docs into texts chunks
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=50,
#         separators=["\n", "。", "！", "？", "，", "、", ""],
#     )
#     chunks = text_splitter.split_documents(docs)
#     return chunks

# load my PDFs


# pdf_folder_path = os.getenv("PDF_FOLDER_PATH")
# docs = load_multiple_pdfs(pdf_folder_path)
# print("type of docs",type(docs))
# texts = get_chunk(docs)
# print("type of texts",type(texts))
# vectorstore.add_texts(texts)

# Create a Qdrant instance
# vectorstore = QdrantVectorStore(
#     client=client,
#     collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
#     embeddings=OpenAIEmbeddings(),
# )
