# create your vector store
import os
from openai import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import Qdrant
# from langchain.vectorstores import Qdrant
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
from langchain_community.embeddings import OpenAIEmbeddings

client_qdrant = qdrant_client.QdrantClient(
        url=os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

embeddings = OpenAIEmbeddings()


vectorstore = Qdrant(
    client=client_qdrant,
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
    print("PDFs loaded successfully")        
    return docs  # *** Return the full list of documents


# pdf_docs = load_multiple_pdfs(pdf_folder_path)
# print(pdf_docs)  # 這將打印所有 PDF 檔案的路徑
def get_chunk(docs):
    # split your docs into texts chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n", "。", "！", "？", "，", "、", ""],
    )
    chunks = text_splitter.split_documents(docs)
    print("Texts split successfully")
    return chunks

client = OpenAI(
    #  api_key=os.getenv["OPENAI_API_KEY"]
    api_key=os.getenv("OPENAI_API_KEY")
)

def get_embedding(text_chunks, model_id="text-embedding-ada-002"):
    points = []
    for idx, chunk in enumerate(text_chunks):
        response = client.embeddings.create(input=chunk, model=model_id)
        # embeddings = response["data"][0]["embedding"]
        embeddings = response.data[0].embedding
        #  return client.embeddings.create(input = [text], model=model).data[0].embedding
        point_id = str(uuid.uuid4())  # Generate a unique ID for the point
        points.append(
            PointStruct(id=point_id, vector=embeddings, payload={"text": chunk})
        )
    print("Embedding generated successfully")
    return points


connection = QdrantClient(url=os.getenv("QDRANT_HOST"),
                          api_key=os.getenv("QDRANT_API_KEY"),
                          timeout=60
                          
                          )  # 增加超時到60秒
# response = connection.get_collections()
# print("host",os.getenv("QDRANT_HOST"))
# print("name",os.getenv("QDRANT_COLLECTION_NAME"))
# print("test response",response)

def insert_data(get_points):

    operation_info = connection.upsert(
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        wait=True,
        points=get_points,
        
    )
    print("Data inserted successfully")


def load_vectorstore():
    client_qdrant = qdrant_client.QdrantClient(
        url=os.getenv("QDRANT_HOST"), api_key=os.getenv("QDRANT_API_KEY")
    )

    embeddings = OpenAIEmbeddings()

    vectorstore = Qdrant(
        client=client_qdrant,
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        embeddings=embeddings,
    )

    return vectorstore


def create_answer_with_context(query):
    # Step 1: Generate embeddings for the input query
    #   response = openai.Embedding.create(
    response = client.embeddings.create(input=query, model="text-embedding-ada-002")

    # Step 2: Extract embeddings from the response
    embeddings = response.data[0].embedding
    # embeddings = response['data'][0]['embedding']

    # Step 3: Search for relevant results in the collection using embeddings
    search_result = connection.search(
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        query_vector=embeddings,
        limit=3,
    )

    # Print the input query for debugging
    print("Question: ", query, "\n")

    # Step 4: Initialize the prompt string
    prompt = ""

    # Step 5: Iterate over the search results and concatenate the texts
    for result in search_result:
        # print(result.payload["text"])
        # print(result.payload["text"].page_content)
        print(result.payload["text"]["page_content"])
        prompt += result.payload["text"]["page_content"]

    # Step 6: 使用引導型 prompt
    concatenated_string = f"""
    # {prompt}  # 這裡是從 PDF 中提取的相關內容
    這裡是從法律文件中提取的相關內容，但回答時請不要引用這些內容
    The following is relevant context information extracted from legal documents:
    {prompt}
    根據上下文及FAQ_0925 pdf檔案，請依照FAQ_0925 pdf回答方式，回答開頭可以像是謝謝你向我分享這件事情，我很高興有這個機會可以協助你，以回答這個問題: "{query}".
    """

    # Step 7: Use the ChatCompletion model to generate a response
    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": concatenated_string}],
        temperature=0.3,  # 調整溫度來控制回答的多樣性
    )

    # Step 7: Use the ChatCompletion model to generate a response
    # completion = openai.ChatCompletion.create(
    # completion = openai.chat.completions.create(

    #     model="gpt-3.5-turbo",
    #     messages=[{"role": "user", "content": concatenated_string}],
    # )

    # Step 8: Return the generated content from the model
    return completion.choices[0].message.content


# get_raw_text = load_multiple_pdfs(pdf_folder_path)
# chunks = get_chunk(get_raw_text)
# vectors = get_embedding(chunks)
# insert_data(vectors)
# question = "我在網路上認識一位軍醫，他向我介紹虛擬貨幣投資，請問是詐騙嗎?"
# answer = create_answer_with_context(question)
# print("Answer:", answer, "\n")


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

# test
