# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

from langchain_openai import OpenAIEmbeddings

from langchain.chains import RetrievalQA, create_retrieval_chain

# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI

# from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import ConversationalRetrievalChain
import os
from dotenv import load_dotenv

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

def get_chunk(docs):
    # split your docs into texts chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n", "。", "！", "？", "，", "、", ""],
    )
    chunks = text_splitter.split_documents(docs)
    return chunks

    # load my PDFs
pdf_folder_path = os.getenv("PDF_FOLDER_PATH")
docs = load_multiple_pdfs(pdf_folder_path)

texts=get_chunk(docs)
vectorstore.add_texts(texts)

openai_api_key = os.getenv("OPENAI_API_KEY")
def create_vector():
    # load your PDFs
    pdf_folder_path = os.getenv("PDF_FOLDER_PATH")
    docs = load_multiple_pdfs(pdf_folder_path)



    # embed the chunks into vectorstore (FAISS)
    embeddings = OpenAIEmbeddings()
    if not embeddings:
        print("Error: No embeddings generated. Check the input data.")
    else:
        try:
            vectorstore = FAISS.from_documents(texts, embeddings)
            # save the vectorstore to disk
            vectorstore.save_local(os.getenv("FAISS_VECTORSTORE_PATH"))
            print("vectorstore created")
            return vectorstore
        except Exception as e:
            print(f"Error creating vectorstore: {e}")


# 第二個函數：直接加載已經存好的向量資料庫  我這樣做是為了閃過上面要載入pdf的步驟

# def load_vectorstore(vectorstore_path="faiss_midjourney_docs"):
def load_vectorstore(vectorstore_path=os.getenv("FAISS_VECTORSTORE_PATH")):
    try:
        vectorstore = FAISS.load_local(vectorstore_path, OpenAIEmbeddings(),allow_dangerous_deserialization=True)
        print(f"Vectorstore loaded from: {vectorstore_path}")
        return vectorstore
    except Exception as e:
        print(f"Error loading vectorstore: {e}")
        print("None")
        return None

# load_vectorstore()


if __name__ == "__main__":
    vectorstore = load_vectorstore()
    if vectorstore is not None:
        print("Vectorstore loaded successfully.")
    else:
        print("Failed to load vectorstore.")
