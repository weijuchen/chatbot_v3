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
# pdf_folder_path=r"D:\Computer Science\AI\LLM model\chatbot_v3\sourcedata"
pdf_folder_path = os.getenv("PDF_FOLDER_PATH")
# pdf_folder_path = "/chat_bot/06afterclean"
# pdf_folder_path = "06afterclean"  # 指向 PDF 資料夾
# pdf_folder_path = r"D:\Computer Science\AI\LLM model\chat_bot\06afterclean"
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
pdf_docs = load_multiple_pdfs(pdf_folder_path)
# print(pdf_docs)  # 這將打印所有 PDF 檔案的路徑


openai_api_key = os.getenv("OPENAI_API_KEY")
def create_vector():
    # pdf_folder_path = "/chat_bot/06afterclean"
    # pdf_folder_path=r"D:\Computer Science\AI\LLM model\chat_bot\06afterclean"
    # pdf_folder_path = r"D:\Computer Science\AI\LLM model\chatbot_v3\sourcedata"
    # pdf_folder_path=r"D:\Computer Science\AI\LLM model\github_chat_bot\chat_bot\06afterclean"

    # load your PDFs
    pdf_folder_path = os.getenv("PDF_FOLDER_PATH")
    docs = load_multiple_pdfs(pdf_folder_path)

    # split your docs into texts chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n", "。", "！", "？", "，", "、", ""],
    )
    texts = text_splitter.split_documents(docs)

    # embed the chunks into vectorstore (FAISS) 使用 OpenAI 嵌入將文本塊轉換為向量
    # 步驟 1：創建一個 OpenAIEmbeddings 的實例，該實例將負責將文本塊轉換為數值向量（嵌入），這是使用 OpenAI 模型生成的嵌入。

    embeddings = OpenAIEmbeddings()
    if not embeddings:
        print("Error: No embeddings generated. Check the input data.")
    else:
        # 步驟 2：調用 FAISS.from_documents() 方法，將拆分的文本塊 (texts) 與它們的嵌入一起存儲到 FAISS 向量存儲庫中。FAISS 是一種用來進行高效向量檢索的工具。
        vectorstore = FAISS.from_documents(texts, embeddings)

        # save the vectorstore to disk
        # The FAISS vector store is saved locally to the file "faiss_midjourney_docs". This allows you to reload the vector store later without recomputing the embeddings.

        # vectorstore.save_local("faiss_midjourney_docs")
        vectorstore.save_local(os.getenv("FAISS_VECTORSTORE_PATH"))
        # vectorstore.save_local("../faiss_vectorstore")

        print("vectorstore created")
        return vectorstore


# 第二個函數：直接加載已經存好的向量資料庫
# def load_vectorstore(vectorstore_path="faiss_midjourney_docs"):

#     try:
#         vectorstore = FAISS.load_local(vectorstore_path, OpenAIEmbeddings(),allow_dangerous_deserialization=True)
#         print(f"Vectorstore loaded from: {vectorstore_path}")
#         return vectorstore
#     except Exception as e:
#         print(f"Error loading vectorstore: {e}")
#         print("None")
#         return None

# create_vector()
