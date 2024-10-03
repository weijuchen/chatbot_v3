# prepare to use Qudrant 


import os
import sys
import openai
from openai import OpenAI


from dotenv import load_dotenv
from flask import Flask, request, abort
from langchain_openai import ChatOpenAI

from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, AudioMessage, TextMessage, TextSendMessage

# 使用pdf.py 中的get_qa_chain函數
# from src.pdf import create_vector, load_vectorstore
from src.pdf import load_vectorstore

# from src.pdf import create_vector

# load environment variable
load_dotenv()


# get channel_access_token from my environment variable
channel_access_token = os.getenv("LINE_ACCESS_TOKEN")
channel_secret = os.getenv("LINE_SECRET")

if channel_secret is None:
    print("Specify LINE_CHANNEL_SECRET as environment variable.")
    sys.exit(1)
if channel_access_token is None:
    print("Specify LINE_CHANNEL_ACCESS_TOKEN as environment variable.")
    sys.exit(1)


# create a Flask app

app = Flask(__name__)
line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)
parser = WebhookHandler(channel_secret)
# create a route for webhook

openai_api_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=openai_api_key,
    temperature=0,
    # streamling=True,
    # streamling means that the model response will be generated in a streaming fashion
)
# vectorstore = create_vector()

# 確保 FAISS 向量存儲目錄存在
vectorstore_path = os.getenv("FAISS_VECTORSTORE_PATH")

if vectorstore_path:
    os.makedirs(vectorstore_path, exist_ok=True)
else:
    print("FAISS_VECTORSTORE_PATH is not set.")

vectorstore = load_vectorstore()

embeddings = OpenAIEmbeddings()
# vectorstore = FAISS.load_local("faiss_midjourney_docs", embeddings)
# vectorstore.save_local("faiss_midjourney_docs")

# Load the FAISS Vector Store with Dangerous Deserialization Enabled
try:
    retriever = FAISS.load_local(
        "faiss_vectorstore", embeddings, allow_dangerous_deserialization=True
    ).as_retriever(search_type="similarity", search_kwargs={"k": 1})
except Exception as e:
    print(f"Error loading vectorstore: {e}")


memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    inupt_key="question",
    output_key="answer",
)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=model,
    memory=memory,
    retriever=retriever,
    verbose=True,
)


@app.route("/callback", methods=["POST"])
def callback():
    # 獲取 LINE 的簽名
    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)

    app.logger.info("Request body: " + body)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    question = event.message.text.strip()

    if question.startswith("/清除") or question.lower().startswith("/clear"):
        memory.clear()
        answer = "歷史訊息清除成功"
    elif (
        question.startswith("/教學")
        or question.startswith("/指令")
        or question.startswith("/說明")
        or question.startswith("/操作說明")
        or question.lower().startswith("/instruction")
        or question.lower().startswith("/help")
    ):
        answer = "指令：\n/清除 or /clear\n👉 當 Bot 開始鬼打牆，可清除歷史訊息來重置"
    else:

        # logger.info(f"question language: {question_lang}")

        # get answer from qa_chain
        response = qa_chain({"question": question})
        answer = response["answer"]

    # reply message
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=answer))


@handler.add(MessageEvent, message=AudioMessage)
# 處理接收到的音訊訊息
def handle_AudioMessage(event):

    if event.message.type == "audio":
        audio_content = line_bot_api.get_message_content(event.message.id)
        path = "./temp.m4a"

        with open(path, "wb") as fd:
            for chunk in audio_content.iter_content():
                fd.write(chunk)
        # 載入音檔 呼叫openai api 模型 並進行語音轉換文字

        if os.path.exists(path):

            try:
                # audio_file = open("/Users/your_username/Documents/temp
                # .mp3", "rb")
                with open("./temp.m4a", "rb") as audio_file:
                    openai_api_key = os.getenv("OPENAI_API_KEY")
                    model_id = "whisper-1"
                    client = OpenAI()
                    response = client.audio.transcriptions.create(
                        model=model_id, file=audio_file
                    )

                    # get answer from qa_chain
                    if response and hasattr(response, "text"):
                        # print("here is the response:", response.text)

                        # 根據轉換的文字進行 Q&A 處理
                        response_qa = qa_chain({"question": response.text})
                        answer = response_qa["answer"]

                        # 回應用戶
                        line_bot_api.reply_message(
                            event.reply_token, TextSendMessage(text=answer)
                        )
                    else:
                        # 如果 response 不包含 text，給出錯誤提示
                        line_bot_api.reply_message(
                            event.reply_token,
                            TextSendMessage(text="無法取得語音轉文字的結果"),
                        )

            except openai.BadRequestError as e:
                # except openai.error.OpenAIError as e:
                print(f"OpenAI API request failed: {e}")
                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text="語音轉換文字失敗，請稍後再試。"),
                )


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
    print("The server has started successfully")
