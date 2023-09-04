from fastapi import APIRouter,File,UploadFile
from fastapi.responses import JSONResponse
import tempfile
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import  Qdrant
import qdrant_client
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import (ConversationalRetrievalChain)
from langchain.memory import RedisChatMessageHistory
from langchain.prompts import PromptTemplate,ChatPromptTemplate
from pydantic import BaseModel
import os
router = APIRouter()



QDRANT_API_KEY=os.environ.get("QDRANT_API_KEY")
OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")

if(QDRANT_API_KEY is None or OPENAI_API_KEY is None):
    raise Exception("Please set QDRANT_API_KEY and OPENAI_API_KEY environment variables")

@router.post('/embeddings')
async def upload_file(file:UploadFile):
    if not file:
        return JSONResponse(status_code=400, content={"message": "No file uploaded"})
    pdf_path=""
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file.file.read())
        pdf_path=temp_file.name
    if pdf_path=="":
        return JSONResponse(status_code=400, content={"message": "No file uploaded"})
    raw_text = ''
    with open(pdf_path, "rb") as pdf_file:
        reader = PdfReader(pdf_file)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text
    # So we have till here the raw text of the pdf
    # We need to split the text that we read into smaller chunks so that during information retreival we don't hit the token size limits.
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)

    # Download embeddings from OpenAI
    embeddings = OpenAIEmbeddings(client=OPENAI_API_KEY)
    #Load QDrant vector store 
    Qdrant.from_texts(
    texts=texts,
    embedding=embeddings,
    url="https://15fd304c-3b57-4c69-820e-c7d100b2cdef.eu-central-1-0.aws.cloud.qdrant.io:6333",
    prefer_grpc=True,
    api_key=os.environ.get("QDRANT_API_KEY"),
    collection_name="JL",
    force_recreate=True,
    )
    return {"message":"success"}


class Chat(BaseModel):
    question:str
    collection_name:str | None =None
    session_id:str | None =None 

class ChatResponse(BaseModel):
    response:str
    status_code:int

@router.post("/chat")
def chat(body:Chat) -> ChatResponse:
    # Download embeddings from OpenAI
    message_history = RedisChatMessageHistory(body.question if body.session_id is None else body.session_id)
    embeddings = OpenAIEmbeddings(client=OPENAI_API_KEY)
    client=qdrant_client.QdrantClient(
        "https://15fd304c-3b57-4c69-820e-c7d100b2cdef.eu-central-1-0.aws.cloud.qdrant.io:6333",
        api_key=os.environ.get("QDRANT_API_KEY")
    )
    doc_store=Qdrant(
        client=client,
        collection_name= "JL" if body.collection_name is None else body.collection_name,
        embeddings=embeddings,
    )


    qa=ConversationalRetrievalChain.from_llm(
        OpenAI(temperature=0),
        doc_store.as_retriever()
        )

    result=qa({"question":body.question,"chat_history":message_history.messages,})

    message_history.add_user_message(result["question"])
    message_history.add_ai_message(result["answer"])
    return ChatResponse(response=result["answer"], status_code=200)


@router.get('/session/{session_id}')
async def get_session(session_id):
    message_history = RedisChatMessageHistory(session_id=session_id)
    formatted_data = []
    # Iterate through the input data and create question-answer pairs
    for i in range(0, len(message_history.messages), 2):
        question = message_history.messages[i].content
        answer = message_history.messages[i + 1].content
        formatted_data.append({"question": question, "answer": answer})

    # Now, 'formatted_data' contains the question-answer pairs
    return {
        "session_id":session_id,
        "messages":formatted_data
    }

