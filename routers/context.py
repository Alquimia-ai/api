from fastapi import APIRouter,File,UploadFile
from fastapi.responses import JSONResponse
from typing import List
import tempfile
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import  Qdrant
import qdrant_client
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import (LLMChain,ConversationalRetrievalChain)
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import RedisChatMessageHistory
from langchain.prompts import PromptTemplate
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
    docs=doc_store.similarity_search(body.question,4)
    context_docs = ""
    for doc in docs:
        context_docs = context_docs + doc.page_content + " \n\n "
        
    template = """ your template
    Context:\"""
    
    {context}
    \"""
    Question:\"
    \"""
    
    Helpful Answer:"""
    prompt= PromptTemplate.from_template(template)
    streaming_llm = OpenAI(
    streaming=True,
    openai_api_key=OPENAI_API_KEY,
    callback_manager=BaseCallbackManager([
        StreamingStdOutCallbackHandler()
    ]),
    verbose=True,
    max_tokens=150,
    temperature=0.2
    )
    # use the streaming LLM to create a question answering chain
    doc_chain = load_qa_chain(
        llm=streaming_llm,
        chain_type="stuff",
        prompt=prompt
    )
    question_generator=LLMChain(
        llm=streaming_llm,
        prompt=prompt
    )
    
    chatbot = ConversationalRetrievalChain(
        retriever=doc_store.as_retriever(),
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
    )

    response = chatbot(
        {"question": body.question, "chat_history": message_history.messages}
    )
    message_history.add_user_message(response["question"])
    message_history.add_ai_message(response["answer"])
    return ChatResponse(response=response["answer"], status_code=200)

