from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chainlit as cl
import os
import traceback

SUPPORTED_FILE_TYPES = {"pdf": PyPDFLoader, "txt": TextLoader}

def load_and_process_file(file_path: str) -> str:
    """Processes an uploaded file and extracts text based on format."""
    file_ext = file_path.split(".")[-1].lower()
    
    if file_ext not in SUPPORTED_FILE_TYPES:
        return "Unsupported file format. Please upload a PDF or TXT file."
    
    try:
        loader = SUPPORTED_FILE_TYPES[file_ext](file_path)
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(pages)
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        return f"Error processing file: {str(e)}"

@cl.on_chat_start
async def on_chat_start():
    elements = [cl.Image(name="image1", display="inline", path="chat.png")]
    await cl.Message(content="Hello! I am Chatbot. Ask me anything or upload a file.", elements=elements).send()

    model = ChatGroq(temperature=0, model_name="llama3-70b-8192")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You're a knowledgeable Machine Learning Engineer."),
        ("human", "{question}")
    ])
    cl.user_session.set("runnable", prompt | model | StrOutputParser())

@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable
    msg = cl.Message(content="")
    user_query = message.content
    file_text = ""
    
    try:
        if message.elements:
            for element in message.elements:
                if isinstance(element, cl.File):
                    file_text = load_and_process_file(element.path)
        
        combined_input = f"User Query: {user_query}\n\nFile Content:\n{file_text}" if file_text else user_query
        
        async for chunk in runnable.astream(
            {"question": combined_input},
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ):
            await msg.stream_token(chunk)
    except Exception as e:
        msg.content = f"An error occurred: {traceback.format_exc()}"
        await msg.send()  # ✅ Fixed the issue here
    else:
        await msg.send()
