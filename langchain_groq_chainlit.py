from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import chainlit as cl

@cl.on_chat_start
async def on_chat_start():
    # Sending an image with the local file path
    elements = [
        cl.Image(name="image1", display="inline", path="chat.png")
    ]
    await cl.Message(content="Hello there, I am Chatbot. You can ask me anything or upload a file for analysis.", elements=elements).send()

    model = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You're a very knowledgeable Machine Learning Engineer."),
        ("human", "{question}")
    ])
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)

@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable
    msg = cl.Message(content="")

    user_query = message.content
    file_text = ""
    
    if message.elements:
        for element in message.elements:
            if isinstance(element, cl.File):
                file_text = await process_file(element.path)
    
    combined_input = f"User Query: {user_query}\n\nFile Content:\n{file_text}" if file_text else user_query
    
    async for chunk in runnable.astream(
        {"question": combined_input},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()

async def process_file(file_path: str) -> str:
    """Processes an uploaded PDF file and extracts text."""
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(pages)
    extracted_text = "\n\n".join([doc.page_content for doc in docs])
    return extracted_text
