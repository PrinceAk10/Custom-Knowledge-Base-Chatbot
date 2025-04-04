from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import chainlit as cl
from transformers import pipeline # text genration for auto completion
import speech_recognition as sr

# load the text_genration model
# for auto completion

text_generator = pipeline("text-generation", model="gpt2")

def generate_auto_completion(prompt: str) -> str:
    """Generates auto-completion for the given prompt."""
    try:
        # Generate text based on the user's input
        completion = text_generator(prompt, max_length=50, num_return_sequences=1)[0]['generated_text']
        return completion
    except Exception as e:
        return f"Error generating auto-completion: {e}"
    
# speech recognition for user input
    
async def process_voice_input():
    """Handles voice input from the user."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        try:
            await cl.Message(content="Listening...").send()
            audio = r.listen(source)
            user_query = r.recognize_google(audio)
            await cl.Message(content=f"You said: {user_query}").send()

            # Process the voice input as a normal query
            await on_message(cl.Message(content=user_query))
        except sr.UnknownValueError:
            await cl.Message(content="Sorry, I could not understand your speech. Please try again.").send()
        except sr.RequestError:
            await cl.Message(content="There was an issue with the speech recognition service. Please try again later.").send()
    



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

# speech recognition if user want to record the audio
 if user_query.lower() == "record voice":
        await process_voice_input()
        return



# procees the file which is uploaded  by user
    if message.elements:
        for element in message.elements:
            if isinstance(element, cl.File):
                file_text = await process_file(element.path)
    
    combined_input = f"User Query: {user_query}\n\nFile Content:\n{file_text}" if file_text else user_query

# auto completion for user query
if user_query:
    auto_completion= generate_auto_completion(user_query)
    await cl.message(content=f"auto-completion:{auto_completion}").send() 

# streaing the chatbot response 

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
