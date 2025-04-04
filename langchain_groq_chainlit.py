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
import pandas as pd  # File processing
from transformers import pipeline  # Text generation for auto-completion
import speech_recognition as sr
from docx import Document

# Load the text generation model for auto-completion
text_generator = pipeline("text-generation", model="gpt2")

def generate_auto_completion(prompt: str) -> str:
    """Generates auto-completion for the given prompt."""
    try:
        completion = text_generator(prompt, max_length=50, num_return_sequences=1)[0]['generated_text']
        return completion
    except Exception as e:
        return f"Error generating auto-completion: {e}"

# Load the emotion detection model
emotion_detector = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

def analyse_emotion(text: str) -> tuple:
    """Analyzes the emotion of the given text."""
    try:
        emotions = emotion_detector(text)[0]
        dominant_emotion = max(emotions, key=lambda x: x['score'])
        return dominant_emotion['label'], dominant_emotion['score']
    except Exception as e:
        return "neutral", 0.0  # Default to neutral if there's an error

# func to process file (csv)
async def process_csv(file_path: str) -> str:
    """Processes a CSV file and extracts its content."""
    try:
        df = pd.read_csv(file_path)
        return df.to_string(index=False)  # Convert the DataFrame to a string
    except Exception as e:
        return f"Error processing CSV file: {e}"

# func to process file (excel)
async def process_excel(file_path: str) -> str:
    """Processes an Excel file and extracts its content."""
    try:
        df = pd.read_excel(file_path)
        return df.to_string(index=False)  # Convert the DataFrame to a string
    except Exception as e:
        return f"Error processing Excel file: {e}"

# func to process file (word)
async def process_word(file_path: str) -> str:
    """Processes a Word document and extracts its content."""
    try:
        doc = Document(file_path)
        if not doc.paragraphs:
            return "The Word document is empty."
        content = "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
        return content if content else "The Word document contains no readable text."
    except Exception as e:
        return f"Error processing Word file: {e}"

# func to process file based on the extension

async def process_file(file_path: str) -> str:
    """Processes an uploaded file and extracts its content."""
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(pages)
        return "\n\n".join([doc.page_content for doc in docs])
    elif file_extension == ".csv":
        return await process_csv(file_path)
    elif file_extension in [".xls", ".xlsx"]:
        return await process_excel(file_path)
    elif file_extension in [".doc", ".docx"]:
        return await process_word(file_path)
    else:
        return "Unsupported file type. Please upload a PDF, CSV, Excel, or Word (.docx) file."

# Speech recognition for user input
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
    elements = [cl.Image(name="image1", display="inline", path="chat.png")]
    await cl.Message(content="Hello! I am Chatbot. Ask me anything or upload a file.", elements=elements).send()

    model = ChatGroq(temperature=0, model_name="llama3-70b-8192")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You're a knowledgeable Machine Learning Engineer."),
        ("human", "{question}")
    ])
    cl.user_session.set("runnable", prompt | model | StrOutputParser())

@cl.on_chat_start
async def on_chat_start():
    # Sending an image with the local file path
    elements = [
        cl.Image(name="image1", display="inline", path="chat.png")
    ]
    await cl.Message(content="Hello there, I am Chatbot. You can ask me anything, upload a file for analysis, or click the button below to use voice input.", elements=elements).send()

    # Add a button for speech recognition
    await cl.Message(content="Click the button below to start voice input:").send()
    await cl.Action(name="Start Voice Input", value="record_voice").send()

    model = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You're a very knowledgeable Machine Learning Engineer."),
        ("human", "{question}")
    ])
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)

@cl.on_message
async def on_message(message: cl.Message):
    user_query = message.content

    # Check if the user clicked the "Start Voice Input" button
    if message.action == "record_voice":
        await process_voice_input()
        return

    # Speech recognition if the user types "record voice"
    if user_query.lower() == "record voice":
        await process_voice_input()
        return

    # Process uploaded files
    if message.elements:
        for element in message.elements:
            if isinstance(element, cl.File):
                file_text = await process_file(element.path)

    combined_input = f"User Query: {user_query}\n\nFile Content:\n{file_text}" if file_text else user_query

    # Auto-completion for user query
    if user_query:
        auto_completion = generate_auto_completion(user_query)
        await cl.Message(content=f"Auto-completion: {auto_completion}").send()

    # Emotion detection for user query
    if user_query:
        emotion, emotion_score = analyse_emotion(user_query)
        await cl.Message(content=f"Emotion: {emotion} (Score: {emotion_score:.2f})").send()

    # Streaming the chatbot response
    runnable = cl.user_session.get("runnable")  # type: Runnable
    async for chunk in runnable.astream(
        {"question": combined_input},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await cl.Message(content=chunk).send()

    # Streaming chatbot response
    try:
        async for chunk in runnable.astream(
            {"question": combined_input},
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ):
            await msg.stream_token(chunk)
    except Exception as e:
        msg.content = f"An error occurred: {traceback.format_exc()}"
        await msg.send()
    else:
        await msg.send()
