# --- Environment Setup ---
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY")

if not GROQ_API_KEY and not OPENAI_API_KEY and not LANGCHAIN_API_KEY:
    raise ValueError("Please set at least one of GROQ_API_KEY, OPENAI_API_KEY, or LANGCHAIN_API_KEY in your .env file.")

# --- Imports ---
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnable import Runnable
from langchain_core.runnable.config import RunnableConfig
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chainlit as cl
import traceback
import pandas as pd     # File processing
from transformers import pipeline       # Text generation for auto-completion
import speech_recognition as sr
from docx import Document
import logging          # to audit the user activity and for effective debugging
import asyncio          # for handling  voice recognition


# configuring the logging
logging.basicConfig(
     level=logging.INFO,
     format="%(asctime)s - %(levelname)s - %(message)s"
)

# chainlit config for handle large file
os.environ["CHAINLIT_MAX_UPLOAD_SIZE"] = "50"
os.environ["CHAINLIT_MAX_MESSAGE_SIZE"] = "60"

# Load the text generation model for auto-completion
text_generator = pipeline("text-generation", model="gpt2")
emotion_detector=pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

def generate_auto_completion(prompt: str) -> str:
    try:
        completion = text_generator(prompt, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)[0]['generated_text']
        return completion[len(prompt):].strip()
    except Exception as e:
        logging.error(f"Error in generating auto-completion: {e}")
        return f"Error generating auto-completion: {e}"


# Load the emotion detection model

def analyse_emotion(text: str) -> tuple:
    try:
        emotions = emotion_detector(text)[0]
        dominant_emotion = max(emotions, key=lambda x: x['score'])
        return dominant_emotion['label'], dominant_emotion['score']
    except Exception as e:
        logging.warning(f" Error in emotion analysis: {e}")
        return "neutral", 0.0  # Default to neutral if there's an error
    

async def process_csv(file_path: str) -> str:
    try:
        df = pd.read_csv(file_path)
        return df.to_string(index=False)
    except Exception as e:
        logging.error(f"Error in processing CSV file: {e}")
        return f"Error processing CSV file: {e}"

async def process_excel(file_path: str) -> str:
    try:
        df = pd.read_excel(file_path)
        return df.to_string(index=False)
    except Exception as e:
        logging.error(f"Error in processing Excel file: {e}")
        return f"Error processing Excel file: {e}"
    
# func to process file (word)

async def process_word(file_path: str) -> str:
    try:
        doc = Document(file_path)
        content="\n".join([p.text for p in doc.paragraphs if p.text])
        return content if content else  "The Word document contains no readable text."
    except Exception as e:
        logging.error(f"Error in processing Word file: {e}")
        return f"Error processing Word file: {e}"


# func to process file based on the extension

async def process_file(file_path: str) -> str:
    """Processes an uploaded file and extracts its content."""
    file_extension = os.path.splitext(file_path)[1].lower()
    logging.info(f"Processing file: {file_path} with extension: {file_extension}")

    if file_extension == ".pdf":
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
        except Exception as e:
            return f" PDF processing error: {str(e)}"
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

    
    
@cl.on_chat_start
async def on_chat_start():
    # Sending an image with the local file path
    elements = [
        cl.Image(name="image1", display="inline", path="chat.png")
    ]
    actions=[cl.Action(name="Record Voice",value="voice_input",label="Record voice",icon="microphone")]
    await cl.Message(content="Hello there, I am Chatbot. You can ask me anything or upload a file for analysis.",
                    elements=elements,
                    actions=actions
    ).send()

    
    
    valid_backends = ["groq", "openai", "anthropic"]
    backend = os.getenv("LANGCHAIN_BACKEND", "groq").lower()
    

    if backend == "groq" and GROQ_API_KEY:
        model = ChatGroq(temperature=0, model_name="llama3-70b-8192", groq_api_key=GROQ_API_KEY)
    elif backend == "openai" and OPENAI_API_KEY:
        model = ChatOpenAI(temperature=0, model_name="gpt-4", openai_api_key=OPENAI_API_KEY)
    elif backend == "anthropic" and ANTHROPIC_API_KEY:
        model = ChatAnthropic(temperature=0, model_name="claude-3-opus-20240229", anthropic_api_key=ANTHROPIC_API_KEY)
    else:
        raise ValueError(f"Invalid backend ,Choose from {valid_backends}")

    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You're a knowledgeable Machine Learning Engineer."),
        ("human", "{question}")
    ])
    cl.user_session.set("runnable", prompt | model | StrOutputParser())


async def process_voice_input():
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            await cl.Message(content="Listening...(10s)").send()
            audio=r.listen(source,timeout=10)
            user_query=await asyncio.get_event_loop().run_in_executor(None,r.recognize_google,audio)
            await cl.Message(content=f"you said: {user_query}").send()
            return user_query
            
    except sr.UnknownValueError:
        await cl.Message(content="Sorry, I could not understand your speech. Please try again.").send()
    except sr.RequestError:
        await cl.Message(content="There was an issue with the speech recognition service. Please try again later.").send()
    except Exception as e:
        logging.error(f"Error in voice input: {e}")
        await cl.Message(content=f"Error processing voice input: {e}").send()

       

@cl.on_action
async def on_action(action:cl.Action):
    if action.value == "voice_input":
        if user_query := await process_voice_input():
            await on_message(cl.Message(content=user_query))
           
             
        

@cl.on_message
async def on_message(message: cl.Message):
    runnable=cl.user_session.get("runnable")
    user_query = message.content.strip()
    msg = cl.Message(content="")
    file_text=''
    MAX_FILE_SIZE=50 * 1024 * 1024  

    # processing uploaded files
    if message.elements:
        for element in message.elements:
            if isinstance(element, cl.File):
                # check for file size before processing
                if os.path.getsize(element.path)>MAX_FILE_SIZE:
                    await cl.Message(content=f"File exceeds the size limit of 50MB.").send()
                    return
                file_text=await process_file(element.path)
    
    if "Unsupported file type" in file_text:
        await cl.Message(content=file_text).send()
        return
    
    
    combined_input = f"User Query: {user_query}\n\nFile Content:\n{file_text}" if file_text else user_query

# Auto completion and Emotion analysis

    if user_query:
        # Auto completion
        auto_completion= generate_auto_completion(user_query)
        await cl.Message(content=f"auto-completion: {auto_completion}").send() 

        # Emotion analysis

        emotion,emotion_score=analyse_emotion(user_query)
        emotion_icons = {
            "joy": "😊", "anger": "😠", "sadness": "😢",
            "fear": "😨", "surprise": "😲", "love": "❤️",
            "neutral": "😐"
        }
        icon = emotion_icons.get(emotion, "")
        await cl.Message(content=f"Emotion: {emotion} {icon} (Score: {emotion_score:.2f})").send()
       
    try:
        async for chunk in runnable.astream(
            {"question": combined_input},
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ):
            await msg.stream_token(chunk)
    except Exception as e:
        logging.error(f"LLM streaming error: {traceback.format_exc()}")
        msg.content=f" An error occured: {str(e)}"
    finally:
        await msg.send()



 


       
