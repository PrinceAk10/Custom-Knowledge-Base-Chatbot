# Custom Knowledge Base Chatbot
Simple Chat UI using Large Language Model Groq, LangChain and Chainlit

# OBJECTIVE OF THE PROJECT

The goal of this project is to create an AI-powered chatbot that can answer user
queries based on uploaded documents (PDFs, text files, etc.). It utilizes LangChain
and Large Language Models (LLMs) to provide relevant and context-aware responses.

# Key Features
Document Upload & Processing – Users can upload PDFs, TXT, or DOCX files, and
the chatbot extracts key information.

LLM-Powered Responses – Uses OpenAI’s GPT-3.5/4 or Hugging Face models for
accurate answers.

Text Chunking & Embeddings – Converts large documents into searchable vector
representations using OpenAI or Hugging Face embeddings.

Interactive Web UI – Built with Streamlit for easy user interaction.

Scalability & Flexibility – Can integrate with other vector stores, APIs, and local
LLMs for more control.
 
# Technologies Used
 LangChain – Framework to integrate LLMs with memory and vector search.
 
 LLMs (GPT-3.5-Turbo/GPT-4, Hugging Face Models) – Generates intelligent
 responses.
 
 OpenAI Embeddings / Hugging Face Transformers – Converts text into vector
 representations.
 
 Chainlit – User-friendly web interface for chatbot interaction.
 
 Python – Data processing and document handling.
 
 # Outcome
 The chatbot allows users to upload documents and query them dynamically, offering
 fast and relevant answers. It is useful for:
 
• Corporate Knowledge Management – Automating document search for
  businesses.
  
• Education & Research – Summarizing and explaining academic papers.

  # Future Enhancements
  Real-time document updates – Update knowledge base dynamically.
  
  Multi-language support – Translate and process content in various languages.
  
  Voice Input/Output – Enable speech-based queries.
  
  Offline Mode – Run chatbot locally without an internet connection.

### Tech stack being used
- LLMs from [Groq](https://groq.com/) website.
- [LangChain](https://www.langchain.com/) as a Framework for LLM
- [LangSmith](https://smith.langchain.com/) for developing, collaborating, testing, deploying, and monitoring LLM applications.
- [Chainlit](https://docs.chainlit.io/langchain) for deploying.

## System Requirements

You must have Python 3.10 or later installed. Earlier versions of python may not compile.

## Steps to Replicate 

1. Fork this repository (optional) and clone it locally.
   ```
   git clone git@github.com:PrinceAk10/Custom-Knowledge-Base-Chatbot.git
   cd langchain-groq-chainlit
   ```

2. Create a virtualenv and activate it.
   ```
   python3 -m venv .venv && source .venv/bin/activate
   ```

3. OPTIONAL - Rename example.env to .env with `cp example.env .env`and input the environment variables from [LangSmith](https://smith.langchain.com/). You need to create an account in LangSmith website if you haven't already. Also, you need to get api key for groq from this [link](https://console.groq.com/keys).
   ``` 
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
   LANGCHAIN_API_KEY="your-api-key"
   LANGCHAIN_PROJECT="your-project"
   GROQ_API_KEY="YOUR_GROQ_API_KEY"
   ```

4. Run the following command in the terminal to install necessary python packages:
   ```
   pip install -r requirements.txt
   ```

5. Run the following command in your terminal to start the chat UI:
   ```
   chainlit run langchain_groq_chainlit.py
   ```
# Collaborator

AKSHAY ABHAY KULLU [Team Lead]

JOHN CHRISTOFER\
SAM THANGA DANIEL\
MARIYA VISWA

# OUTPUT

![Screenshot 2025-03-14 205857](https://github.com/user-attachments/assets/e64815ff-acfc-4913-aa7a-dd7df1bbb205)

![Screenshot 2025-03-14 205830](https://github.com/user-attachments/assets/fa3ffa79-d5bc-4e75-a575-cbe7e83d5996)

