# Custom Knowledge Base Chatbot
Simple Chat UI using Large Language Model Groq, LangChain and Chainlit

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

