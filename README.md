## Custom Knowledge Base Chatbot

# 🚀 Project Objective

The Custom Knowledge Base Chatbot is an AI-powered chatbot that enables users to query uploaded documents (PDFs, text files, DOCX) dynamically. By leveraging LangChain and Large Language Models (LLMs), it provides context-aware and relevant responses based on document contents.

# 🏆 Key Features

✅ Document Upload & Processing – Users can upload PDFs, TXT, or DOCX files, and the chatbot extracts key information for querying.✅ LLM-Powered Responses – Utilizes GPT-3.5/4 (Groq) and Hugging Face models for intelligent answers.✅ Text Chunking & Embeddings – Converts large documents into searchable vector representations using OpenAI or Hugging Face embeddings.✅ Interactive Web UI – Built with Chainlit, providing a user-friendly interface.✅ Scalability & Flexibility – Can integrate with other vector stores, APIs, and local LLMs for better control.

# 🛠 Technologies Used

🔹 LangChain – Framework for integrating LLMs with memory and vector search.🔹 LLMs (GPT-3.5-Turbo/GPT-4, Hugging Face Models) – To generate intelligent responses.🔹 OpenAI Embeddings / Hugging Face Transformers – Converts text into vector representations for semantic search.🔹 Chainlit – User-friendly web interface for chatbot interaction.🔹 Python – Core programming language for data processing and document handling.

# 🎯 Use Cases

🔹 Corporate Knowledge Management – Automates document search for businesses.🔹 Education & Research – Summarizes and explains academic papers efficiently.🔹 Customer Support – Enables instant, knowledge-driven support.

# 📌 Future Enhancements

✅ Real-time Document Updates – Dynamically update the knowledge base.✅ Multi-language Support – Translate and process documents in various languages.✅ Voice Input/Output – Enable speech-based queries and responses.✅ Offline Mode – Run chatbot locally without an internet connection.

# ⚡ Tech Stack

🟢 LLMs from Groq🟢 LangChain – As a Framework for LLM🟢 LangSmith – For developing, testing, and monitoring LLM applications🟢 Chainlit – For seamless deployment🟢 Python 3.10+ – Ensure compatibility for proper execution

# 🛠 System Requirements

🔹 Python 3.10+ (Older versions may not compile)🔹 Git – For cloning the repository

# ⚙️ Steps to Replicate

# 1️⃣ Clone the Repository

git clone git@github.com:PrinceAk10/Custom-Knowledge-Base-Chatbot.git
cd langchain-groq-chainlit

# 2️⃣ Create a Virtual Environment & Activate It

python3 -m venv .venv && source .venv/bin/activate

# 3️⃣ Set Up Environment Variables (Optional but Recommended)

Rename example.env to .env and configure it with your LangSmith API Key and Groq API Key.

cp example.env .env

🔹 Create an account on LangSmith and obtain API keys.🔹 Get your Groq API Key from Groq's API Key Page.

# Add the following details in your .env file:

LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="your-api-key"
LANGCHAIN_PROJECT="your-project"
GROQ_API_KEY="YOUR_GROQ_API_KEY"

# 4️⃣ Install Dependencies

pip install -r requirements.txt

# 5️⃣ Run the Chatbot UI

chainlit run langchain_groq_chainlit.py

👥 Collaborators

👨‍💻 Akshay Abhay Kullu – Team Lead👨‍💻 John Christofer Sam👨‍💻 Thanga Daniel👨‍💻 Mariya Viswa
