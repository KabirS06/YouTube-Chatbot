# 🎥 YouTube Chatbot (RAG-Powered)

An interactive Streamlit web application that allows users to ask questions about any YouTube video. The application fetches the video's transcript, processes the text into manageable chunks, indexes it using a vector store, and utilizes a Retrieval-Augmented Generation (RAG) pipeline to deliver precise answers using a local or open-source LLM.

---

## 🚀 Features

- **Instant Transcript Extraction:** Automatically extracts text transcripts from YouTube URLs (supporting standard and shortened formats).
- **Smart Text Chunking:** Utilizes LangChain's `RecursiveCharacterTextSplitter` to optimize long transcripts for processing.
- **Semantic Search Vector Store:** Embeds transcript data with `sentence-transformers` and indexes it using **FAISS** for fast similarity lookups.
- **Context-Bound Answers:** Uses `Llama-3.1-8B-Instruct` to guarantee that answers are strictly grounded within the video's content.
- **Streamlit UI:** Clean, modern, and user-friendly web interface.

---

## 🛠️ Architecture Workflow

1. **User Input:** Enter a YouTube link and a specific question.
2. **Parsing & Fetching:** Extracts the unique Video ID and pulls the raw transcript text.
3. **Chunking & Embedding:** Slices the text down into overlapping semantic fragments and embeds them into vector space.
4. **Retrieval Chain:** Retrieves the top 4 most contextually relevant chunks matching the user's question.
5. **Generation:** Passes the isolated context chunks alongside the user query to the LLM to formulate an answer.

---

## 📋 Prerequisites & Installation

```bash
git clone [https://github.com/KabirS06/Gen-AI-.git](https://github.com/KabirS06/Gen-AI-.git)
cd Gen-AI-
cd LangChain
python -m venv venv
# Activate the Virtual Environment
    # On Windows:
venv\Scripts\activate
    # On macOS/Linux:
source venv/bin/activate
pip install streamlit langchain-huggingface langchain-core langchain-classic youtube-transcript-api python-dotenv faiss-cpu sentence-transformers

#Set Api Token :
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token_here

### Run the Application:
streamlit run app.py
```
