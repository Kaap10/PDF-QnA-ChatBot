# 📚 Intelligent PDF QnA Chatbot

A sophisticated conversational agent that allows users to upload PDF documents and receive intelligent, context-aware answers to their questions. This project showcases the power of **Retrieval Augmented Generation (RAG)** for building robust document-based QnA systems.

![PDF Chatbot](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-00FF00?style=for-the-badge&logo=langchain&logoColor=black)
![FAISS](https://img.shields.io/badge/FAISS-FF6B6B?style=for-the-badge&logo=facebook&logoColor=white)

---

## ✨ Features

- **📄 Smart PDF Processing**: Upload any PDF document (resumes, research papers, notes, etc.)
- **🤖 Intelligent QnA**: Get accurate and contextual answers directly from your PDF content
- **🔍 Advanced Search**: FAISS-powered semantic similarity search for lightning-fast retrieval
- **🎨 Beautiful UI**: Modern, responsive interface with dark/light theme support
- **⚡ Real-time Processing**: Instant feedback and responses
- **🔄 Chat Interface**: Natural conversation flow with chat history
- **💾 Persistent Storage**: Vector embeddings are saved for quick reloading
- **🛡️ Error Handling**: Graceful error management and user feedback

---

## 🏗️ Architecture

This project implements a **Retrieval Augmented Generation (RAG)** system with the following components:

```
📄 PDF Upload → 🔧 Text Processing → 🔍 Vector Embedding → 💾 FAISS Storage → ❓ Query → 🔍 Semantic Search → 🤖 RAG Generation → 💬 Answer
```

### Core Modules:

1. **`app.py`** - Main Streamlit application with beautiful UI
2. **`loader.py`** - PDF processing and text chunking using LangChain
3. **`vectorstore.py`** - FAISS vector database integration
4. **`qa_chain.py`** - RAG-based question answering logic

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit (Modern web interface)
- **AI/ML**: OpenAI GPT-3.5/4 + OpenAI Embeddings
- **Framework**: LangChain (LLM orchestration)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Document Processing**: PyPDFLoader + RecursiveCharacterTextSplitter
- **Language**: Python 3.8+
- **Environment**: python-dotenv

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API Key ([Get one here](https://platform.openai.com/account/api-keys))

### Installation

1. **Clone or download the project files**

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows:
   .\venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your OpenAI API Key**
   - Rename `env_template.txt` to `.env`
   - Replace `your_openai_api_key_here` with your actual OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - Upload a PDF and start asking questions!

---

## 📖 How to Use

### 1. Upload a PDF
- Click "Browse files" in the sidebar
- Select any PDF document
- Click "Process PDF" to extract and embed the content

### 2. Ask Questions
- Type your question in the chat input
- The system will search for relevant content and generate an answer
- View the conversation history in the chat interface

### 3. Manage Your Session
- **Clear Chat**: Remove conversation history
- **Clear Vector Store**: Remove processed PDF data
- **Upload New PDF**: Process a different document

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
```

### Customization

You can modify the following parameters in the code:

- **Chunk Size**: Adjust text chunking in `loader.py`
- **Model**: Change OpenAI model in `qa_chain.py`
- **Search Results**: Modify number of retrieved documents in `vectorstore.py`
- **UI**: Customize styling in `app.py`

---

## 📁 Project Structure

```
PDF Chatbot/
├── app.py              # Main Streamlit application
├── loader.py           # PDF processing & chunking
├── vectorstore.py      # FAISS vector database
├── qa_chain.py         # RAG question answering
├── requirements.txt    # Python dependencies
├── env_template.txt    # Environment template
├── README.md          # This file
└── sample.pdf         # Test document (add your own)
```

---

## 🎯 How It Works

### 1. Document Processing
- **PDF Upload**: User uploads a PDF via Streamlit interface
- **Text Extraction**: LangChain's PyPDFLoader extracts text content
- **Chunking**: RecursiveCharacterTextSplitter breaks text into manageable chunks

### 2. Vector Embedding
- **Embedding Generation**: OpenAI Embeddings API converts text chunks to vectors
- **FAISS Storage**: Vectors stored in FAISS index for fast similarity search

### 3. Question Answering (RAG)
- **Query Processing**: User's question is processed
- **Semantic Search**: FAISS finds most relevant document chunks
- **Context Assembly**: Relevant chunks are combined with the question
- **Answer Generation**: OpenAI GPT model generates contextual answer

---

## 🚨 Troubleshooting

### Common Issues

1. **"OpenAI API key not found"**
   - Ensure `.env` file exists and contains your API key
   - Check that the key is valid and has sufficient credits

2. **"Error loading PDF"**
   - Verify the PDF file is not corrupted
   - Ensure the PDF contains extractable text (not just images)

3. **"Error creating vector store"**
   - Check your internet connection
   - Verify OpenAI API key is working

4. **Slow performance**
   - Large PDFs may take longer to process
   - Consider reducing chunk size for faster processing

### Getting Help

- Check the console output for detailed error messages
- Ensure all dependencies are installed correctly
- Verify your OpenAI API key has sufficient credits

---

## 🔒 Security & Privacy

- **API Keys**: Never commit your `.env` file to version control
- **Data Processing**: PDFs are processed locally, not uploaded to external servers
- **Vector Storage**: Embeddings are stored locally in FAISS format
- **OpenAI**: Only text chunks and queries are sent to OpenAI API

---

## 🤝 Contributing

Feel free to contribute to this project by:

1. Forking the repository
2. Creating a feature branch
3. Making your changes
4. Submitting a pull request

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- **OpenAI** for providing powerful language models
- **LangChain** for the excellent LLM orchestration framework
- **FAISS** for efficient similarity search
- **Streamlit** for the beautiful web interface framework

---

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the console output for error messages
3. Ensure all dependencies are properly installed
4. Verify your OpenAI API key is valid

---

**Happy PDF Chatting! 🎉** 