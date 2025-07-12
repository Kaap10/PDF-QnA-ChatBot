"""
PDF Document Loader and Processor
Handles PDF loading, text extraction, and chunking for the QnA system.
"""

import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import streamlit as st


class PDFProcessor:
    """Handles PDF document processing and text chunking."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the PDF processor.
        
        Args:
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between chunks for context preservation
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_pdf(self, pdf_file) -> List[Document]:
        """
        Load and process a PDF file.
        
        Args:
            pdf_file: Uploaded PDF file from Streamlit
            
        Returns:
            List of Document objects containing text chunks
        """
        try:
            # Save uploaded file temporarily
            with open("temp_pdf.pdf", "wb") as f:
                f.write(pdf_file.getvalue())
            
            # Load PDF using LangChain
            loader = PyPDFLoader("temp_pdf.pdf")
            documents = loader.load()
            
            # Clean up temporary file
            os.remove("temp_pdf.pdf")
            
            return documents
            
        except Exception as e:
            st.error(f"Error loading PDF: {str(e)}")
            return []
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for better processing.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        try:
            if not documents:
                return []
            
            chunks = self.text_splitter.split_documents(documents)
            st.success(f"✅ Document processed into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            st.error(f"Error splitting documents: {str(e)}")
            return []
    
    def process_pdf(self, pdf_file) -> List[Document]:
        """
        Complete PDF processing pipeline.
        
        Args:
            pdf_file: Uploaded PDF file
            
        Returns:
            List of processed Document chunks
        """
        with st.spinner("📄 Processing PDF document..."):
            # Load PDF
            documents = self.load_pdf(pdf_file)
            if not documents:
                return []
            
            # Split into chunks
            chunks = self.split_documents(documents)
            return chunks 