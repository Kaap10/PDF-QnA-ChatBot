"""
FAISS Vector Store Integration
Handles document embedding, storage, and similarity search using FAISS.
"""

import os
import pickle
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import streamlit as st


class VectorStore:
    """Manages FAISS vector store for document embeddings and similarity search."""
    
    def __init__(self, embeddings_model: str = "text-embedding-ada-002"):
        """
        Initialize the vector store.
        
        Args:
            embeddings_model: OpenAI embedding model to use
        """
        self.embeddings = OpenAIEmbeddings(model=embeddings_model)
        self.vector_store = None
        self.index_path = "faiss_index"
        self.metadata_path = "faiss_metadata.pkl"
    
    def create_vector_store(self, documents: List[Document]) -> bool:
        """
        Create FAISS vector store from documents.
        
        Args:
            documents: List of Document objects to embed
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not documents:
                st.warning("⚠️ No documents to process")
                return False
            
            with st.spinner("🔍 Creating vector embeddings..."):
                # Create FAISS vector store
                self.vector_store = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
                
                st.success(f"✅ Vector store created with {len(documents)} documents")
                return True
                
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            return False
    
    def save_vector_store(self) -> bool:
        """
        Save the vector store to disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.vector_store is None:
                return False
            
            # Save FAISS index
            self.vector_store.save_local(self.index_path)
            
            # Save metadata
            metadata = {
                "document_count": len(self.vector_store.docstore._dict),
                "embedding_dimension": self.vector_store.embedding_function.client.dimensions
            }
            
            with open(self.metadata_path, "wb") as f:
                pickle.dump(metadata, f)
            
            st.success("💾 Vector store saved successfully")
            return True
            
        except Exception as e:
            st.error(f"Error saving vector store: {str(e)}")
            return False
    
    def load_vector_store(self) -> bool:
        """
        Load existing vector store from disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(self.index_path):
                return False
            
            with st.spinner("📂 Loading existing vector store..."):
                self.vector_store = FAISS.load_local(
                    folder_path=self.index_path,
                    embeddings=self.embeddings
                )
                
                st.success("✅ Vector store loaded successfully")
                return True
                
        except Exception as e:
            st.error(f"Error loading vector store: {str(e)}")
            return False
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform similarity search for relevant documents.
        
        Args:
            query: User's question
            k: Number of similar documents to retrieve
            
        Returns:
            List of relevant Document objects
        """
        try:
            if self.vector_store is None:
                st.error("❌ No vector store available. Please upload a PDF first.")
                return []
            
            # Perform similarity search
            relevant_docs = self.vector_store.similarity_search(
                query=query,
                k=k
            )
            
            return relevant_docs
            
        except Exception as e:
            st.error(f"Error performing similarity search: {str(e)}")
            return []
    
    def get_store_info(self) -> dict:
        """
        Get information about the current vector store.
        
        Returns:
            Dictionary with store information
        """
        if self.vector_store is None:
            return {"status": "No vector store loaded"}
        
        try:
            return {
                "status": "Loaded",
                "document_count": len(self.vector_store.docstore._dict),
                "embedding_dimension": self.vector_store.embedding_function.client.dimensions
            }
        except:
            return {"status": "Error getting info"}
    
    def clear_store(self):
        """Clear the current vector store."""
        self.vector_store = None
        st.success("🗑️ Vector store cleared") 