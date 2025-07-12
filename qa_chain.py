"""
Question Answering Chain with RAG
Implements Retrieval Augmented Generation for intelligent QnA.
"""

from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import Document, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
import streamlit as st


class QAChatbot:
    """Handles question answering using RAG (Retrieval Augmented Generation)."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.7):
        """
        Initialize the QA chatbot.
        
        Args:
            model_name: OpenAI model to use for generation
            temperature: Controls randomness in responses
        """
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature
        )
        self.system_prompt = self._create_system_prompt()
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the chatbot."""
        return """You are an intelligent assistant that helps users understand documents. 
        
Your task is to answer questions based ONLY on the provided document context. 

IMPORTANT RULES:
1. Only use information from the provided context to answer questions
2. If the context doesn't contain enough information to answer the question, say "I don't have enough information from the document to answer this question."
3. Be accurate, helpful, and concise
4. Cite specific parts of the document when possible
5. If asked about something not in the document, politely redirect to document content

Context: {context}

Question: {question}

Answer:"""
    
    def _format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into context string.
        
        Args:
            documents: List of relevant documents
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            # Clean and format the document content
            content = doc.page_content.strip()
            if content:
                context_parts.append(f"Document Section {i}:\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _create_messages(self, question: str, context: str) -> List[Dict[str, str]]:
        """
        Create messages for the chat model.
        
        Args:
            question: User's question
            context: Retrieved document context
            
        Returns:
            List of message dictionaries
        """
        return [
            {
                "role": "system",
                "content": self.system_prompt.format(context=context, question=question)
            },
            {
                "role": "user", 
                "content": question
            }
        ]
    
    def answer_question(self, question: str, relevant_docs: List[Document]) -> str:
        """
        Generate an answer using RAG.
        
        Args:
            question: User's question
            relevant_docs: Retrieved relevant documents
            
        Returns:
            Generated answer
        """
        try:
            if not relevant_docs:
                return "I don't have enough information from the document to answer this question. Please make sure you've uploaded a PDF and try asking a different question."
            
            # Format context from retrieved documents
            context = self._format_context(relevant_docs)
            
            # Create messages for the chat model
            messages = self._create_messages(question, context)
            
            with st.spinner("🤔 Generating answer..."):
                # Generate response using the chat model
                response = self.llm.invoke(messages)
                
                return response.content.strip()
                
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
            return "Sorry, I encountered an error while generating the answer. Please try again."
    
    def get_answer_with_sources(self, question: str, relevant_docs: List[Document]) -> Dict[str, Any]:
        """
        Generate answer with source information.
        
        Args:
            question: User's question
            relevant_docs: Retrieved relevant documents
            
        Returns:
            Dictionary containing answer and source information
        """
        answer = self.answer_question(question, relevant_docs)
        
        # Extract source information
        sources = []
        for doc in relevant_docs:
            if hasattr(doc, 'metadata') and doc.metadata:
                source_info = {
                    'page': doc.metadata.get('page', 'Unknown'),
                    'source': doc.metadata.get('source', 'Document')
                }
                sources.append(source_info)
        
        return {
            'answer': answer,
            'sources': sources,
            'context_count': len(relevant_docs)
        } 