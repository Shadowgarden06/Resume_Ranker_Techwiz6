import os
import json
import re
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx
from datetime import datetime

class DocumentQA:
    def __init__(self, api_key: str = None):
        """
        Initialize Document QA system with Google Gemini API
        """
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable.")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Try different models in order of preference
        model_names = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
        self.model = None
        
        for model_name in model_names:
            try:
                self.model = genai.GenerativeModel(model_name)
                # Test the model with a simple request
                test_response = self.model.generate_content("test")
                print(f"✅ Using Gemini model: {model_name}")
                break
            except Exception as e:
                print(f"⚠️ Model {model_name} not available: {e}")
                continue
        
        if self.model is None:
            raise ValueError("No compatible Gemini model found. Please check your API key and try again.")
        
        # Initialize sentence transformer for embeddings
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Document storage
        self.documents = {}
        self.document_embeddings = {}
        
    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from PDF or DOCX files"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".pdf":
            text = ""
            with open(file_path, "rb") as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    if page.extract_text():
                        text += page.extract_text()
            return text
            
        elif ext == ".docx":
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        
        return ""
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks for better processing"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > start + chunk_size // 2:  # Only break if we don't lose too much
                    chunk = text[start:start + break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            
        return chunks
    
    def add_document(self, file_path: str, document_id: str = None) -> str:
        """Add a document to the QA system"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if document_id is None:
            document_id = os.path.basename(file_path)
        
        # Extract text
        text = self.extract_text_from_file(file_path)
        if not text.strip():
            raise ValueError("No text content found in the document")
        
        # Chunk the text
        chunks = self.chunk_text(text)
        
        # Create embeddings for each chunk
        chunk_embeddings = self.embedding_model.encode(chunks)
        
        # Store document data
        self.documents[document_id] = {
            'file_path': file_path,
            'text': text,
            'chunks': chunks,
            'added_at': datetime.now().isoformat()
        }
        
        self.document_embeddings[document_id] = chunk_embeddings
        
        return document_id
    
    def find_relevant_chunks(self, query: str, document_id: str, top_k: int = 3) -> List[Dict]:
        """Find most relevant text chunks for a query"""
        if document_id not in self.documents:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(
            query_embedding, 
            self.document_embeddings[document_id]
        )[0]
        
        # Get top-k most similar chunks
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        relevant_chunks = []
        for idx in top_indices:
            relevant_chunks.append({
                'text': self.documents[document_id]['chunks'][idx],
                'similarity': float(similarities[idx]),
                'chunk_index': int(idx)
            })
        
        return relevant_chunks
    
    def ask_question(self, question: str, document_id: str = None, context_chunks: int = 3) -> Dict[str, Any]:
        """Ask a question about a specific document or all documents"""
        try:
            if document_id and document_id in self.documents:
                # Question about specific document
                relevant_chunks = self.find_relevant_chunks(question, document_id, context_chunks)
                context_text = "\n\n".join([chunk['text'] for chunk in relevant_chunks])
                source_doc = document_id
            else:
                # Question about all documents
                all_chunks = []
                for doc_id in self.documents:
                    chunks = self.find_relevant_chunks(question, doc_id, context_chunks)
                    for chunk in chunks:
                        chunk['document_id'] = doc_id
                        all_chunks.append(chunk)
                
                # Sort by similarity and take top chunks
                all_chunks.sort(key=lambda x: x['similarity'], reverse=True)
                relevant_chunks = all_chunks[:context_chunks]
                context_text = "\n\n".join([chunk['text'] for chunk in relevant_chunks])
                source_doc = "multiple documents" if len(self.documents) > 1 else list(self.documents.keys())[0]
            
            if not context_text.strip():
                return {
                    'answer': "I couldn't find relevant information to answer your question.",
                    'sources': [],
                    'confidence': 0.0
                }
            
            # Create prompt for Gemini
            prompt = f"""
            Based on the following document content, please answer the question: "{question}"
            
            Document content:
            {context_text}
            
            Please provide a clear, accurate answer based only on the provided content. If the answer is not available in the content, please say so.
            """
            
            # Get response from Gemini
            response = self.model.generate_content(prompt)
            answer = response.text if response.text else "I couldn't generate an answer."
            
            # Calculate confidence based on similarity scores
            avg_similarity = np.mean([chunk['similarity'] for chunk in relevant_chunks]) if relevant_chunks else 0.0
            confidence = min(avg_similarity * 100, 100.0)
            
            return {
                'answer': answer,
                'sources': relevant_chunks,
                'confidence': round(confidence, 2),
                'document_id': source_doc
            }
            
        except Exception as e:
            return {
                'answer': f"Error processing question: {str(e)}",
                'sources': [],
                'confidence': 0.0
            }
    
    def get_document_list(self) -> List[Dict[str, Any]]:
        """Get list of all documents in the system"""
        doc_list = []
        for doc_id, doc_data in self.documents.items():
            doc_list.append({
                'id': doc_id,
                'file_path': doc_data['file_path'],
                'added_at': doc_data['added_at'],
                'chunk_count': len(doc_data['chunks'])
            })
        return doc_list
    
    def remove_document(self, document_id: str) -> bool:
        """Remove a document from the system"""
        if document_id in self.documents:
            del self.documents[document_id]
            del self.document_embeddings[document_id]
            return True
        return False
    
    def clear_all_documents(self):
        """Clear all documents from the system"""
        self.documents.clear()
        self.document_embeddings.clear()

# Global instance
qa_system = None

def initialize_qa_system(api_key: str = None):
    """Initialize the global QA system"""
    global qa_system
    try:
        qa_system = DocumentQA(api_key)
        print("✅ Document QA system initialized successfully")
        return True
    except Exception as e:
        print(f"⚠️ Failed to initialize Document QA system: {e}")
        qa_system = None
        return False

def get_qa_system():
    """Get the global QA system instance"""
    global qa_system
    if qa_system is None:
        raise RuntimeError("QA system not initialized. Call initialize_qa_system() first.")
    return qa_system

def is_qa_available():
    """Check if QA system is available"""
    global qa_system
    return qa_system is not None
