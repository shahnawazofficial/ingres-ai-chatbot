"""
Simple Chatbot for INGRES
"""

import google.generativeai as genai
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

class INGRESChatbot:
    def __init__(self):
        # Configure Gemini
        api_key = os.getenv('GEMINI_API_KEY')
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Load vector store
        client = PersistentClient(path="data/embeddings")
        self.collection = client.get_collection("groundwater_data")
        
        # Load embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("✅ Chatbot initialized!")
    
    def get_response(self, query, n_results=5):
        # Search vector store
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        context = "\n\n".join(results['documents'][0])
        
        # Create prompt
        prompt = f"""You are an expert for INGRES groundwater system.

Context: {context}

Question: {query}

Provide a clear answer with specific data."""
        
        # Get response
        response = self.model.generate_content(prompt)
        
        return {
            "status": "success",
            "query": query,
            "response": response.text
        }