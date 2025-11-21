"""
FastAPI Server for INGRES Chatbot
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
import requests
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chromadb import PersistentClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="INGRES Chatbot API",
    description="AI-powered chatbot for India Groundwater Resource Estimation System",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
collection = None
api_key = None

# Request/Response Models
class ChatRequest(BaseModel):
    query: str
    n_results: int = 5

class ChatResponse(BaseModel):
    status: str
    query: str
    response: str

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    global collection, api_key
    
    try:
        print("\n🚀 Starting INGRES Chatbot API...")
        print("="*60)
        
        # Load API key
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("❌ ERROR: GEMINI_API_KEY not found in .env file")
            return
        print(f"✅ API key loaded")
        
        # Load vector store
        client = PersistentClient(path="data/embeddings")
        collection = client.get_collection("groundwater_data")
        print(f"✅ Vector store loaded ({collection.count()} documents)")
        
        print("\n📚 API Documentation: http://localhost:8000/docs")
        print("🌐 API Root: http://localhost:8000")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Startup error: {e}")

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "status": "success",
        "message": "INGRES Chatbot API is running!",
        "docs": "http://localhost:8000/docs"
    }

@app.get("/health")
async def health():
    """Health check"""
    if collection is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    return {
        "status": "healthy",
        "documents": collection.count()
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    
    if collection is None or api_key is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Search vector store
        results = collection.query(
            query_texts=[request.query],
            n_results=request.n_results
        )
        
        # Get context
        context = "\n\n".join(results['documents'][0])
        
        # Create prompt
        prompt = f"""You are an expert assistant for INGRES (India Groundwater Resource Estimation System).

Context from database:
{context}

User Question: {request.query}

Provide a clear, accurate answer based on the context. Include specific numbers and district names.

Categories:
- Safe: Stage of Extraction < 70%
- Semi-Critical: 70-90%
- Critical: 90-100%
- Over-Exploited: > 100%

Answer:"""
        
        # Call Gemini API
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        
        headers = {'Content-Type': 'application/json'}
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        
        # Retry logic
        max_retries = 3
        for attempt in range(max_retries):
            response = requests.post(
                f"{url}?key={api_key}",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result['candidates'][0]['content']['parts'][0]['text']
                
                return ChatResponse(
                    status="success",
                    query=request.query,
                    response=answer
                )
            
            elif response.status_code == 503:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    raise HTTPException(status_code=503, detail="AI temporarily unavailable")
            
            else:
                raise HTTPException(status_code=response.status_code, detail=response.text)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run server
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "🌊 "*15)
    print("     INGRES CHATBOT API SERVER")
    print("🌊 "*15 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )