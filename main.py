from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import os
from pathlib import Path
from real_rag import get_rag_system

app = FastAPI(title="Agentic RAG Assistant")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the HTML file
@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

# Pydantic models for API
class AskIn(BaseModel):
    question: str

class AskOut(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    status: str = "success"

class DocumentIn(BaseModel):
    content: str
    filename: str

@app.get("/api")
def read_root():
    return {"message": "Welcome to Agentic RAG Assistant 🚀", "status": "Server running successfully!"}

@app.get("/health")
def health():
    return {"status": "ok", "message": "RAG system is healthy"}

@app.post("/ask", response_model=AskOut)
def ask(payload: AskIn):
    """
    Ask a question to the RAG system
    """
    try:
        # Get RAG system instance
        rag = get_rag_system()
        
        # Ask the question using real RAG
        result = rag.ask(payload.question)
        
        return AskOut(
            answer=result["answer"],
            sources=result["sources"],
            status=result["status"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/add-document")
def add_document(payload: DocumentIn):
    """
    Add a new document to the RAG system
    """
    try:
        # Save to documents folder
        documents_path = Path("documents")
        documents_path.mkdir(exist_ok=True)
        
        # Handle filename - add .txt if not present
        filename = payload.filename
        if not filename.endswith('.txt'):
            filename = f"{filename}.txt"
        
        file_path = documents_path / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(payload.content)

        # Add to RAG system
        rag = get_rag_system()
        success = rag.add_document(payload.content, filename)
        
        if success:
            return {
                "message": f"Document '{filename}' added successfully to RAG system",
                "status": "success"
            }
        else:
            return {
                "message": f"Document saved but failed to add to RAG system",
                "status": "warning"
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding document: {str(e)}")

@app.delete("/delete-document/{filename}")
def delete_document(filename: str):
    """
    Delete a document from the RAG system and file system
    """
    try:
        # Get RAG system instance
        rag = get_rag_system()
        
        # Delete from RAG system and file system
        success = rag.delete_document(filename)
        
        if success:
            return {
                "message": f"Document '{filename}' deleted successfully",
                "status": "success"
            }
        else:
            raise HTTPException(status_code=404, detail=f"Document '{filename}' not found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@app.get("/view-document/{filename}")
def view_document(filename: str):
    """
    View the content of a specific document
    """
    try:
        file_path = Path("documents") / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Document '{filename}' not found")
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        return PlainTextResponse(content=content)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error viewing document: {str(e)}")

@app.get("/documents")
def list_documents():
    """
    List all documents in the system - FIXED VERSION
    """
    try:
        documents = []
        documents_path = Path("documents")
        
        if documents_path.exists():
            for file_path in documents_path.glob("*.txt"):
                stat = file_path.stat()
                documents.append({
                    "filename": file_path.name,
                    "size": stat.st_size,
                    "size_formatted": f"{stat.st_size} bytes"
                })
        
        return {
            "documents": documents,
            "count": len(documents),
            "status": "success"
        }
        
    except Exception as e:
        return {
            "documents": [],
            "count": 0,
            "status": "error",
            "message": str(e)
        }

@app.post("/refresh-documents")
def refresh_documents():
    """
    Refresh the document index - reload all documents
    """
    try:
        # Get RAG system instance
        rag = get_rag_system()
        
        # Re-embed all documents
        rag.embed_documents()
        
        return {
            "message": "Documents refreshed successfully",
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error refreshing documents: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)