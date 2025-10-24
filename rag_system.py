import os
import pickle
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

# FREE libraries - no API keys needed!
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Document:
    """Simple document class"""
    def __init__(self, content: str, metadata: Dict[str, Any] = None):
        self.page_content = content
        self.metadata = metadata or {}

class FreeRAGSystem:
    def __init__(self):
        """Initialize the FREE RAG system with no API keys required"""
        self.embedding_model = None
        self.index = None
        self.documents = []
        self.document_embeddings = []
        self.documents_path = Path("documents")
        self.vectorstore_path = Path("vectorstore")
        
        # Ensure directories exist
        self.documents_path.mkdir(exist_ok=True)
        self.vectorstore_path.mkdir(exist_ok=True)
        
        # Initialize embedding model (FREE!)
        self._setup_embedding_model()
        
    def _setup_embedding_model(self):
        """Setup FREE sentence transformer model"""
        try:
            logger.info("Loading FREE embedding model...")
            # This is completely free - no API key needed!
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise
    
    def load_documents(self) -> List[Document]:
        """Load documents from the documents directory"""
        documents = []
        
        if not any(self.documents_path.iterdir()):
            # Create sample documents if directory is empty
            self._create_sample_documents()
        
        # Load all text files
        for file_path in self.documents_path.glob("*.txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                
                if content:  # Only add non-empty documents
                    doc = Document(
                        content=content,
                        metadata={"source": file_path.name, "path": str(file_path)}
                    )
                    documents.append(doc)
                    logger.info(f"Loaded document: {file_path.name}")
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
        
        if not documents:
            raise ValueError("No documents found in documents directory")
        
        logger.info(f"Successfully loaded {len(documents)} documents")
        return documents
    
    def _create_sample_documents(self):
        """Create sample documents for demonstration"""
        sample_docs = [
            {
                "filename": "ai_basics.txt",
                "content": """
                Artificial Intelligence (AI) is a branch of computer science that aims to create 
                intelligent machines capable of performing tasks that typically require human intelligence. 
                These tasks include learning, reasoning, problem-solving, perception, and language understanding.
                
                Machine Learning is a subset of AI that enables computers to learn and improve from 
                experience without being explicitly programmed. It uses algorithms and statistical models 
                to analyze and draw inferences from patterns in data.
                
                Deep Learning is a subset of machine learning that uses artificial neural networks 
                with multiple layers to model and understand complex patterns in data. It has revolutionized 
                fields like computer vision, natural language processing, and speech recognition.
                """
            },
            {
                "filename": "rag_systems.txt",
                "content": """
                Retrieval-Augmented Generation (RAG) is an AI framework that combines information 
                retrieval with text generation. It works by first retrieving relevant information 
                from a knowledge base, then using that information to generate more accurate and 
                contextually relevant responses.
                
                RAG systems typically consist of two main components:
                1. A retrieval system that finds relevant documents or passages using semantic search
                2. A generation model that produces responses based on the retrieved information
                
                Vector databases are commonly used in RAG systems to store document embeddings 
                and enable efficient similarity search for information retrieval. Popular vector 
                databases include FAISS, Chroma, and Pinecone.
                """
            },
            {
                "filename": "langchain_intro.txt",
                "content": """
                LangChain is a framework for developing applications powered by language models. 
                It provides tools and utilities for building complex applications that can reason 
                about data and interact with various data sources.
                
                Key components of LangChain include:
                - Document Loaders: For loading data from various sources
                - Text Splitters: For breaking large documents into smaller chunks
                - Embeddings: For converting text to numerical representations
                - Vector Stores: For storing and retrieving embeddings
                - Chains: For combining different components into workflows
                
                LangChain makes it easy to build RAG systems, chatbots, and other AI applications
                by providing pre-built components and abstractions.
                """
            }
        ]
        
        for doc in sample_docs:
            file_path = self.documents_path / doc["filename"]
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(doc["content"])
        
        logger.info("Created sample documents for demonstration")
    
    def split_text(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
        """Split text into chunks"""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            if end > text_len:
                end = text_len
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - chunk_overlap
            if start <= 0:
                break
                
        return chunks
    
    def create_embeddings(self, documents: List[Document]) -> np.ndarray:
        """Create embeddings for documents"""
        all_chunks = []
        chunk_metadata = []
        
        # Split documents into chunks
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    **doc.metadata,
                    "chunk_id": i,
                    "chunk_text": chunk[:100] + "..." if len(chunk) > 100 else chunk
                })
        
        logger.info(f"Creating embeddings for {len(all_chunks)} chunks...")
        
        # Create embeddings using FREE model
        embeddings = self.embedding_model.encode(all_chunks, show_progress_bar=True)
        
        # Store document chunks and metadata
        self.documents = []
        for i, chunk in enumerate(all_chunks):
            self.documents.append(Document(
                content=chunk,
                metadata=chunk_metadata[i]
            ))
        
        logger.info(f"Created embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def create_vector_index(self, embeddings: np.ndarray):
        """Create FAISS vector index"""
        logger.info("Creating FAISS vector index...")
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        # Save index and documents
        self._save_index()
        
        logger.info(f"Vector index created with {self.index.ntotal} vectors")
    
    def _save_index(self):
        """Save FAISS index and documents"""
        try:
            index_path = self.vectorstore_path / "faiss.index"
            docs_path = self.vectorstore_path / "documents.pkl"
            
            # Save FAISS index
            faiss.write_index(self.index, str(index_path))
            
            # Save documents
            with open(docs_path, "wb") as f:
                pickle.dump(self.documents, f)
            
            logger.info("Index and documents saved successfully")
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    def _load_index(self) -> bool:
        """Load existing FAISS index and documents"""
        try:
            index_path = self.vectorstore_path / "faiss.index"
            docs_path = self.vectorstore_path / "documents.pkl"
            
            if not (index_path.exists() and docs_path.exists()):
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            
            # Load documents
            with open(docs_path, "rb") as f:
                self.documents = pickle.load(f)
            
            logger.info(f"Loaded existing index with {self.index.ntotal} vectors")
            return True
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
    
    def setup_system(self):
        """Setup the complete RAG system"""
        # Try to load existing index
        if self._load_index():
            logger.info("Using existing vector index")
            return
        
        # Create new index
        logger.info("Creating new vector index...")
        documents = self.load_documents()
        embeddings = self.create_embeddings(documents)
        self.create_vector_index(embeddings)
    
    def search_similar_documents(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if not self.index or not self.documents:
            self.setup_system()
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search similar documents
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score),
                    "rank": i + 1
                })
        
        return results
    
    def generate_response(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Generate response using retrieved context (FREE - no LLM API needed)"""
        if not context_docs:
            return f"I don't have enough information to answer your question about '{query}'. Please add relevant documents to help me provide better answers."
        
        # Simple but effective response generation
        context_text = "\n\n".join([doc["content"] for doc in context_docs])
        
        # Create a structured response
        response_parts = []
        
        # Find the most relevant parts
        query_lower = query.lower()
        best_matches = []
        
        for doc in context_docs:
            content_lower = doc["content"].lower()
            # Simple relevance scoring
            if any(word in content_lower for word in query_lower.split()):
                best_matches.append(doc["content"])
        
        if best_matches:
            # Use the most relevant content
            main_content = best_matches[0]
            
            # Extract key sentences
            sentences = main_content.split('.')
            relevant_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and any(word in sentence.lower() for word in query_lower.split()):
                    relevant_sentences.append(sentence)
            
            if relevant_sentences:
                response = '. '.join(relevant_sentences[:3]) + '.'
                response = response.replace('..', '.')
                return response
        
        # Fallback: return first chunk of most relevant document
        return context_docs[0]["content"][:500] + "..." if len(context_docs[0]["content"]) > 500 else context_docs[0]["content"]
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system"""
        try:
            # Ensure system is setup
            if not self.index:
                self.setup_system()
            
            # Search for relevant documents
            similar_docs = self.search_similar_documents(question, k=3)
            
            if not similar_docs:
                return {
                    "answer": f"I couldn't find relevant information to answer '{question}'. Please add more documents to the knowledge base.",
                    "sources": [],
                    "status": "no_results"
                }
            
            # Generate response
            answer = self.generate_response(question, similar_docs)
            
            # Format sources
            sources = []
            for doc in similar_docs:
                source_info = {
                    "content": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                    "metadata": {
                        "source": doc["metadata"].get("source", "unknown"),
                        "score": f"{doc['score']:.3f}",
                        "rank": doc["rank"]
                    }
                }
                sources.append(source_info)
            
            return {
                "answer": answer,
                "sources": sources,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "answer": f"Error processing your question: {str(e)}",
                "sources": [],
                "status": "error"
            }
    
    def add_document(self, content: str, filename: str):
        """Add a new document to the system"""
        try:
            # Save document
            file_path = self.documents_path / filename
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            # Rebuild the index with new document
            logger.info(f"Adding document: {filename}")
            self.setup_system()
            
            logger.info(f"Document '{filename}' added successfully")
            return True
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return False

# Global instance
free_rag_system = FreeRAGSystem()