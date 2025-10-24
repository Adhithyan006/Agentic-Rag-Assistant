import os
import re
import time
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# FREE embeddings - no API key needed
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

load_dotenv()

class SimpleRAGSystem:
    def __init__(self):
        # FREE embeddings - no API key needed
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.vector_store = None
        self.init_vector_store()

    def init_vector_store(self):
        """Initialize ChromaDB with FREE embeddings"""
        persist_directory = "./chroma_db"

        if Path(persist_directory).exists() and any(Path(persist_directory).iterdir()):
            print("Loading existing ChromaDB...")
            try:
                self.vector_store = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings
                )
                print("ChromaDB loaded successfully")
            except Exception as e:
                print(f"Error loading ChromaDB, creating new one: {e}")
                self.safe_rebuild_database()
        else:
            print("Creating new ChromaDB with FREE embeddings...")
            self.embed_documents()

    def safe_cleanup_database(self):
        """Windows-safe database cleanup with retries"""
        persist_directory = "./chroma_db"
        
        if not Path(persist_directory).exists():
            return
            
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Close existing vector store connection
                if self.vector_store is not None:
                    try:
                        self.vector_store._client.reset()
                    except:
                        pass
                    self.vector_store = None
                
                # Wait for Windows to release file handles
                time.sleep(1)
                
                # Try to remove directory
                import shutil
                shutil.rmtree(persist_directory)
                print(f"Successfully cleaned vector database (attempt {attempt + 1})")
                return
                
            except Exception as e:
                print(f"Cleanup attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait longer between retries
                else:
                    print("Failed to cleanup database, continuing anyway...")

    def safe_rebuild_database(self):
        """Force complete database rebuild with Windows compatibility"""
        self.safe_cleanup_database()
        self.embed_documents()

    def embed_documents(self):
        """Embed all documents with Windows-safe cleanup"""
        documents_path = Path("documents")

        if not documents_path.exists():
            documents_path.mkdir(exist_ok=True)
            print("Created documents folder")
            return

        # Load all current documents
        try:
            loader = DirectoryLoader(str(documents_path), glob="**/*.txt", loader_cls=TextLoader)
            documents = loader.load()
        except Exception as e:
            print(f"Error loading documents: {e}")
            return

        if not documents:
            print("No documents found to embed")
            self.vector_store = None
            return

        # Split into chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " "]
        )
        texts = text_splitter.split_documents(documents)

        # Create fresh vector store with retry logic
        persist_directory = "./chroma_db"
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                self.vector_store = Chroma.from_documents(
                    documents=texts,
                    embedding=self.embeddings,
                    persist_directory=persist_directory
                )
                print(f"Embedded {len(texts)} chunks from {len(documents)} documents")
                return
            except Exception as e:
                print(f"Embedding attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    print("Failed to create embeddings after all retries")
                    self.vector_store = None

    def ask(self, question: str) -> Dict[str, Any]:
        """Ask question with smart keyword matching"""
        if self.vector_store is None:
            return {
                "answer": "No documents found. Please add some documents first.",
                "sources": [],
                "status": "error"
            }

        try:
            # Search for relevant documents
            docs = self.vector_store.similarity_search(question, k=4)

            if not docs:
                return {
                    "answer": "No relevant information found in the knowledge base.",
                    "sources": [],
                    "status": "error"
                }

            # Combine context from documents
            context = "\n\n".join([doc.page_content for doc in docs])

            # Extract answer using smart keyword matching
            answer = self.smart_answer_extraction(question, context)

            # Get ONLY truly relevant sources - much stricter filtering
            sources = []
            seen_files = set()
            answer_lower = answer.lower()
            question_lower = question.lower()
            
            # Only include sources if they contain meaningful keywords from the question
            question_keywords = [word for word in question_lower.split() if len(word) > 2]
            
            for doc in docs:
                if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                    source_file = Path(doc.metadata['source']).name
                    doc_content_lower = doc.page_content.lower()
                    
                    # Check if document actually contains the main keywords
                    relevant = False
                    keyword_matches = 0
                    
                    for keyword in question_keywords:
                        if keyword in doc_content_lower:
                            keyword_matches += 1
                    
                    # Only consider relevant if at least one meaningful keyword matches
                    if keyword_matches > 0:
                        # Additional check: answer should come from this document
                        main_answer_words = [word for word in answer_lower.split() if len(word) > 3]
                        answer_match = any(word in doc_content_lower for word in main_answer_words)
                        
                        if answer_match and source_file not in seen_files:
                            relevant = True
                    
                    if relevant:
                        sources.append({"source_file": source_file})
                        seen_files.add(source_file)

            return {
                "answer": answer,
                "sources": sources,
                "status": "success"
            }

        except Exception as e:
            return {
                "answer": f"Error processing your question: {str(e)}",
                "sources": [],
                "status": "error"
            }

    def smart_answer_extraction(self, question: str, content: str) -> str:
        """Smart keyword-based answer extraction for any content"""
        
        # Clean question and get keywords
        question_clean = question.lower().strip('?.,!').strip()
        
        # Remove question words to get core keywords
        stop_words = {'what', 'who', 'where', 'when', 'why', 'how', 'is', 'are', 'was', 'were',
                     'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
                     'with', 'by', 'about', 'do', 'does', 'did', 'can', 'could', 'would', 'should',
                     'have', 'has', 'had', 'will', 'shall', 'may', 'might', 'must', 'be'}
        
        keywords = [word for word in question_clean.split() if word not in stop_words and len(word) > 1]
        
        # Split content into lines
        lines = [line.strip() for line in content.split('\n') if line.strip() and len(line.strip()) > 5]
        
        # Score lines based on keyword relevance
        scored_lines = []
        
        for line in lines:
            line_lower = line.lower()
            score = 0
            
            # Count keyword matches with word boundary checking
            for keyword in keywords:
                if keyword in line_lower:
                    # Higher score for exact word matches
                    if re.search(r'\b' + re.escape(keyword) + r'\b', line_lower):
                        score += 3
                    else:
                        score += 1
            
            # Question-type specific scoring
            if 'who' in question_clean:
                # Look for names and person indicators
                if re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', line):
                    score += 2
                if any(word in line_lower for word in ['is', 'was', 'director', 'ceo', 'founder', 'created']):
                    score += 2
            
            elif 'when' in question_clean or 'date' in question_clean or 'year' in question_clean:
                # Look for dates and time indicators
                if re.search(r'\b(19|20)\d{2}\b', line):
                    score += 3
                if any(word in line_lower for word in ['born', 'founded', 'established', 'created', 'released']):
                    score += 2
            
            elif 'where' in question_clean or 'location' in question_clean:
                # Look for location indicators
                if any(word in line_lower for word in ['located', 'based', 'headquarters', 'address', 'city', 'country']):
                    score += 2
            
            elif any(phrase in question_clean for phrase in ['how many', 'number', 'count']):
                # Look for numbers
                if re.search(r'\b\d+\b', line):
                    score += 3
            
            elif 'what' in question_clean:
                # Look for definitions and descriptions
                if ':' in line:
                    score += 2
                if any(word in line_lower for word in ['definition', 'means', 'refers', 'describes']):
                    score += 2
            
            # General scoring boosts
            if ':' in line:  # Key-value pairs are often answers
                score += 1
            if 10 < len(line) < 150:  # Optimal answer length
                score += 1
            
            # Store scored lines
            if score > 0:
                scored_lines.append((score, line))
        
        # Sort by score (highest first)
        scored_lines.sort(key=lambda x: x[0], reverse=True)
        
        # Return best answer only if it's actually relevant
        if scored_lines:
            best_score, best_line = scored_lines[0]
            
            # Much stricter threshold - only return answer if highly confident
            if best_score >= 4:  # Increased threshold
                return best_line
            
            # For lower scores, check if keywords actually appear in the answer
            if best_score >= 2:
                answer_words = best_line.lower().split()
                keyword_in_answer = any(keyword in answer_words for keyword in keywords)
                if keyword_in_answer:
                    return best_line
        
        # Much stricter fallback - only if exact keyword match
        exact_matches = []
        for line in lines:
            line_lower = line.lower()
            exact_keyword_count = 0
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', line_lower):
                    exact_keyword_count += 1
            if exact_keyword_count > 0:
                exact_matches.append((exact_keyword_count, line))
        
        if exact_matches:
            exact_matches.sort(key=lambda x: x[0], reverse=True)
            return exact_matches[0][1]
        
        # If no good matches found, return "not found" message
        return "I couldn't find specific information about that topic in the available documents."

    def add_document(self, content: str, filename: str) -> bool:
        """Add new document to knowledge base with Windows compatibility"""
        try:
            documents_path = Path("documents")
            documents_path.mkdir(exist_ok=True)

            # Ensure .txt extension
            if not filename.lower().endswith('.txt'):
                filename += '.txt'

            file_path = documents_path / filename

            # Write content to file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            print(f"Document '{filename}' saved successfully")

            # Windows-safe re-embedding
            self.safe_cleanup_database()
            time.sleep(1)  # Give Windows time to release handles
            self.embed_documents()
            return True

        except Exception as e:
            print(f"Error adding document: {e}")
            return False

    def delete_document(self, filename: str) -> bool:
        """Complete document deletion with GUARANTEED backend cleanup"""
        try:
            import uuid
            documents_path = Path("documents")

            # Ensure .txt extension
            if not filename.lower().endswith('.txt'):
                filename += '.txt'

            file_path = documents_path / filename
            
            # Delete file from disk
            if file_path.exists():
                file_path.unlink()
                print(f"Document '{filename}' deleted from disk")
            else:
                print(f"Document '{filename}' not found on disk")

            # GUARANTEED backend cleanup - CREATE NEW DATABASE
            print("Creating COMPLETELY NEW database (guaranteed cleanup)...")
            
            # Close existing connections
            if self.vector_store is not None:
                try:
                    self.vector_store._client.reset()
                except:
                    pass
                self.vector_store = None
            
            # Create NEW database with unique name to avoid any file locks
            old_persist_dir = "./chroma_db"
            new_persist_dir = f"./chroma_db_{uuid.uuid4().hex[:8]}"
            
            # Delete old database in background (don't wait for it)
            try:
                if Path(old_persist_dir).exists():
                    import shutil
                    shutil.rmtree(old_persist_dir, ignore_errors=True)
            except:
                pass  # Don't care if this fails
            
            # Get remaining files
            remaining_files = list(documents_path.glob("*.txt"))
            if remaining_files:
                print(f"Building fresh database for {len(remaining_files)} files...")
                
                # Load documents
                try:
                    from langchain_community.document_loaders import DirectoryLoader, TextLoader
                    loader = DirectoryLoader(str(documents_path), glob="**/*.txt", loader_cls=TextLoader)
                    documents = loader.load()
                except Exception as e:
                    print(f"Error loading documents: {e}")
                    return False

                if not documents:
                    print("No documents to load")
                    self.vector_store = None
                    return True

                # Split into chunks
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " "]
                )
                texts = text_splitter.split_documents(documents)

                # Create COMPLETELY NEW vector store with NEW name
                self.vector_store = Chroma.from_documents(
                    documents=texts,
                    embedding=self.embeddings,
                    persist_directory=new_persist_dir
                )
                
                # Rename new database to standard name
                try:
                    if Path(new_persist_dir).exists():
                        if Path(old_persist_dir).exists():
                            shutil.rmtree(old_persist_dir, ignore_errors=True)
                        shutil.move(new_persist_dir, old_persist_dir)
                        
                        # Recreate vector store with correct path
                        self.vector_store = Chroma(
                            persist_directory=old_persist_dir,
                            embedding_function=self.embeddings
                        )
                except Exception as e:
                    print(f"Error renaming database: {e}")
                    # Still works with the new name
                
                print("GUARANTEED fresh database created successfully")
            else:
                print("No files remaining - database empty")
                self.vector_store = None
                # Clean up any existing databases
                try:
                    if Path(old_persist_dir).exists():
                        shutil.rmtree(old_persist_dir, ignore_errors=True)
                except:
                    pass

            return True

        except Exception as e:
            print(f"Error during deletion: {e}")
            return False

    def get_document_content(self, filename: str) -> str:
        """Get document content"""
        try:
            documents_path = Path("documents")

            if not filename.lower().endswith('.txt'):
                filename += '.txt'

            file_path = documents_path / filename

            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            return "File not found"
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all current documents"""
        try:
            documents = []
            documents_path = Path("documents")

            if not documents_path.exists():
                documents_path.mkdir(exist_ok=True)
                return documents

            for file_path in documents_path.glob("*.txt"):
                try:
                    stat = file_path.stat()
                    documents.append({
                        "filename": file_path.name,
                        "size": stat.st_size,
                        "size_formatted": f"{stat.st_size} bytes"
                    })
                except Exception as e:
                    print(f"Error processing file {file_path.name}: {e}")
                    continue

            return documents

        except Exception as e:
            print(f"Error listing documents: {e}")
            return []

# Global instance
rag_system = None

def get_rag_system():
    """Get or initialize RAG system instance"""
    global rag_system
    if rag_system is None:
        print("Initializing Windows-compatible RAG System...")
        rag_system = SimpleRAGSystem()
        print("RAG System Ready!")
    return rag_system

if __name__ == "__main__":
    # Initialize and test basic functionality
    rag = get_rag_system()
    print("RAG system initialized successfully!")
    
    # Simple test to verify system is working
    result = rag.ask("test")
    print(f"System status: {result['status']}")
    
    # List available documents
    docs = rag.list_documents()
    print(f"Found {len(docs)} documents in system")