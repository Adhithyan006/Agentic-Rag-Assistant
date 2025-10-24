import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from sentence_transformers import SentenceTransformer

class RealRAGSystem:
    def __init__(self):
        # Initialize the embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )

        # Load existing documents
        self._load_existing_documents()

    def _load_existing_documents(self):
        """Load documents from the documents folder into ChromaDB"""
        documents_path = Path("documents")
        if not documents_path.exists():
            documents_path.mkdir()
            print("Created documents folder")
            return

        for file_path in documents_path.glob("*.txt*"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                if content:
                    # Check if document already exists
                    existing = self.collection.get(ids=[file_path.name])
                    if not existing['ids']:
                        self.add_document(content, file_path.name)
                        print(f"Loaded document: {file_path.name}")
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")

    def add_document(self, content: str, filename: str):
        """Add a document with sentence-level chunking for precision"""
        try:
            # Sentence-level chunking for maximum precision
            sentences = [s.strip() + '.' for s in re.split(r'[.!?]+', content.strip()) if s.strip()]
            
            if not sentences:
                sentences = [content]

            # Generate embeddings for each sentence
            embeddings = self.model.encode(sentences)

            # Create unique IDs
            ids = [f"{filename}_sent_{i}" for i in range(len(sentences))]

            # Enhanced metadata for each sentence
            metadatas = [
                {
                    "source": filename,
                    "sentence_id": i,
                    "content": sentence,
                    "word_count": len(sentence.split()),
                    "content_type": self._classify_content(sentence)
                }
                for i, sentence in enumerate(sentences)
            ]

            # Add to ChromaDB
            self.collection.add(
                documents=sentences,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
                ids=ids
            )

            print(f"Added {len(sentences)} sentences from {filename}")
            return True
        except Exception as e:
            print(f"Error adding document {filename}: {e}")
            return False

    def _classify_content(self, sentence: str) -> str:
        """Classify content type for better matching"""
        sentence_lower = sentence.lower()
        
        if any(word in sentence_lower for word in ['my name', 'i am', 'called']):
            return 'identity'
        elif any(word in sentence_lower for word in ['i work', 'job', 'profession', 'engineer']):
            return 'work'
        elif any(word in sentence_lower for word in ['i live', 'from', 'city', 'place']):
            return 'location'
        elif any(word in sentence_lower for word in ['hobby', 'like', 'enjoy', 'favorite']):
            return 'preference'
        elif any(word in sentence_lower for word in ['pet', 'dog', 'cat', 'animal']):
            return 'pet'
        elif any(word in sentence_lower for word in ['color', 'red', 'blue', 'green', 'yellow']):
            return 'color'
        elif any(word in sentence_lower for word in ['is a', 'are', 'fruit', 'animal']):
            return 'definition'
        else:
            return 'general'

    def search(self, query: str, n_results: int = 15) -> List[Dict[str, Any]]:
        """Advanced search with multiple strategies"""
        try:
            query_embedding = self.model.encode([query])

            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results
            )

            formatted_results = []
            for i in range(len(results['documents'][0])):
                distance = results['distances'][0][i]
                similarity_score = max(0, 1 - distance)

                formatted_results.append({
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "score": similarity_score,
                    "distance": distance
                })

            # Filter and sort
            formatted_results = [r for r in formatted_results if r['score'] > 0.1]
            formatted_results.sort(key=lambda x: x['score'], reverse=True)

            return formatted_results
        except Exception as e:
            print(f"Error searching: {e}")
            return []

    def _extract_pure_answer(self, question: str, sentence: str) -> str:
        """Extract ONLY the answer part from sentence - like Claude"""
        question_lower = question.lower().strip()
        sentence = sentence.strip()
        
        print(f"[DEBUG] Extracting from: {sentence}")
        print(f"[DEBUG] Question: {question}")
        
        # Question pattern analysis for pure answer extraction
        
        # Name questions
        if any(pattern in question_lower for pattern in ['what is your name', 'what\'s your name', 'your name']):
            # Extract name from "My name is X" or "I am X"
            name_match = re.search(r'(?:my name is|i am|called)\s+([A-Za-z]+)', sentence, re.IGNORECASE)
            if name_match:
                return name_match.group(1).strip()
        
        # Pet name questions
        elif any(pattern in question_lower for pattern in ['pet name', 'pet\'s name', 'dog name', 'cat name']):
            # Extract pet name from "pet dog named X" or "dog is X"
            pet_match = re.search(r'(?:named|called)\s+([A-Za-z]+)', sentence, re.IGNORECASE)
            if pet_match:
                return pet_match.group(1).strip()
        
        # Hobby questions
        elif any(pattern in question_lower for pattern in ['hobby', 'hobbies', 'what do you like']):
            # Extract hobby from "My hobby is X" or "I like X"
            hobby_match = re.search(r'(?:hobby is|hobbies are|i like|enjoy)\s+([^.]+)', sentence, re.IGNORECASE)
            if hobby_match:
                hobby = hobby_match.group(1).strip()
                # Clean up common prefixes
                hobby = re.sub(r'^(playing|doing|watching)\s+', '', hobby, flags=re.IGNORECASE)
                return hobby
        
        # Location questions
        elif any(pattern in question_lower for pattern in ['where do you live', 'where are you from', 'your location']):
            # Extract location from "I live in X" or "from X"
            location_match = re.search(r'(?:live in|from|in)\s+([A-Za-z\s]+)', sentence, re.IGNORECASE)
            if location_match:
                location = location_match.group(1).strip()
                return location
        
        # Work questions
        elif any(pattern in question_lower for pattern in ['where do you work', 'what do you do', 'your job', 'work as']):
            # Extract job from "I work as X" or "I am a X"
            job_match = re.search(r'(?:work as|i am|as a)\s+([^.]+)', sentence, re.IGNORECASE)
            if job_match:
                job = job_match.group(1).strip()
                job = re.sub(r'^(a|an)\s+', '', job, flags=re.IGNORECASE)
                return job
        
        # Color questions
        elif any(pattern in question_lower for pattern in ['what color', 'color is', 'color of']):
            # Extract color from "X is red" or "red fruit"
            color_match = re.search(r'(?:is a?\s+)?(\w+)(?:\s+(?:fruit|color|animal))?', sentence, re.IGNORECASE)
            colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'black', 'white', 'pink', 'brown']
            for color in colors:
                if color in sentence.lower():
                    return color.title()
        
        # Action questions (what does X do)
        elif any(pattern in question_lower for pattern in ['what does', 'what do', 'do they do']):
            # Extract action from "X barks" or "dogs bark"
            action_match = re.search(r'(?:dogs?|cats?|birds?|fish)\s+(\w+)', sentence, re.IGNORECASE)
            if action_match:
                action = action_match.group(1)
                return action
        
        # Location questions for animals/objects
        elif any(pattern in question_lower for pattern in ['where do', 'where does', 'live in', 'swim in']):
            # Extract location from "fish swim in water"
            location_words = ['water', 'sky', 'trees', 'ocean', 'forest', 'house', 'nest']
            for word in location_words:
                if word in sentence.lower():
                    return f"in {word}" if word in ['water', 'sky', 'ocean', 'forest'] else word
        
        # Time questions
        elif any(pattern in question_lower for pattern in ['when does', 'when do', 'what time']):
            # Extract time from "sun rises in the morning"
            time_match = re.search(r'(?:in the|at|during)\s+(\w+)', sentence, re.IGNORECASE)
            if time_match:
                return time_match.group(1)
        
        # Yes/No questions converted to facts
        elif any(pattern in question_lower for pattern in ['is ', 'are ', 'does ', 'do ', 'can ', 'will ']):
            # For yes/no questions, return the key fact
            # Remove common prefixes and return core information
            cleaned = re.sub(r'^(the\s+|a\s+|an\s+)', '', sentence, flags=re.IGNORECASE)
            return cleaned
        
        # Default: return the most informative part
        # Remove common sentence starters
        sentence = re.sub(r'^(the\s+|a\s+|an\s+|my\s+|i\s+)', '', sentence, flags=re.IGNORECASE)
        return sentence

    def _find_best_answer(self, question: str, relevant_docs: List[Dict]) -> str:
        """Find the best answer with pure answer extraction"""
        if not relevant_docs:
            return "I don't have information about that in my knowledge base."
        
        question_lower = question.lower().strip()
        
        print(f"[DEBUG] Analyzing question: {question}")
        print(f"[DEBUG] Found {len(relevant_docs)} relevant documents")
        
        # Score each document based on question relevance
        scored_results = []
        
        for doc in relevant_docs:
            content = doc['content']
            metadata = doc['metadata']
            base_score = doc['score']
            
            # Content type matching bonus
            content_type = metadata.get('content_type', 'general')
            type_bonus = 0
            
            if 'name' in question_lower and content_type == 'identity':
                type_bonus = 2.0
            elif any(word in question_lower for word in ['hobby', 'like']) and content_type == 'preference':
                type_bonus = 2.0
            elif any(word in question_lower for word in ['work', 'job']) and content_type == 'work':
                type_bonus = 2.0
            elif any(word in question_lower for word in ['live', 'from']) and content_type == 'location':
                type_bonus = 2.0
            elif any(word in question_lower for word in ['pet', 'dog', 'cat']) and content_type == 'pet':
                type_bonus = 2.0
            elif 'color' in question_lower and content_type == 'color':
                type_bonus = 2.0
            
            # Keyword matching score
            question_words = [w for w in question_lower.split() if len(w) > 2]
            keyword_matches = sum(1 for word in question_words if word in content.lower())
            
            final_score = base_score + type_bonus + (keyword_matches * 0.5)
            
            scored_results.append({
                'content': content,
                'score': final_score,
                'metadata': metadata
            })
        
        # Sort by final score
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Debug top results
        print(f"[DEBUG] Top 3 results:")
        for i, result in enumerate(scored_results[:3]):
            print(f"[DEBUG] {i+1}. Score: {result['score']:.2f} | Content: {result['content'][:60]}...")
        
        # Get best result and extract pure answer
        best_result = scored_results[0]
        best_sentence = best_result['content']
        
        # Extract pure answer
        pure_answer = self._extract_pure_answer(question, best_sentence)
        
        print(f"[DEBUG] Pure answer extracted: {pure_answer}")
        
        return pure_answer

    def ask(self, question: str) -> Dict[str, Any]:
        """Answer questions with pure answer extraction"""
        try:
            print(f"\n=== Processing Question: {question} ===")
            
            # Search for relevant documents
            relevant_docs = self.search(question, n_results=15)
            
            if not relevant_docs:
                return {
                    "answer": "I don't have relevant information to answer your question.",
                    "sources": [],
                    "status": "no_results"
                }
            
            # Find the best pure answer
            answer = self._find_best_answer(question, relevant_docs)
            
            # Format sources
            sources = []
            seen_sources = set()
            for doc in relevant_docs[:3]:
                source_name = doc['metadata']['source']
                if source_name not in seen_sources:
                    sources.append({
                        "source_file": source_name,
                        "content": doc['content'][:80] + "..." if len(doc['content']) > 80 else doc['content'],
                        "score": f"{doc['score']:.3f}"
                    })
                    seen_sources.add(source_name)
            
            return {
                "answer": answer,
                "sources": sources,
                "status": "success"
            }
            
        except Exception as e:
            print(f"Error in ask: {e}")
            return {
                "answer": f"Error processing question: {str(e)}",
                "sources": [],
                "status": "error"
            }

    def delete_document(self, filename: str) -> bool:
        """Delete a document from the vector database and file system"""
        try:
            results = self.collection.get(where={"source": filename})
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                print(f"Deleted {len(results['ids'])} entries from ChromaDB for {filename}")
            
            file_path = Path("documents") / filename
            if file_path.exists():
                file_path.unlink()
                print(f"Deleted file: {filename}")
            
            return True
            
        except Exception as e:
            print(f"Error deleting document {filename}: {e}")
            return False

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the knowledge base"""
        try:
            documents_path = Path("documents")
            if not documents_path.exists():
                return []
            
            docs = []
            for file_path in documents_path.glob("*.txt*"):
                try:
                    file_size = file_path.stat().st_size
                    docs.append({
                        "filename": file_path.name,
                        "size": file_size,
                        "size_formatted": f"{file_size} bytes"
                    })
                except Exception as e:
                    print(f"Error reading {file_path.name}: {e}")
            
            return docs
            
        except Exception as e:
            print(f"Error listing documents: {e}")
            return []

    def get_document_content(self, filename: str) -> str:
        """Get the content of a specific document"""
        try:
            file_path = Path("documents") / filename
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            return "File not found"
        except Exception as e:
            return f"Error reading file: {e}"


# Global instance
rag_system = None

def get_rag_system():
    """Get or create RAG system instance"""
    global rag_system
    if rag_system is None:
        print("Initializing Super Intelligent RAG system...")
        rag_system = RealRAGSystem()
        print("🚀 Super Intelligent RAG system ready! Pure answers only!")
    return rag_system