"""
Vector store management using FAISS.
Handles document loading, chunking, embedding, and semantic search.
"""

import os
from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import (
    FILES_DIRECTORY, VECTOR_STORE_PATH, CHUNK_SIZE, 
    CHUNK_OVERLAP, TOP_K_CHUNKS, REBUILD_VECTORSTORE
)
from retrieval.embeddings import get_embeddings


class VectorStoreManager:
    """
    Manages the FAISS vector store for semantic search.
    Handles document loading, chunking, and retrieval.
    """
    
    def __init__(self):
        """Initialize the vector store manager."""
        self.embeddings = get_embeddings()
        self.vectorstore = None
        self.documents = []
        
    def build_vectorstore(self) -> None:
        """
        Build the vector store from .txt files in FILES_DIRECTORY.
        Creates document chunks and embeds them using the configured embedding model.
        """
        print(f"\n{'='*60}")
        print("BUILDING VECTOR STORE")
        print(f"{'='*60}")
        
        # Check if vector store already exists
        if os.path.exists(VECTOR_STORE_PATH) and not REBUILD_VECTORSTORE:
            print("Loading existing vector store...")
            self.vectorstore = FAISS.load_local(
                VECTOR_STORE_PATH, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("✓ Vector store loaded successfully")
            return
        
        # Load all .txt files
        print(f"Loading documents from: {FILES_DIRECTORY}")
        loader = DirectoryLoader(
            FILES_DIRECTORY,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        
        documents = loader.load()
        print(f"✓ Loaded {len(documents)} documents")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        
        self.documents = text_splitter.split_documents(documents)
        print(f"✓ Split into {len(self.documents)} chunks")
        
        # Create vector store
        print("Embedding documents (this may take a minute)...")
        self.vectorstore = FAISS.from_documents(self.documents, self.embeddings)
        
        # Save for future use
        self.vectorstore.save_local(VECTOR_STORE_PATH)
        print(f"✓ Vector store saved to: {VECTOR_STORE_PATH}")
        print(f"{'='*60}\n")
    
    def search_similar_chunks(self, query: str, k: int = TOP_K_CHUNKS) -> List[Dict]:
        """
        Perform semantic search to find the most relevant document chunks.
        
        Args:
            query (str): The search query
            k (int): Number of results to return
            
        Returns:
            List[Dict]: List of relevant chunks with metadata
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call build_vectorstore() first.")
        
        # Perform similarity search with scores
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=k)
        
        # Format results
        results = []
        for doc, score in docs_and_scores:
            results.append({
                'content': doc.page_content,
                'source': os.path.basename(doc.metadata['source']),
                'relevance_score': float(score)
            })
        
        return results
    
    def get_file_relevance_scores(self, query: str) -> Dict[str, float]:
        """
        Get relevance scores for each file based on the query.
        Used by the Retrieval Planner to decide which files to examine.
        
        Args:
            query (str): The search query
            
        Returns:
            Dict[str, float]: Mapping of filename to relevance score
        """
        # Get top chunks from each file
        all_chunks = self.search_similar_chunks(query, k=20)
        
        # Aggregate scores by file
        file_scores = {}
        for chunk in all_chunks:
            filename = chunk['source']
            score = chunk['relevance_score']
            
            if filename not in file_scores:
                file_scores[filename] = []
            file_scores[filename].append(score)
        
        # Average the scores for each file
        averaged_scores = {
            filename: sum(scores) / len(scores) 
            for filename, scores in file_scores.items()
        }
        
        # Sort by relevance
        return dict(sorted(averaged_scores.items(), key=lambda x: x[1]))


# Global instance
vectorstore_manager = VectorStoreManager()