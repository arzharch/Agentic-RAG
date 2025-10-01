"""
Tools for the ReAct agent in the Evidence Gatherer node.
Provides semantic search and file reading capabilities.
"""

import os
from langchain.tools import tool
from typing import List
from config import FILES_DIRECTORY, TOP_K_CHUNKS
from retrieval.vectorstore import vectorstore_manager


@tool
def vector_search_chunks(query: str, k: int = TOP_K_CHUNKS) -> str:
    """
    Search for semantically similar content across all document chunks.
    Use this to find relevant passages that might answer the query.
    
    Args:
        query (str): What to search for (e.g., "Q1 budget allocation")
        k (int): Number of results to return (default: 5)
    
    Returns:
        str: Formatted search results with sources and content
    
    Example:
        vector_search_chunks("Christmas sales performance")
    """
    try:
        results = vectorstore_manager.search_similar_chunks(query, k=k)
        
        if not results:
            return "No relevant information found."
        
        # Format results for the agent
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(
                f"--- Result {i} (from {result['source']}) ---\n"
                f"{result['content']}\n"
            )
        
        return "\n".join(formatted)
    
    except Exception as e:
        return f"Error during search: {str(e)}"


@tool
def read_file_section(filename: str, start_line: int = 0, num_lines: int = 50) -> str:
    """
    Read a specific section of a file to get more context.
    Use this when search results mention something but you need surrounding context.
    
    Args:
        filename (str): Name of the file (e.g., "budget_report_q1.txt")
        start_line (int): Line number to start reading from (0-indexed)
        num_lines (int): How many lines to read
    
    Returns:
        str: The requested file content
    
    Example:
        read_file_section("budget_report_q1.txt", start_line=10, num_lines=20)
    """
    try:
        filepath = os.path.join(FILES_DIRECTORY, filename)
        
        if not os.path.exists(filepath):
            return f"Error: File '{filename}' not found in {FILES_DIRECTORY}"
        
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if start_line >= len(lines):
            return f"Error: start_line {start_line} exceeds file length ({len(lines)} lines)"
        
        end_line = min(start_line + num_lines, len(lines))
        section = lines[start_line:end_line]
        
        return (
            f"--- {filename} (lines {start_line+1} to {end_line}) ---\n"
            + "".join(section)
        )
    
    except Exception as e:
        return f"Error reading file: {str(e)}"


# List of all tools available to the ReAct agent
all_tools = [vector_search_chunks, read_file_section]