# Agentic RAG System

It implements a sophisticated multi-agent RAG pipeline using LangGraph, built on a decoupled, hybrid-model architecture. This design maximizes performance and cost-efficiency by delegating tasks to the most appropriate resource: a fast, local, open-source model for high-volume retrieval and the powerful Gemini for nuanced, complex reasoning.

## 🎯 Architectural Highlights

- **Decoupled Reasoning & Retrieval**: Implements a sophisticated hybrid architecture, decoupling the reasoning engine (Gemini) from the retrieval engine (local Sentence Transformer + FAISS). This is the core design principle for building scalable, cost-effective AI systems.

- **In-Memory Vector Search w/ FAISS**: Leverages **FAISS (Facebook AI Similarity Search)** for production-grade, in-memory vector search. This enables stateful and near-instantaneous semantic retrieval on a pre-computed index, eliminating the latency and cost of retrieval via API calls.

- **Agentic State Machine**: Orchestrates a 4-node workflow as a deterministic state machine using LangGraph. This ensures a reliable, observable, and extensible reasoning loop, moving beyond simple chains to true agentic behavior.

- **LLM-Controlled Tool Use**: Features a ReAct agent (powered by Gemini) that is empowered with specialized, high-speed local tools. This showcases a powerful paradigm where the LLM acts as a reasoning core, delegating retrieval tasks to the local, purpose-built FAISS index.

## 📋 Prerequisites

1.  **Python 3.11+**
2.  A **Google Gemini API Key**. You can get a free key from [Google AI Studio](https://aistudio.google.com/app/apikey).

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Your API Key

Create a file named `.env` in the main project directory and add your API key:

```
GEMINI_API_KEY=PASTE_YOUR_API_KEY_HERE
```

### 3. Prepare Your Documents

Place all `.txt` files you want to analyze in the `files_to_work_with/` directory.

### 4. Run the System

```bash
python src/main.py
```

## Future Scope : 
By creating a folder to put any file the system can answer most questions regarding it.
Suppose you clone a git repo of old project - you can relearn forgotten content.
More applications of this include creating reports from Excel sheets (convert to text format using Python Libraries), understanding Legal Documents and so on. 

### One-Time Setup: Index Ingestion & Vectorization

- **Technology**: `sentence-transformers/all-MiniLM-L6-v2` + `FAISS`
- **Process**: On first launch, the system performs a one-time ingestion process. All documents are vectorized locally by the Sentence Transformer. These vectors are then used to build a dense FAISS index, which is persisted to disk for future, stateful operations.

### Step 1: Query Analysis (Gemini)

- **Technology**: `gemini-pro`
- **Process**: The high-level reasoning task of understanding the user's intent is delegated to Gemini, which generates a structured analytical breakdown.

### Step 2: Retrieval Planning (Local FAISS Query)

- **Technology**: `FAISS`
- **Process**: In a cost-free, zero-latency operation, the system queries the local FAISS index to perform a semantic search. It aggregates relevance scores to identify the most promising source files, all without a single LLM call.

### Step 3: Evidence Gathering (ReAct Agent)

- **Technology**: `gemini-pro` (Reasoning) + `FAISS` (Tool)
- **Process**: Gemini acts as the agent's reasoning core. It intelligently dispatches retrieval jobs to its specialized `vector_search` tool, which in turn queries the local FAISS index. This allows the agent to iteratively and efficiently build a body of evidence.

### Step 4: Synthesis (Gemini)

- **Technology**: `gemini-pro`
- **Process**: With the evidence gathered, the final, high-value task of synthesis and analysis is handed to Gemini, which constructs the comprehensive, final answer.

## ⏱️ Performance

This decoupled architecture achieves impressive performance:
- **Most queries are answered within 30 seconds.**

By reserving the powerful remote LLM for high-value reasoning and delegating high-volume retrieval to a local, optimized index, the system remains both fast and cost-effective.
