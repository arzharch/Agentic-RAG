# LangGraph Agentic RAG System

A multi-agent system built with LangGraph that intelligently answers queries about documents using a 4-node agentic workflow with semantic search and ReAct reasoning.

## 🎯 Features

- **4-Node Agentic Workflow**: Query Analysis → Retrieval Planning → Evidence Gathering → Synthesis
- **Semantic Search**: FAISS vector store for accurate document retrieval
- **ReAct Agent**: Tool-using agent that searches and reads files intelligently
- **Chain-of-Thought**: Transparent reasoning process for complex inferences
- **Local LLM**: Uses Ollama with Mistral for all reasoning tasks
- **Modular & Well-Commented**: Clean code structure for easy understanding

## 📋 Prerequisites

1. **Ollama** installed and running locally
2. **Mistral model** pulled: `ollama pull mistral`
3. **Python 3.9+**

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Documents

Place all `.txt` files you want to analyze in the `files_to_work_with/` directory:

```
files_to_work_with/
├── budget_report_q1.txt
├── customer_feedback_dec.txt
├── technical_debt_analysis.txt
└── ... (more .txt files)
```

### 3. Run the System

```bash
python main.py
```

On first run, the system will:
1. Load all `.txt` files
2. Create document chunks
3. Generate embeddings
4. Build a FAISS vector store
5. Start the interactive CLI

### 4. Ask Questions

```
🔍 Your question: How did the Christmas sale impact our Q1 budget planning?
```

The system will:
1. **Analyze** the query to understand what information is needed
2. **Plan** which files are most relevant using semantic search
3. **Gather** evidence using a ReAct agent with search tools
4. **Synthesize** a final answer with citations and reasoning

## 🏗️ Architecture

```
User Query
    ↓
┌─────────────────────────────────┐
│ Node 1: QUERY ANALYZER          │
│ - Breaks down the query         │
│ - Identifies info needed        │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ Node 2: RETRIEVAL PLANNER       │
│ - Uses vector similarity        │
│ - Selects top 3 relevant files  │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ Node 3: EVIDENCE GATHERER       │
│ - ReAct agent with tools        │
│ - Searches & reads files        │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ Node 4: SYNTHESIS ANALYZER      │
│ - Chain-of-Thought reasoning    │
│ - Generates final answer        │
└─────────────────────────────────┘
```

## 📁 Project Structure

```
langgraph-project/
├── src/
│   ├── agent/
│   │   ├── graph.py          # LangGraph workflow assembly
│   │   ├── nodes.py          # 4 node implementations
│   │   └── state.py          # GraphState definition
│   ├── retrieval/
│   │   ├── vectorstore.py    # FAISS vector store manager
│   │   └── embeddings.py     # Embedding provider wrapper
│   ├── tools/
│   │   └── tools.py          # ReAct agent tools
│   ├── prompts/
│   │   └── prompts.py        # All LLM prompts
│   └── config.py             # Configuration settings
├── files_to_work_with/       # Your .txt documents
├── vectorstore/              # Generated FAISS index
├── main.py                   # CLI entrypoint
├── requirements.txt
└── README.md
```

## ⚙️ Configuration

Edit `src/config.py` to customize:

```python
# Model settings
LLM_MODEL = "mistral"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Retrieval parameters
TOP_K_CHUNKS = 5
CHUNK_SIZE = 500
MAX_FILES_TO_EXAMINE = 3

# Agent parameters
MAX_REACT_ITERATIONS = 5
```

## 🔧 Tools Available to Agent

The Evidence Gatherer agent has access to:

1. **`vector_search_chunks(query, k=5)`**
   - Searches for semantically similar content
   - Returns top-k relevant passages with sources

2. **`read_file_section(filename, start_line=0, num_lines=50)`**
   - Reads specific sections of files for more context
   - Useful when search results need surrounding information

## 📊 Example Queries

Simple factual:
```
What was the Q1 budget?
```

Complex inference:
```
How did customer feedback from December influence our technical priorities in Q1?
```

Multi-document synthesis:
```
What were the main risks and how did they affect budget allocation?
```

## 🎓 How It Works

### 1. Query Analysis (Node 1)
The LLM analyzes your query to extract:
- Information needed
- Time periods mentioned
- Metrics requested
- Whether inference is required

### 2. Retrieval Planning (Node 2)
- Performs semantic search across all documents
- Aggregates relevance scores by file
- LLM selects top-N most relevant files

### 3. Evidence Gathering (Node 3)
- ReAct agent uses tools to search and read
- Iteratively refines search based on findings
- Collects specific evidence with citations

### 4. Synthesis (Node 4)
- Uses Chain-of-Thought reasoning
- Synthesizes evidence from multiple sources
- Performs inference when needed
- Provides answer with citations

## ⏱️ Performance

With Ollama Mistral on a typical machine:
- **Simple queries**: 20-30 seconds
- **Complex queries**: 45-60 seconds

Breakdown:
- Query Analysis: ~5-8 sec
- Retrieval Planning: ~5-8 sec
- Evidence Gathering: ~15-25 sec
- Synthesis: ~10-15 sec

## 🐛 Troubleshooting

**"Ollama connection error"**
- Ensure Ollama is running: `ollama serve`
- Verify Mistral is installed: `ollama pull mistral`

**"No relevant information found"**
- Check that .txt files are in `files_to_work_with/`
- Verify files have readable content
- Try rebuilding vector store (delete `vectorstore/` folder)

**Slow performance**
- Reduce `MAX_REACT_ITERATIONS` in config
- Reduce `TOP_K_CHUNKS` for faster retrieval
- Use a smaller embedding model

## 🔄 Rebuilding Vector Store

The vector store is built on first run. To rebuild:

1. Delete the `vectorstore/` directory
2. Or set `REBUILD_VECTORSTORE = True` in `src/config.py`
3. Run `python main.py`

## 📝 Notes

- First run takes longer due to embedding generation
- Vector store is saved and reused for faster subsequent runs
- The system works entirely locally with Ollama
- All reasoning traces are visible in the console output

## 🎯 Design Principles

1. **Modularity**: Each component is independent and testable
2. **Clarity**: Well-commented code with clear function purposes
3. **Agentic**: Multi-step reasoning with explicit decision points
4. **Accuracy**: Semantic search + ReAct tools ensure relevant retrieval
5. **Transparency**: Chain-of-Thought provides visible reasoning
