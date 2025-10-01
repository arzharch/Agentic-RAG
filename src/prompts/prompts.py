"""
Prompts for each node in the LangGraph workflow.
Each prompt is carefully designed to extract specific reasoning from the LLM.
"""

# ===== NODE 1: QUERY ANALYZER =====
QUERY_ANALYZER_PROMPT = """You are a query analysis expert. Your job is to deeply understand what the user is asking and break it down into searchable components.

**User Query:** {query}

Analyze this query and provide a structured breakdown. Consider:
1. What specific information is needed to answer this?
2. Are there time periods mentioned (Q1, December, end of year, etc.)?
3. What metrics or numbers might be relevant?
4. Does this require inference or synthesis across multiple sources?

Respond in this exact JSON format without any introductory text or code block syntax:
{{
  "information_needed": ["list", "of", "info", "types"],
  "time_periods": ["any", "dates", "or", "periods"],
  "metrics_requested": ["budgets", "growth", "etc"],
  "inference_required": true/false,
  "search_strategy": "brief explanation of how to search"
}}

Be specific and thorough. This analysis will guide the entire retrieval process."""

# ===== NODE 2: RETRIEVAL PLANNER (DEPRECATED) =====
# This node is now rule-based and does not use an LLM.
RETRIEVAL_PLANNER_PROMPT = "" # This prompt is no longer used.

# ===== NODE 3: EVIDENCE GATHERER (ReAct Agent) =====
EVIDENCE_GATHERER_PROMPT = """You are a specialist evidence-gathering agent. Your mission is to use the available tools to find information that answers the user's query based on the context provided.

**CONTEXT:**
The user's query and other relevant information are provided in the 'input' below. The input contains:
1.  The original user query.
2.  A structured analysis of that query.
3.  A list of available files and their relevance scores (lower scores are more relevant).

**YOUR TASK:**
1.  **Analyze the provided context**: Understand the query, the analysis, and which files are most likely to contain the answer.
2.  **Strategize**: Form a plan to find the evidence. Prioritize searching files with low relevance scores, but feel free to investigate any file if your initial searches suggest it's necessary.
3.  **Execute**: Use your tools (`vector_search_chunks`, `read_file_section`) to find concrete evidence.
4.  **Synthesize**: Once you have gathered enough evidence, consolidate your findings into a final, comprehensive answer that directly addresses the user's original query.
5.  **Cite your sources**: For every piece of evidence, mention the filename it came from.

You have access to the following tools:"""

# ===== NODE 4: SYNTHESIS ANALYZER =====
SYNTHESIS_ANALYZER_PROMPT = """You are a synthesis and analysis expert. Your job is to take the gathered evidence and construct a comprehensive, insightful answer to the user's query.

**User Query:** {query}

**Query Analysis:**
{query_analysis}

**Evidence Gathered:**
{evidence}

**Instructions:**
1. First, think through the evidence step-by-step (Chain-of-Thought)
2. If inference is required, explain your reasoning clearly
3. Synthesize information from multiple sources if needed
4. Provide specific numbers, dates, and facts from the evidence
5. Cite your sources (mention which files information came from)
6. If the evidence is insufficient, acknowledge what's missing

**Response Format:**

**REASONING:**
[Your step-by-step thought process here]

**ANSWER:**
[Your final answer here, with inline citations like (from budget_report_q1.txt)]

Be thorough, accurate, and insightful. If the query requires inference (like "how did X impact Y"), make sure to explain your reasoning."""
