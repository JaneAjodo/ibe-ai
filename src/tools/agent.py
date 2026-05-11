"""
Ibe Agent — Router

Reads the user query and decides which tool to invoke:
  - SemanticSearchTool  : client lookups, policy details, qualitative questions
  - AnalyticsTool       : numbers, averages, totals, comparisons, statistics
  - SummaryTool         : summarise groups, overviews, list-all requests
"""

from google import genai
import json
from typing import Dict, Any
from src.core.config import settings
from src.core.logging import setup_logger
from src.models.schemas import ToolUsed
from src.tools.agent_tools import SemanticSearchTool, AnalyticsTool, SummaryTool
from src.utils.vector_store import VectorStore
from src.utils.memory import ConversationMemory

logger = setup_logger("agent")

TOOL_DEFINITIONS = """
You are the Ibe AI routing agent. Decide which tool to use for the user's query.

TOOLS AVAILABLE:
1. semantic_search — use for: specific client queries, policy details, agent names, notes, qualitative questions, "tell me about X", "find policies for Y"
2. analytics — use for: numerical questions, averages, totals, counts, comparisons, statistics, "how many", "what is the average", "which region has the most", "total claims"
3. summary — use for: broad overviews, "summarise all X", "give me an overview of Y", "list all Z policies", pattern questions across multiple records

Respond with ONLY a JSON object in this exact format:
{"tool": "semantic_search" | "analytics" | "summary", "reasoning": "one sentence why"}
"""

class IbeAgent:
    def __init__(self, vector_store: VectorStore, memory: ConversationMemory):
        self.vector_store = vector_store
        self.memory = memory
        
        # Router model
        self.gemini = genai.Client(api_key=settings.GEMINI_API_KEY)
        
        # Tools
        self.tools = {
            ToolUsed.SEMANTIC_SEARCH: SemanticSearchTool(vector_store),
            ToolUsed.ANALYTICS: AnalyticsTool(),
            ToolUsed.SUMMARY: SummaryTool(vector_store),
        }
        
        logger.info("Ibe Agent initialised with 3 tools")

    def _route(self, query: str) -> ToolUsed:
        """Ask the router model which tool to use."""
        try:
            response = self.gemini.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=f"Query: {query}",
                config=genai.types.GenerateContentConfig(
                    system_instruction=TOOL_DEFINITIONS,
                    temperature=0.0,
                    response_mime_type="application/json",
                )
            )
            raw = response.text.strip()
            parsed = json.loads(raw)
            tool_name = parsed.get("tool", "semantic_search")
            reasoning = parsed.get("reasoning", "")
            logger.info(f"Router decision: {tool_name} — {reasoning}")
            
            mapping = {
                "semantic_search": ToolUsed.SEMANTIC_SEARCH,
                "analytics": ToolUsed.ANALYTICS,
                "summary": ToolUsed.SUMMARY,
            }
            return mapping.get(tool_name, ToolUsed.SEMANTIC_SEARCH)
        
        except Exception as e:
            logger.warning(f"Router failed ({e}), defaulting to semantic_search")
            return ToolUsed.SEMANTIC_SEARCH

    def run(self, query: str, session_id: str) -> Dict[str, Any]:
        """
        Full agent pipeline:
        1. Route query to appropriate tool
        2. Retrieve conversation history
        3. Run selected tool
        4. Store exchange in memory
        5. Return structured response
        """
        logger.info(f"Agent received query [{session_id}]: {query[:80]}...")
        
        # Sanitise input
        query = query.strip()
        if len(query) < 3:
            return {
                "answer": "Please provide a more detailed question.",
                "tool_used": ToolUsed.SEMANTIC_SEARCH,
                "confidence": "low",
                "sources": None,
                "session_id": session_id
            }

        # Route
        tool_key = self._route(query)
        tool = self.tools[tool_key]
        
        # Get conversation history
        history = self.memory.format_for_prompt(session_id)
        
        # Run tool
        try:
            result = tool.run(query=query, conversation_history=history)
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {
                "answer": f"I encountered an error processing your request. Please try rephrasing your question.",
                "tool_used": tool_key,
                "confidence": "low",
                "sources": None,
                "session_id": session_id
            }

        # Store in memory
        self.memory.add(session_id, "user", query)
        self.memory.add(session_id, "assistant", result["answer"])
        
        result["session_id"] = session_id
        result["tool_used"] = tool_key
        
        logger.info(f"Agent responded — tool: {tool_key}, confidence: {result.get('confidence')}")
        return result
