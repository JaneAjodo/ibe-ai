"""
Ibe AI — Three Agent Tools

SemanticSearchTool  : vector similarity search for policy/client queries
AnalyticsTool       : numerical reasoning over structured DataFrame
SummaryTool         : summarise filtered subsets of records
"""

import pandas as pd
from google import genai
from typing import Dict, Any, Optional, Tuple
from src.core.config import settings
from src.core.logging import setup_logger
from src.utils.vector_store import VectorStore
from src.models.schemas import Source, ToolUsed

logger = setup_logger("tools")

# ── Shared DataFrame (loaded once at ingest time) ──────────────────────────────
_dataframe: Optional[pd.DataFrame] = None

def set_dataframe(df: pd.DataFrame):
    global _dataframe
    _dataframe = df.copy()
    logger.info(f"DataFrame loaded into tools: {len(df)} records")

def get_dataframe() -> Optional[pd.DataFrame]:
    return _dataframe


# ── Tool 1: Semantic Search ────────────────────────────────────────────────────
class SemanticSearchTool:
    """
    Retrieves the most semantically relevant policy records
    for open-ended, narrative, or client-specific queries.
    """
    name = "semantic_search"
    description = "Use for: client lookups, policy details, notes, qualitative questions"

    def __init__(self, vector_store: VectorStore):
        self.vs = vector_store
        self.gemini = genai.Client(api_key=settings.GEMINI_API_KEY)
        self._system_instruction = """You are Ibe, an intelligent insurance data assistant for a Nigerian insurance platform.
Answer questions based ONLY on the policy records provided as context.
Be precise with figures. Always mention Policy IDs when referencing specific records.
If something is not in the context, say so clearly — never fabricate data.
Format currency as ₦X,XXX,XXX."""

    def run(self, query: str, conversation_history: str = "") -> Dict[str, Any]:
        results = self.vs.search(query)

        if not results["documents"][0]:
            return {
                "answer": "No relevant policy records found for your query.",
                "sources": [],
                "confidence": "low"
            }

        context_parts = []
        sources = []

        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            context_parts.append(f"Record {i+1}:\n{doc}")
            relevance = round(1 - dist, 3)
            sources.append(Source(
                policy_id=meta.get("policy_id", "N/A"),
                client_name=meta.get("client_name", "N/A"),
                policy_type=meta.get("policy_type", "N/A"),
                region=meta.get("region", "N/A"),
                relevance_score=relevance
            ))

        context = "\n\n".join(context_parts)
        history_block = f"\nConversation so far:\n{conversation_history}\n" if conversation_history else ""

        prompt = f"""{history_block}
Policy Records Context:
{context}

Question: {query}

Answer based strictly on the records above. Be specific and cite Policy IDs."""

        response = self.gemini.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=self._system_instruction,
                temperature=settings.TEMPERATURE_CHAT,
            )
        )

        top_score = 1 - results["distances"][0][0] if results["distances"][0] else 0
        confidence = "high" if top_score > 0.75 else "medium" if top_score > 0.5 else "low"

        return {
            "answer": response.text.strip(),
            "sources": sources,
            "confidence": confidence,
            "tool_used": ToolUsed.SEMANTIC_SEARCH
        }


# ── Tool 2: Analytics Tool ─────────────────────────────────────────────────────
class AnalyticsTool:
    """
    Performs numerical and statistical reasoning over the full DataFrame.
    Use for aggregations, comparisons, averages, counts, totals.
    """
    name = "analytics"
    description = "Use for: averages, totals, counts, comparisons, rankings, statistics"

    def __init__(self):
        self.gemini = genai.Client(api_key=settings.GEMINI_API_KEY)
        self._system_instruction = """You are Ibe, an insurance data analyst for a Nigerian insurance platform.
You will receive pre-computed statistics from a real dataset.
Present findings clearly with exact figures. Format currency as ₦X,XXX,XXX.
Always contextualise numbers — don't just list them, explain what they mean."""

    def _compute_stats(self, query: str) -> str:
        df = get_dataframe()
        if df is None:
            return "No data available."

        stats = []
        q = query.lower()

        # General dataset overview
        stats.append(f"DATASET OVERVIEW: {len(df)} total policies")
        stats.append(f"Regions covered: {', '.join(df['Region'].unique())}")
        stats.append(f"Policy types: {', '.join(df['Policy_Type'].unique())}")

        # Revenue stats
        stats.append(f"\nPREMIUM STATISTICS:")
        stats.append(f"Total premium revenue: ₦{df['Premium_Amount_NGN'].sum():,.0f}")
        stats.append(f"Average premium: ₦{df['Premium_Amount_NGN'].mean():,.0f}")
        stats.append(f"Highest premium: ₦{df['Premium_Amount_NGN'].max():,.0f}")
        stats.append(f"Lowest premium: ₦{df['Premium_Amount_NGN'].min():,.0f}")

        # Claims stats
        claimed = df[df["Claim_Amount_NGN"] > 0]
        stats.append(f"\nCLAIMS STATISTICS:")
        stats.append(f"Total policies with claims: {len(claimed)}")
        stats.append(f"Total claims value: ₦{claimed['Claim_Amount_NGN'].sum():,.0f}")
        stats.append(f"Average claim amount: ₦{claimed['Claim_Amount_NGN'].mean():,.0f}")
        stats.append(f"Claim status breakdown: {df['Claim_Status'].value_counts().to_dict()}")

        # By policy type
        stats.append(f"\nBY POLICY TYPE:")
        for ptype in df["Policy_Type"].unique():
            sub = df[df["Policy_Type"] == ptype]
            claims_sub = sub[sub["Claim_Amount_NGN"] > 0]
            avg_claim = claims_sub["Claim_Amount_NGN"].mean() if len(claims_sub) > 0 else 0
            stats.append(f"  {ptype}: {len(sub)} policies, avg premium ₦{sub['Premium_Amount_NGN'].mean():,.0f}, avg claim ₦{avg_claim:,.0f}")

        # By region
        stats.append(f"\nBY REGION:")
        for region in df["Region"].unique():
            sub = df[df["Region"] == region]
            stats.append(f"  {region}: {len(sub)} policies, avg risk score {sub['Risk_Score'].mean():.1f}")

        # Risk analysis
        high_risk = df[df["Risk_Score"] >= 7.0]
        stats.append(f"\nRISK ANALYSIS:")
        stats.append(f"High risk clients (score >= 7.0): {len(high_risk)}")
        stats.append(f"Average risk score overall: {df['Risk_Score'].mean():.2f}")
        if len(high_risk) > 0:
            stats.append(f"High risk by region: {high_risk['Region'].value_counts().to_dict()}")

        return "\n".join(stats)

    def run(self, query: str, conversation_history: str = "") -> Dict[str, Any]:
        computed_stats = self._compute_stats(query)
        history_block = f"\nConversation so far:\n{conversation_history}\n" if conversation_history else ""

        prompt = f"""{history_block}
Pre-computed statistics from the Ibe Insurance dataset:
{computed_stats}

Question: {query}

Provide a precise analytical answer using the statistics above."""

        response = self.gemini.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=self._system_instruction,
                temperature=settings.TEMPERATURE_EXTRACTION,
            )
        )

        return {
            "answer": response.text.strip(),
            "sources": None,
            "confidence": "high",
            "tool_used": ToolUsed.ANALYTICS
        }


# ── Tool 3: Summary Tool ───────────────────────────────────────────────────────
class SummaryTool:
    """
    Summarises a filtered subset of records — e.g. all claims in Lagos,
    all high-risk clients, all pending policies.
    """
    name = "summary"
    description = "Use for: summarise all X, overview of Y, list all Z"

    def __init__(self, vector_store: VectorStore):
        self.vs = vector_store
        self.gemini = genai.Client(api_key=settings.GEMINI_API_KEY)
        self._system_instruction = """You are Ibe, an insurance data assistant.
Provide clear, structured summaries of the policy records given.
Group by relevant dimensions. Highlight patterns and anomalies.
Format currency as ₦X,XXX,XXX. Always include record counts."""

    def run(self, query: str, conversation_history: str = "") -> Dict[str, Any]:
        # Retrieve broader set for summary
        results = self.vs.search(query, n_results=8)

        if not results["documents"][0]:
            return {
                "answer": "No records found matching your summary request.",
                "sources": [],
                "confidence": "low"
            }

        context_parts = []
        sources = []

        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            context_parts.append(f"Record {i+1}:\n{doc}")
            sources.append(Source(
                policy_id=meta.get("policy_id", "N/A"),
                client_name=meta.get("client_name", "N/A"),
                policy_type=meta.get("policy_type", "N/A"),
                region=meta.get("region", "N/A"),
                relevance_score=round(1 - dist, 3)
            ))

        context = "\n\n".join(context_parts)
        history_block = f"\nConversation so far:\n{conversation_history}\n" if conversation_history else ""

        prompt = f"""{history_block}
Policy Records:
{context}

Request: {query}

Provide a structured summary. Group findings, identify patterns, flag anything notable."""

        response = self.gemini.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=self._system_instruction,
                temperature=0.4,
            )
        )

        return {
            "answer": response.text.strip(),
            "sources": sources,
            "confidence": "medium",
            "tool_used": ToolUsed.SUMMARY
        }
