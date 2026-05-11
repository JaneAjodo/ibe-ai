from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import io
from src.core.security import verify_api_key
from src.core.logging import setup_logger
from src.models.schemas import IngestResponse
from src.utils.vector_store import get_vector_store
from src.tools.agent_tools import set_dataframe

router = APIRouter(prefix="/ingest", tags=["Ingestion"])
logger = setup_logger("ingest_router")

def _row_to_document(row: pd.Series) -> str:
    """Convert a DataFrame row into a rich text document for embedding."""
    return f"""Policy ID: {row['Policy_ID']}
Client: {row['Client_Name']}, Age: {row['Age']}, Gender: {row['Gender']}
Occupation: {row['Occupation']}, Region: {row['Region']}
Policy Type: {row['Policy_Type']}
Policy Period: {row['Policy_Start_Date']} to {row['Policy_End_Date']}
Premium: ₦{row['Premium_Amount_NGN']:,.0f} | Coverage: ₦{row['Coverage_Amount_NGN']:,.0f}
Claim Status: {row['Claim_Status']} | Claim Amount: ₦{row['Claim_Amount_NGN']:,.0f}
Claim Date: {row['Claim_Date']}
Risk Score: {row['Risk_Score']}/10
Agent: {row['Agent_Name']}
Notes: {row['Notes']}"""

@router.post("/", response_model=IngestResponse, summary="Ingest Excel insurance data")
async def ingest_excel(
    file: UploadFile = File(..., description="Excel file (.xlsx) with insurance policy data"),
    _: str = Depends(verify_api_key)
):
    """
    Ingest an Excel file containing insurance policy records.
    
    Each row is converted to a rich text document, embedded using Gemini,
    and stored in ChromaDB for semantic retrieval.
    
    The structured DataFrame is also kept in memory for analytics queries.
    """
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Only Excel files (.xlsx, .xls) are accepted.")

    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        
        required_cols = ["Policy_ID", "Client_Name", "Policy_Type", "Region", "Claim_Status"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Missing required columns: {missing}"
            )

        # Fill nulls
        df = df.fillna("N/A")
        
        # Store DataFrame for analytics tool
        set_dataframe(df)
        
        # Build documents and metadata
        documents, metadatas, ids = [], [], []
        for _, row in df.iterrows():
            doc = _row_to_document(row)
            meta = {
                "policy_id": str(row["Policy_ID"]),
                "client_name": str(row["Client_Name"]),
                "policy_type": str(row["Policy_Type"]),
                "region": str(row["Region"]),
                "claim_status": str(row["Claim_Status"]),
                "risk_score": str(row.get("Risk_Score", "N/A")),
            }
            documents.append(doc)
            metadatas.append(meta)
            ids.append(str(row["Policy_ID"]))

        vs = get_vector_store()
        vs.clear()
        vs.add_documents(documents, metadatas, ids)
        
        logger.info(f"Ingested {len(df)} records from {file.filename}")

        return IngestResponse(
            status="success",
            message=f"Successfully ingested {len(df)} policy records from '{file.filename}'",
            records_ingested=len(df),
            collection_name="ibe_insurance_policies"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
