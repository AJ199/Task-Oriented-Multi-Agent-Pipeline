# app.py
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from graph_pipeline import build_pipeline

graph = build_pipeline()

app = FastAPI(title="Task-Oriented Multi-Agent Pipeline")


class PipelineRequest(BaseModel):
    query: str


class PipelineResponse(BaseModel):
    query: str
    retrieved_docs: list[str] | None = None
    validated_docs: list[str] | None = None
    answer: str | None = None


@app.post("/pipeline", response_model=PipelineResponse)
async def run_pipeline(req: PipelineRequest):
    """
    Call: POST /pipeline { "query": "..." }
    """
    result = graph.invoke({"query": req.query})
    return PipelineResponse(
        query=req.query,
        retrieved_docs=result.get("retrieved_docs"),
        validated_docs=result.get("validated_docs"),
        answer=result.get("answer"),
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
