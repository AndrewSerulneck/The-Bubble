from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from uuid import UUID
from typing import List, Optional
from backend.graph_model import NewsNode, NewsGraph

app = FastAPI()
graph = NewsGraph()

# Pydantic models for request/response validation

class StoryIn(BaseModel):
    headline: str
    category: str
    summary: Optional[str] = None
    source: Optional[str] = None

class StoryOut(BaseModel):
    id: str
    headline: str
    category: str
    timestamp: str
    summary: Optional[str] = None
    source: Optional[str] = None

class LinkIn(BaseModel):
    from_id: str
    to_id: str
    explanation: str

# Routes

@app.post("/story", response_model=StoryOut)
def add_story(story: StoryIn):
    node = NewsNode(
        headline=story.headline,
        category=story.category,
        summary=story.summary,
        source=story.source
    )
    graph.add_story(node)
    return node.to_dict()

@app.get("/story/{node_id}", response_model=StoryOut)
def get_story(node_id: str):
    try:
        node = graph.get_story(node_id)
        return node
    except KeyError:
        raise HTTPException(status_code=404, detail="Story not found")

@app.post("/link")
def add_link(link: LinkIn):
    if link.from_id not in graph.graph.nodes or link.to_id not in graph.graph.nodes:
        raise HTTPException(status_code=404, detail="One or both story IDs not found")
    graph.add_causal_link(link.from_id, link.to_id, link.explanation)
    return {"message": "Link added successfully"}

@app.get("/story/{node_id}/links", response_model=List[StoryOut])
def get_connected_stories(node_id: str):
    if node_id not in graph.graph.nodes:
        raise HTTPException(status_code=404, detail="Story not found")

    connected_ids = graph.get_connected_stories(node_id)
    return [graph.get_story(n) for n in connected_ids]
