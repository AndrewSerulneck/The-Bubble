import networkx as nx
import uuid
from datetime import datetime

class NewsNode:
    def __init__(self, headline, category, timestamp=None, summary=None, source=None):
        self.id = str(uuid.uuid4())
        self.headline = headline
        self.category = category
        self.timestamp = timestamp or datetime.utcnow().isoformat()
        self.summary = summary
        self.source = source

    def to_dict(self):
        return {
            "id": self.id,
            "headline": self.headline,
            "category": self.category,
            "timestamp": self.timestamp,
            "summary": self.summary,
            "source": self.source,
        }

class NewsGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_story(self, news_node):
        self.graph.add_node(news_node.id, **news_node.to_dict())

    def add_causal_link(self, from_node_id, to_node_id, explanation):
        self.graph.add_edge(from_node_id, to_node_id, explanation=explanation)

    def get_story(self, node_id):
        return self.graph.nodes[node_id]

    def get_connected_stories(self, node_id):
        return list(self.graph.successors(node_id))

    def get_causal_explanation(self, from_node_id, to_node_id):
        return self.graph.edges[from_node_id, to_node_id]['explanation']
