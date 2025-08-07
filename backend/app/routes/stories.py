import os
import requests
from flask import Blueprint, jsonify
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

bp = Blueprint("stories", __name__)
load_dotenv()

NYT_KEY = os.getenv("NYT_API_KEY")
GUARDIAN_KEY = os.getenv("GUARDIAN_API_KEY")

def fetch_nyt():
    url = f"https://api.nytimes.com/svc/topstories/v2/world.json?api-key={NYT_KEY}"
    res = requests.get(url).json()
    stories = []
    for article in res.get("results", [])[:20]:
        story = {
            "id": article["url"],
            "headline": article["title"],
            "lede": article.get("abstract", ""),
            "nutgraf": article.get("abstract", ""),
            "source": "NYT",
            "tags": article.get("des_facet", []) + article.get("geo_facet", []),
        }
        stories.append(story)
    return stories

def fetch_guardian():
    url = f"https://content.guardianapis.com/search?api-key={GUARDIAN_KEY}&show-fields=trailText,bodyText&page-size=20"
    res = requests.get(url).json()
    stories = []
    for result in res.get("response", {}).get("results", []):
        fields = result["fields"]
        story = {
            "id": result["webUrl"],
            "headline": result["webTitle"],
            "lede": fields.get("trailText", ""),
            "nutgraf": fields.get("bodyText", "")[:500],
            "source": "Guardian",
            "tags": [result.get("sectionName", "News")],
        }
        stories.append(story)
    return stories

def create_graph(stories):
    nodes = stories
    links = []

    texts = [s["lede"] + " " + s["nutgraf"] for s in stories]
    tfidf = TfidfVectorizer(stop_words='english').fit_transform(texts)
    sim_matrix = cosine_similarity(tfidf)

    for i in range(len(stories)):
        for j in range(i + 1, len(stories)):
            score = sim_matrix[i][j]
            if score > 0.15:
                links.append({
                    "source": stories[i]["id"],
                    "target": stories[j]["id"],
                    "strength": score
                })

    return {"nodes": nodes, "links": links}

@bp.route("/stories", methods=["GET"])
def get_stories():
    nyt = fetch_nyt()
    guardian = fetch_guardian()
    all_stories = nyt + guardian
    graph = create_graph(all_stories)
    return jsonify(graph)

