import os
import tweepy
from crewai import Agent
from langchain.tools import Tool
from litellm import completion 

import http.client
import json




os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY")

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST")  # Replace with the correct host for the API

class OpenRouterLLM:
    def __init__(self, model_name="openrouter/google/gemini-2.0-flash-lite-001"):
        self.model_name = model_name

    def __call__(self, prompt: str):
        response = completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["choices"][0]["message"]["content"]  


llm = OpenRouterLLM()




def fetch_trending_topics():
    """Fetch trending topics from the Twitter API."""
    conn = http.client.HTTPSConnection(RAPIDAPI_HOST)
    headers = {
        'x-rapidapi-key': RAPIDAPI_KEY,
        'x-rapidapi-host': RAPIDAPI_HOST
    }
    
    conn.request("GET", "/trends/23424922", headers=headers)
    res = conn.getresponse()

    try:
        data = json.loads(res.read().decode("utf-8"))
        trends = [trend["name"] for trend in data[0]["trends"]]
        print("🔥 Trending Topics Fetched:", trends)
        return trends[:10]  # Return top 10 trending topics
    except Exception as e:
        return [f"Error fetching trends: {str(e)}"]

fetch_trending_topics_tool = Tool(
    name="Fetch Trending Topics",
    func=fetch_trending_topics,
    description="Fetches the top trending topics from X.com (Twitter) in real-time."
)

# Research Agent
research_agent = Agent(
   
    name="ResearchAgent",
    role="Scrapes and analyzes viral content on X.com.",
    goal="Identify trending topics and engagement metrics.",
    backstory="Expert in social media analytics and trend detection.",
    tools=[fetch_trending_topics_tool],
    llm=llm,
    output_type="text",
    expected_output="List of trending topics from X.com. Ensure this tool is executed before proceeding."
)

knowledge_agent = Agent(
    name="KnowledgeBaseAgent",
    role="Stores and organizes research insights for better content generation.",
    goal="Maintain a repository of high-impact social media trends.",
    backstory="A data-driven strategist who compiles valuable social media insights.",
    llm=llm,
    output_type="text"
)

# Ideation Agent
def generate_ideas(selected_pillar):
    """Generate 5 viral ideas based on trending topics."""
    trends = fetch_trending_topics()
    if not trends:
        return ["Error: No trending topics found."]
    return llm(f"Generate exactly 5 ideas based on these trending topics: {', '.join(trends)}. The ideas must be strictly related to the {selected_pillar} category.")

ideation_agent = Agent(
    name="IdeationAgent",
    role="Generates creative and engaging ideas strictly from research insights.",
    goal="Produce 5 total compelling ideas directly based on X trending topics.",
    backstory="A creative strategist who curates viral-worthy ideas from trending discussions.",
    llm=llm,
    tools=[Tool(
        name="Generate Ideas",
        func=lambda selected_pillar: generate_ideas(selected_pillar),
        description=(
            "Using the 5 top trending topics extracted from X, generate exactly 5 engaging content ideas. "
            "Ensure that you do NOT generate 5 ideas per topic but one idea per topic, but rather 5 ideas in total across all trends."
        ),
    )],
    output_type="text and every idea should be start with new line"
)
writing_agent = Agent(
    name="WritingAgent",
    role="Writes an engaging Twitter thread based on the approved idea.",
    goal="Produce high-quality, engaging content.",
    backstory="A skilled writer specializing in social media storytelling.",
    llm=llm,
    output_type="text"
)

editing_agent = Agent(
    name="EditorAgent",
    role="Refines and polishes the draft based on user feedback.",
    goal="Ensure clarity, impact, and engagement.",
    backstory="A meticulous editor with an eye for impactful messaging.",
    llm=llm,
    output_type="text"
)
