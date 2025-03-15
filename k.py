import os
import requests
import json
from crewai import Agent, Task, Crew
from langchain.tools import Tool
import litellm
from litellm import completion
from dotenv import load_dotenv
import http.client
from PIL import Image
import urllib.parse 
import re
import string
litellm._turn_on_debug()
load_dotenv()

# API Credentials
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST")  # Ensure this is correctly set in .env
SERPER_API_KEY = os.getenv("SERPER_API_KEY")



# llm = OpenRouterLLM()
class LiteLLMGemini:
    def __init__(self, model_name="gemini/gemini-2.0-flash", max_tokens=600):
        self.model_name = model_name
        self.max_tokens = max_tokens  # Set max tokens limit

    def __call__(self, prompt: str):
        """
        Calls Google Gemini via LiteLLM with a token limit.
        """
        try:
            response = completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens  # Apply token limit
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            return f"❌ Error: {str(e)}"


# Example Usage:
llm = LiteLLMGemini(max_tokens=513)  # Set max tokens dynamically



# -------------- Fetch Trending Topics -----------------


def fetch_trending_topics(topic):
    """Fetch trending topics from X.com (Twitter) API via RapidAPI using extracted keywords."""
    conn = http.client.HTTPSConnection("twitter-aio.p.rapidapi.com")


    def extract_keywords(topic):
        """Extracts keywords using regex instead of NLTK."""
        words = re.findall(r'\b\w+\b', topic.lower())  # Extract words
        stop_words = {"business","the", "is", "on", "a", "and", "of", "to", "in"}  # Basic stopwords set
        keywords = [word for word in words if word not in stop_words]
        return " OR ".join(keywords)

   
    headers = {
        'x-rapidapi-key': RAPIDAPI_KEY,
        'x-rapidapi-host': "twitter-aio.p.rapidapi.com"
    }

    search_query = extract_keywords(topic)

    # Encode query for URL
    encoded_query = urllib.parse.quote(search_query)

    # Construct the API endpoint with the encoded query
    endpoint = f"/search/{encoded_query}?count=20&category=Top&filters=%7B%22since%22%3A%20%222025-03-05%22%7D&includeTimestamp=false"
    print(endpoint)

    conn.request("GET", endpoint, headers=headers)

    res = conn.getresponse()
    data = res.read()
    try:
        json_data = json.loads(data.decode("utf-8"))

        if "entries" in json_data and isinstance(json_data["entries"], list):
            for entry in json_data["entries"]:
                if entry.get("type") == "TimelineAddEntries" and "entries" in entry:
                    for inner_entry in entry["entries"]:
                        if inner_entry.get("entryId") and inner_entry.get("entryId").startswith("tweet-"):
                            if "content" in inner_entry and "itemContent" in inner_entry["content"] and "tweet_results" in inner_entry["content"]["itemContent"] and "result" in inner_entry["content"]["itemContent"]["tweet_results"]:
                                tweet_result = inner_entry["content"]["itemContent"]["tweet_results"]["result"]
                                if "legacy" in tweet_result and "full_text" in tweet_result["legacy"]:
                                    print(tweet_result["legacy"]["full_text"])

    except json.JSONDecodeError:
        print("Error decoding JSON data.")
    except Exception as e:
        print(f"An error occurred: {e}")

    conn.close()
# -------------- Image Generation API Call -----------------


def search_google(query):
    url = "https://serpapi.com/search"
    params = {"q": query, "api_key": SERPER_API_KEY}
    response = requests.get(url, params=params)
    return response.json()

# -------------- Define Tools -----------------
fetch_trending_topics_tool = Tool(
    name="Fetch Trending Topics",
    func= fetch_trending_topics,  # Allows dynamic input

    description="Fetches recent tweets containing the given query."
)



search_google_tool = Tool(
    name="Search Google",
    func=search_google,
    description="Searches Google for the given query and returns the results."
)
# -------------- Define Agents -----------------
# Research Agent
research_agent = Agent(
    name="ResearchAgent",
    role="Scrapes and analyzes viral content on Google using Google Search.",
    goal="Find the best resources on a given topic from Google",
    backstory="An expert in search optimization,research and trend detection.",
    tools=[search_google_tool],
    llm=llm,
    output_type="text",
    verbose=True,
  
)
# Trending Topics Agent
trending_topics_agent = Agent(
    name="TwitterSearchAgent",
    role="Finds recent tweets containing trending topic keywords.",
    goal="Extract relevant tweets containing trending topics.",
    backstory="A social media analyst skilled at finding real-time discussions.",
    llm=llm,
    tools=[fetch_trending_topics_tool],
    output_type="text",
    verbose=True,
  
)

# Writing Agent
writing_agent = Agent(
    name="WritingAgent",
    role="Writes an engaging Tweet based on the topic.",
    goal="Produce high-quality, engaging content.",
    backstory="A skilled writer specializing in social media storytelling by searching from the google search results.",
    output_type="text",
    llm=llm,
    verbose=True,
    
)
# Image Generation Agent


# -------------- Define Tasks -----------------
def create_tasks(selected_pillar):
    """
    Creates research, writing, and image generation tasks.
    """

    research_task = Task(
        description=f"Perform a Google search on the given topic and return key insights related to '{selected_pillar}' on google",
        agent=research_agent,
        expected_output="A summary of the top search results."
    )

    trending_topics_task = Task (
        description=f"Find recent tweets that contain the keywords from the topic: '{selected_pillar}'.",
        agent=trending_topics_agent,
        expected_output="A list of relevant tweets."
    )

    writing_task = Task(
        description="Generate a compelling tweet based on the following research:\n"
                    "{{research_task.output or 'No research data found'}}\n"
                    "and the following trending topics:\n"
                    "{{trending_topics_task.output or 'No trending topics found'}}\n"
                    "\nEnsure the tweet is engaging, concise, and provides valuable insights."
                    "\nSTRICT REQUIREMENT: The tweet MUST NOT exceed 280 characters."
                    "\nUse short, impactful sentences and avoid unnecessary fluff."
                    "\n\nFinally, include a strong Call-To-Action (CTA) promoting Alphabase. "
                    "\nInclude relevant hashtags for better reach.",
        agent=writing_agent,
        expected_output="A well-structured tweet within 280 characters."
    )


   
    return [research_task ,trending_topics_task ,writing_task]

# -------------- Function to Run CrewAI -----------------
def generate_tweet_and_image(selected_pillar):
    """
    Runs CrewAI and returns the generated tweet and image.
    """
    crew = Crew(
        agents=[research_agent, trending_topics_agent, writing_agent],  
        tasks=create_tasks(selected_pillar)
    )

    results = crew.kickoff(inputs={"selected_pillar": selected_pillar})

    print("\n🔍 CrewAI Full Output:", results)  

    tweet_text, image_path = None, None

    if hasattr(results, "tasks_output"):  
        task_outputs = results.tasks_output  
    else:
        print("❌ Error: `tasks_output` not found in CrewOutput.")
        return "⚠️ Error: No valid response from CrewAI.", None

    for idx, task_output in enumerate(task_outputs):
        print(f"\n🔍 Task {idx+1} Output:", task_output)

    if len(task_outputs) > 2 and hasattr(task_outputs[2], "raw"):
        tweet_text = task_outputs[2].raw
    else:
        tweet_text = "⚠️ Error: Tweet could not be generated."

    image_path = "generated_image.jpg"
   
    print("✅ Extracted Tweet:", tweet_text)
    print("✅ Extracted Image Path:", image_path)

    return tweet_text, image_path

