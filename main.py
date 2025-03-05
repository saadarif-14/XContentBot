
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from crewai import Crew  # Ensure crewai is installed
from tasks import create_tasks
from agents import research_agent, knowledge_agent, ideation_agent, writing_agent, editing_agent
import logging
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CrewRequest(BaseModel):
    selected_pillar: str
    approved_ideas: Optional[List[str]] = []
    draft_input: Optional[str] = ""
    feedback: Optional[str] = ""

def run_crew(selected_pillar, approved_ideas=[], draft_input="", feedback=""):
    logging.info("🚀 Starting Crew Execution")

    approved_ideas = approved_ideas or []
    tasks = create_tasks(selected_pillar, approved_ideas, draft_input, feedback)

    crew = Crew(
        agents=[research_agent, knowledge_agent, ideation_agent, writing_agent, editing_agent],
        tasks=tasks
    )

    results = crew.kickoff()

    logging.info(f"✅ Crew Execution Completed. Results:\n{results}")

    return results

@app.post("/run_crew")
async def execute_crew(request: CrewRequest):
    try:
        results = run_crew(
            selected_pillar=request.selected_pillar,
            approved_ideas=request.approved_ideas,
            draft_input=request.draft_input,
            feedback=request.feedback
        )
        return {"results": results}
    except Exception as e:
        logging.error(f"Error running CrewAI: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing request")
