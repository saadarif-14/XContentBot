from crewai import Task
from agents import research_agent, knowledge_agent, ideation_agent, writing_agent, editing_agent

def create_tasks(selected_pillar, approved_ideas, draft_input, feedback):
    research_task = Task(
        description=f"Scrape and analyze viral content related to {selected_pillar} on X.com.",
        agent=research_agent,
        expected_output="List of top trending topics related to the selected pillar."
    )

    knowledge_task = Task(
        description="Store and structure insights from research for later content generation.",
        agent=knowledge_agent,
        expected_output="Organized research data for idea generation."
    )
    
    ideation_task = Task(
        description=f"Using the following trending topics, generate 5 ideas:\n{{research_task.output}}\nIdeas must be strictly based on these trends.",
        agent=ideation_agent,
        expected_output="List of 5 ideas strictly based on X trending topics."
    )

    writing_task = Task(
        description=f"Write compelling Twitter threads based on these ideas: {', '.join(approved_ideas)}",
        agent=writing_agent,
        expected_output="Well-structured Twitter threads. Each thread must be start with new line"
    )
    
    editing_task = Task(
        description=f"Refine and enhance the following draft based on this feedback:\nDraft: {draft_input}\nFeedback: {feedback}",
        agent=editing_agent,
        expected_output="Final polished version of the Twitter thread incorporating user feedback."
    )
    
    return [research_task, knowledge_task, ideation_task, writing_task, editing_task]

