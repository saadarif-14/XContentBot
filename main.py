from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from k import generate_tweet_and_image  # Import your CrewAI function
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Define request model
class TweetRequest(BaseModel):
    selected_pillar: str

# Define response model
class TweetResponse(BaseModel):
    tweet_text: str
    image_path: Optional[str] = None

@app.post("/generate_tweet", response_model=TweetResponse)
async def generate_tweet(request: TweetRequest):
    tweet_text, image_path = generate_tweet_and_image(request.selected_pillar)
    return {"tweet_text": tweet_text, "image_path": image_path}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0")
