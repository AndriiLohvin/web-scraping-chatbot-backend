from pydantic import BaseModel
from app.Database import db
import os
from dotenv import load_dotenv

load_dotenv()
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

ChatbotsDB = db.chatbots
