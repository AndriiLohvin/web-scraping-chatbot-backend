from pydantic import BaseModel
from app.Database import db
from typing import List
from bson import json_util
from bson.objectid import ObjectId
from fastapi import Form
ChatbotsDB = db.chatbots


class ChatBotIdModel(BaseModel):
    id: str

class AddNewBotModel(BaseModel):
    name: str = ""
    description: str = ""
    welcomeMessage: str = "Hello friend! How can I help you today?"
    model: str = "gpt-4"
    language: str = "English"
    tone: str = "Friendly"
    format: str = "FAQ"
    style: str = "Friendly"
    length: str = "50 words"
    password: str = ""


class Chatbot(AddNewBotModel):
    email: str = ""
    pages: List = []
    files: List = []

class AskQuestionModel(BaseModel):
    usermsg: str
    id: str
    chatlogId: str


class UserForClient(BaseModel):
    username: str
    email: str | None = None


class User(UserForClient):
    hashed_password: str


def add_new_chatbot(email: str, botmodel: AddNewBotModel):
    new_chatbot = Chatbot(email=email, pages=[], files=[], **botmodel.dict())
    # print(new_chatbot.name, new_chatbot.description, new_chatbot.language)
    result = ChatbotsDB.insert_one(new_chatbot.dict())
    return str(result.inserted_id)


def add_page(id: str, url: str):
    ChatbotsDB.update_one({"_id": ObjectId(id)}, {"$push": {"pages": url}})
    return True


def add_file(id: str, filename: str):
    ChatbotsDB.update_one({"_id": ObjectId(id)}, {
                          "$push": {"files": filename}})
    return True


def find_chatbot_by_id(id: str):
    result = ChatbotsDB.find_one({"_id": ObjectId(id)})
    return Chatbot(**result)


def find_all_chatbots(email: str):
    result = ChatbotsDB.find({"email": email})
    all_bots = []
    for bot in result:
        bot["_id"] = str(bot["_id"])
        all_bots.append(bot)
    for bot in all_bots:
        print(bot)
    return all_bots


def remove_chatbot(id: str, email: str):
    ChatbotsDB.delete_one({"_id": ObjectId(id), "email": email})
    return True
