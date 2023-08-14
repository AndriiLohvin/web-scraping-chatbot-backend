from pydantic import BaseModel
from app.Database import db
from typing import List
from bson import json_util
from bson.objectid import ObjectId

ChatbotsDB = db.chatbots


class ChatBotIdModel(BaseModel):
    id: str


class Chatbot(BaseModel):
    name: str
    email: str
    pages: List
    files: List


class AskQuestionModel(BaseModel):
    usermsg: str
    id: str
    chatlogId: str


class UserForClient(BaseModel):
    username: str
    email: str | None = None


class User(UserForClient):
    password: str


def add_new_chatbot(name: str, email: str):
    print(name, email)
    new_chatbot = Chatbot(name=name, email=email, pages=[], files=[])
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
