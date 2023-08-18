from fastapi import APIRouter, HTTPException, Depends, status
from app.Models.ChatbotModel import User
from app.Models.ChatLogModel import Chatlog, ChatlogIdModel, find_all_chatlogs, find_messages_by_id
from app.Utils.Auth import get_current_user

from typing import Annotated
router = APIRouter()


@router.post("/find-chatlogs")
def find_all_chatlogs_api(user: Annotated[User, Depends(get_current_user)]):
    print(user.email)
    try:
        return find_all_chatlogs(email=user.email)
    except Exception as e:
        raise e


@router.post("/find_messages_by_id")
def find_messages_by_id_api(model: ChatlogIdModel, user: Annotated[User, Depends(get_current_user)]):
    try:
        return find_messages_by_id(model.logId)
    except Exception as e:
        raise e
