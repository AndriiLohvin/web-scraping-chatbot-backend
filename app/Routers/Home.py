import shutil
from fastapi import APIRouter, HTTPException, Depends, status, UploadFile, File, Form
from app.Utils.pinecone import get_answer, get_context, train_csv, train_pdf, train_txt, train_url, train_ms_word, delete_all_data, set_prompt, delete_data_by_metadata
from app.Models.ChatbotModel import add_page, add_file, add_new_chatbot, find_all_chatbots, remove_chatbot, find_chatbot_by_id
from app.Models.ChatbotModel import ChatBotIdModel, User
from app.Utils.web_scraping import extract_content_from_url

from app.Utils.Auth import get_current_user
from fastapi.responses import StreamingResponse
from typing import Annotated
import os

router = APIRouter()


supported_file_extensions = [".csv", ".pdf", ".txt", ".doc", ".docx"]


@router.post("/add-new-chatbot")
def add_new_chatbot_api(user: Annotated[User, Depends(get_current_user)], name: str = Form(...)):
    try:
        return add_new_chatbot(email=user["email"], name=name)
    except Exception as e:
        raise e


@router.post("/find-all-chatbots")
def find_all_chatbots_api(user: Annotated[User, Depends(get_current_user)]):
    try:
        return find_all_chatbots(user["email"])
    except Exception as e:
        raise e


@router.post("/remove-chatbot")
def remove_chatbot_api(user: Annotated[User, Depends(get_current_user)], id: str = Form(...)):
    try:
        return remove_chatbot(id, user["email"])
    except Exception as e:
        raise e


@router.post("/find-pages-by-id")
def find_pages(id: ChatBotIdModel, user: Annotated[User, Depends(get_current_user)]):
    try:
        result = find_chatbot_by_id(id.id)
        return result.pages
    except Exception as e:
        raise e


@router.post("/extract-content")
def add_new_chatbot_api(user: Annotated[User, Depends(get_current_user)], link: str = Form(...)):
    try:
        return extract_content_from_url(link)
    except Exception as e:
        raise e


@router.post("/add-page")
def add_page_api(user: Annotated[User, Depends(get_current_user)], id: str = Form(...), url: str = Form(...)):
    try:
        add_page(id, url)
        train_url(url, id)
        return True
    except Exception as e:
        raise e


@router.post("/add-training-file")
def add_training_file_api(user: Annotated[User, Depends(get_current_user)], file: UploadFile = File(...), bot_id: str = Form(...)):
    extension = os.path.splitext(file.filename)[1]
    if extension not in supported_file_extensions:
        raise HTTPException(
            status_code=500, detail="Invalid file type!")
    # print("valid filetype")
    try:
        # save file to server
        directory = "./train-data"
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(f"{directory}/{bot_id}-{file.filename}", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # add training file
        if extension == ".csv":
            train_csv(f"{bot_id}-{file.filename}", bot_id)
        elif extension == ".pdf":
            train_pdf(f"{bot_id}-{file.filename}", bot_id)
        elif extension == ".txt":
            train_txt(f"{bot_id}-{file.filename}", bot_id)
        elif extension == ".docx":
            train_ms_word(f"{bot_id}-{file.filename}", bot_id)
        print("end-training")
        # add_file(file.filename)
    except Exception as e:
        print("training error")
        raise HTTPException(
            status_code=500, detail=e)


@router.post("/similar-context")
def find_similar_context(user: Annotated[User, Depends(get_current_user)], msg: str = Form(...), bot_id: str = Form(...)):
    print("msg: " + str(msg))
    if len(msg.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No Input Message",
        )
    return get_context(msg, bot_id, user["email"])


@router.post("/user-question")
def answer_to_user_question(user: Annotated[User, Depends(get_current_user)], msg: str = Form(...), bot_id: str = Form(...)):
    try:
        return StreamingResponse(get_answer(msg, bot_id, user["email"]), media_type='text/event-stream')
    except Exception as e:
        raise e


@router.post("/clear-database")
def clear_database():
    delete_all_data()
    return True


@router.post("/clear-database-by-metadata")
def clear_database_by_metadata(filename: str = Form(...)):
    delete_data_by_metadata(filename)


@router.post("/set-prompt")
def set_prompt_by_user(prompt: str = Form(...)):
    set_prompt(prompt)
