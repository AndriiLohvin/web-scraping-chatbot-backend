from fastapi import APIRouter, Form
# from app.Models.UserModel import SignInModel, SignUpModel
from app.Utils.Auth import authenticate_user, create_access_token, get_password_hash, UserDB
from fastapi import HTTPException, status
from app.Utils.Auth import get_user

router = APIRouter()


@router.post("/signin")
def signin_for_access_token(email: str = Form(...), password: str = Form(...)):
    user = authenticate_user(email, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.email})
    print(access_token)
    user_to_return = {'username': user.username,
                      'email': user.email, 'hashed_password': user.hashed_password}
    return {"access_token": access_token, "token_type": "bearer", "user": user_to_return}


@router.post("/signup")
def signup(firstname: str = Form(...), lastname: str = Form(...), email: str = Form(...), password: str = Form(...), confirm_password: str = Form(...)):
    username = firstname + lastname
    if password != confirm_password:
        raise ValueError("The two passwords did not match.")
    password_in_db = get_password_hash(password)
    user = get_user(email)
    if not user:
        UserDB.insert_one({"username": username, "email": email,
                           "hashed_password": password_in_db})
        return True
    else:
        return "That email alrealy exist"
        # raise ValueError("That email alrealy exist")
