from pydantic import BaseModel, root_validator


class SignInModel(BaseModel):
    email: str
    password: str

# class SignUpModel(SignInModel):
#     username: str
#     confirm_password: str

#     @root_validator()
#     def verify_password_match(cls, values):
#         password = values.get("password")
#         confirm_password = values.get("confirm_password")
#         if password != confirm_password:
#             raise ValueError("The two passwords did not match.")
#         return values


class User(BaseModel):
    email: str
