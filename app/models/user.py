# app/models/user.py
class User:
    def __init__(self, username: str, email: str):
        self.username = username
        self.email = email

    def get_domain(self) -> str:
        return self.email.split("@")[-1]
