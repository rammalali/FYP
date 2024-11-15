# app/utils/authentication.py

import bcrypt
from .database import get_user_password

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def verify_password(username, password):
    stored_password = get_user_password(username)
    if stored_password:
        return bcrypt.checkpw(password.encode(), stored_password[0])
    return False
