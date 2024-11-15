# app/utils/database.py

'''
Security and Professional Enhancements
Modular Design: Separating the code into modules (database.py and authentication.py) enhances readability and maintainability.
Password Security: Passwords are securely hashed using bcrypt before storage.
Database Security: Using parameterized queries to prevent SQL injection.
Scalability: The project structure allows for easy expansion, such as adding more utilities or features.

'''

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'db', 'users.db')

def create_usertable():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users(
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def add_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO users(username, password) VALUES (?, ?)', (username, password))
    conn.commit()
    conn.close()

def get_user_password(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT password FROM users WHERE username = ?', (username,))
    data = c.fetchone()
    conn.close()
    return data

def username_exists(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT 1 FROM users WHERE username = ?', (username,))
    data = c.fetchone()
    conn.close()
    return data is not None
