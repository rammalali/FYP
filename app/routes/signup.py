import streamlit as st
import psycopg2
from psycopg2 import sql

class Signup:
    def __init__(self, db_conn):
        self.conn = db_conn

    def create_user_table(self):
        with self.conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    password VARCHAR(255) NOT NULL
                )
            """)
            self.conn.commit()

    def signup(self):
        left, middle, right = st.columns(3)
        with middle:
            st.subheader("Sign Up")
            username = st.text_input("Username", key="signup_username")
            password = st.text_input("Password", type="password", key="signup_password")
            col1, col2, col3 = st.columns([2,1,2])
            with col2:
                signup_btn = st.button("Sign Up")
            if signup_btn:
                if username and password:
                    try:
                        with self.conn.cursor() as cursor:
                            cursor.execute(
                                sql.SQL("INSERT INTO users (username, password) VALUES (%s, %s)"),
                                [username, password]
                            )
                            self.conn.commit()
                            st.success("User registered successfully!")
                    except psycopg2.Error as e:
                        st.error(f"Error: {e}")
                else:
                    st.error("Please fill out all fields.")
