import streamlit as st
import psycopg2
from psycopg2 import sql


class Login:
    def __init__(self, db_conn):
        self.conn = db_conn
        self.logged_in = False

    def login(self):
        left, middle, right = st.columns(3)
        with middle:
            st.subheader("Login")
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            col1, col2, col3 = st.columns([2,1,2])
            with col2:
                login_btn = st.button("Login")
            if login_btn:
                if username and password:
                    try:
                        with self.conn.cursor() as cursor:
                            cursor.execute(
                                sql.SQL("SELECT * FROM users WHERE username = %s AND password = %s"),
                                [username, password]
                            )
                            user = cursor.fetchone()
                            if user:
                                st.session_state["logged_in"] = True
                                st.session_state["username"] = username
                                st.success("Login successful!")
                                st.rerun()
                                self.logged_in = True
                            else:
                                st.error("Invalid credentials.")
                    except psycopg2.Error as e:
                        st.error(f"Error: {e}")
                else:
                    st.error("Please fill out all fields.")