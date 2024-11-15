# app/app.py

import streamlit as st
import os, time
from utils.database import create_usertable, add_user, username_exists
from utils.authentication import hash_password, verify_password
from streamlit_option_menu import option_menu  # Assuming you're using streamlit-option-menu

# Apply custom CSS (optional)
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

css_file = os.path.join(os.path.dirname(__file__), '.', 'styles', 'styles.css')
local_css(css_file)

# Initialize Database
create_usertable()

# Main application with session management and redirection
def main():
    st.title("Welcome to the Secure App")

    # Initialize session state variables
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        st.session_state['username'] = ''

    if 'choice' not in st.session_state:
        st.session_state['choice'] = 'Home'  # Default page

    # Define the menu items and icons
    if st.session_state['logged_in']:
        menu_items = ["Home", "Logout", "About"]
        icons = ["house", "box-arrow-right", "info-circle"]
    else:
        menu_items = ["Home", "Login", "Sign Up", "About"]
        icons = ["house", "box-arrow-in-right", "person-plus", "info-circle"]

    # Create the navigation bar using session state
    selected = option_menu(
        menu_title='',  # Leave empty to remove the menu title
        options=menu_items,
        icons=icons,
        menu_icon='cast',
        default_index=menu_items.index(st.session_state['choice']),
        orientation='horizontal',
    )

    # Update the choice in session state
    st.session_state['choice'] = selected
    choice = st.session_state['choice']

    # Render the selected page
    if choice == "Home":
        st.subheader("Home")
        if st.session_state['logged_in']:
            st.write(f"Welcome back, **{st.session_state['username']}**!")
            # Protected content can be added here
        else:
            st.write("Welcome to the secure application. Please navigate using the menu.")

    elif choice == "Login":
        if st.session_state['logged_in']:
            st.success(f"You are already logged in as {st.session_state['username']}")
            # Redirect to Home page
            st.session_state['choice'] = "Home"
            # st.rerun()
        else:
            st.subheader("Login")

            username = st.text_input("Username")
            password = st.text_input("Password", type='password')

            if st.button("Login"):
                if verify_password(username, password):
                    st.success(f"Logged in as {username}")
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    # Redirect to Home page
                    st.session_state['choice'] = "Home"
                    st.rerun()
                else:
                    st.error("Invalid username or password")

    elif choice == "Sign Up":
        if st.session_state['logged_in']:
            st.warning("You are already logged in. Please log out to create a new account.")
            # Redirect to Home page
            st.session_state['choice'] = "Home"
            st.rerun()
        else:
            st.subheader("Create a New Account")

            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type='password')

            if st.button("Sign Up"):
                if username_exists(new_username):
                    st.warning("Username already exists. Please choose another.")
                else:
                    if new_username and new_password:
                        hashed_pwd = hash_password(new_password)
                        add_user(new_username, hashed_pwd)
                        st.success("Account created successfully!")
                        time.sleep(2)
                        st.session_state['choice'] = "Login"
                        st.rerun()
                    else:
                        st.error("Please enter a valid username and password.")

    elif choice == "Logout":
        if st.session_state['logged_in']:
            st.session_state['logged_in'] = False
            st.session_state['username'] = ''
            st.success("You have been logged out.")
            # Redirect to Home page
            st.session_state['choice'] = "Login"
            # st.rerun()
        else:
            st.warning("You are not logged in.")
            # Redirect to Home page
            st.session_state['choice'] = "Login"
            st.rerun()

    elif choice == "About":
        st.subheader("About")
        st.write("This is a secure application built with Streamlit.")
        st.write("It demonstrates user authentication with session management.")

if __name__ == '__main__':
    main()
