# app/app.py

import streamlit as st
import os, time
import pandas as pd
from utils.database import create_usertable, add_user, username_exists
from utils.authentication import hash_password, verify_password
from utils.verify import is_valid_email, is_strong_password
from streamlit_option_menu import option_menu  # Assuming you're using streamlit-option-menu
from utils.frontend.home import render_prediction_section

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
    st.title("Welcome to the App")

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

    # Check if the selected option is different from the current choice in session state
    if selected != st.session_state['choice']:
        st.session_state['choice'] = selected
        st.rerun()  # Immediately rerun to update the page

    choice = st.session_state['choice']

    # Render the selected page
    if choice == "Home":
        st.subheader("Home")
        if st.session_state['logged_in']:
            st.write(f"Welcome back, **{st.session_state['username']}**!")

            if 'home_choice' not in st.session_state:
                st.session_state['home_choice'] = 'Prediction'  # Default item
            
            home_items = ["Prediction", "Forecasting"]
            home_icons = ["predict", "forecast"]
            home_selected = option_menu(
                menu_title='',  # Leave empty to remove the menu title
                options=home_items,
                icons=home_icons,
                menu_icon='cast',
                default_index=home_items.index(st.session_state['home_choice']),
                orientation='horizontal',
            )

            if home_selected != st.session_state['home_choice']:
                st.session_state['home_choice'] = home_selected
                st.rerun()
            home_choice = st.session_state['home_choice']

            if home_choice == "Prediction":
                render_prediction_section()
            
        else:
            st.write("Welcome to the application. Please navigate using the menu.")

    elif selected == "Login":
        if st.session_state['logged_in']:
            st.success(f"You are already logged in as {st.session_state['username']}")
        else:
            st.subheader("Login")
            email = st.text_input("Email Address")
            password = st.text_input("Password", type='password')

            if st.button("Login"):
                if not is_valid_email(email):
                    st.error("Please enter a valid email address.")
                elif verify_password(email, password):
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = email
                    st.success(f"Logged in as {email}")

                    time.sleep(1)  # Simulate a delay
                    st.session_state['choice'] = "Home"
                    st.rerun()  # Refresh the page
                    # Optionally, redirect to Home
                    # selected = "Home"
                else:
                    st.error("Invalid email or password")

    elif selected == "Sign Up":
        if st.session_state['logged_in']:
            st.warning("You are already logged in.")
        else:
            st.subheader("Create a New Account")
            new_email = st.text_input("Email Address")
            new_password = st.text_input("Password", type='password')

            if st.button("Sign Up"):
                if not is_valid_email(new_email):
                    st.error("Please enter a valid email address.")
                elif not is_strong_password(new_password):
                    st.error("Password must be at least 8 characters long and include uppercase letters, "
                             "lowercase letters, numbers, and special characters.")
                elif username_exists(new_email):
                    st.warning("An account with this email already exists. Please use another email.")
                else:
                    hashed_pwd = hash_password(new_password)
                    add_user(new_email, hashed_pwd)
                    st.success("Account created successfully! Please log in.")
                    time.sleep(1)  # Simulate a delay
                    st.session_state['choice'] = "Login"
                    st.rerun()  # Refresh the page

    elif selected == "Logout":
        if st.session_state['logged_in']:
            st.session_state['logged_in'] = False
            st.session_state['username'] = ''
            st.success("You have been logged out.")
            time.sleep(1)  # Simulate a delay
            st.session_state['choice'] = "Home"
            st.rerun()  # Refresh the page
        else:
            st.warning("You are not logged in.")

    elif selected == "About":
        st.subheader("About")
        st.write("This is a application built with Streamlit.")
        st.write("It demonstrates user authentication with email validation and password strength checks.")

if __name__ == '__main__':
    main()
