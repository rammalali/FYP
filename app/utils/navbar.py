# app/utils/navbar.py

from streamlit_option_menu import option_menu
import streamlit as st

def st_navbar(menu_options, icons, default_index=0):
    selected = option_menu(
        menu_title='',  # No title for the navbar
        options=menu_options,
        icons=icons,
        menu_icon='cast',
        default_index=default_index,
        orientation='horizontal',
        styles={
            'container': {'padding': '0!important', 'background-color': '#f0f2f6'},
            'nav-link': {'font-size': '18px', 'margin': '0px', '--hover-color': '#eee'},
            'nav-link-selected': {'background-color': '#4CAF50'},
        }
    )
    return selected
