import streamlit as st
import pandas as pd
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from routes.displacement import elastic_displacement, plot_displacement_high_res

def acceleration():
    st.title("Acceleration")
    st.write("This section is under development.")

def permanent_settlement():
    st.title("Permanent Settlement")
    st.write("This section is under development.")

def dynamic_stiffness():
    st.title("Dynamic Stiffness")
    st.write("This section is under development.")

# Navigation Bar
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to:", [
    "Elastic Displacement",
    "Acceleration",
    "Permanent Settlement",
    "Dynamic Stiffness"
])

if option == "Elastic Displacement":
    elastic_displacement()
elif option == "Acceleration":
    acceleration()
elif option == "Permanent Settlement":
    permanent_settlement()
elif option == "Dynamic Stiffness":
    dynamic_stiffness()
