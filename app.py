#Main dependencies:
import pandas as pd
import numpy as np

#Streamlit dependencies:
import streamlit as st
from streamlit_option_menu import option_menu

#Custom dependencies:
import md_functions as md

def home_page():
    st.title('Superstore Marketing Data Analysis')

menu_styling_css ={
        "container": {"padding": "25px 50px 75px 100px", "background-color": "#f7f7f7"},
        "icon": {"color": "#627cd1", "font-size": "25px"}, 
        "nav-link": {"font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "#6f7178"},
        "nav-link-selected": {"background-color": "#5f78b8"},
}

with st.sidebar:
    selected = option_menu(menu_title="Main Menu",
                           options = ["Home"],
                           icons=["house"], #Bootsdrap Icons: https://icons.getbootstrap.com/ 
                           menu_icon="list",
                           default_index=0,
                           styles=menu_styling_css #See the css dictionary above)
                        )

