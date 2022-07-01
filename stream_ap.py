from calendar import day_abbr
from matplotlib.pyplot import xlabel
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt


df_data = pd.read_csv("")

with st.sidebar:
    
    app_mode = option_menu(None, ["Home", "Sign in", "Create an Account", "Logout"],
                        icons=['house', 'person-circle', 'person-plus', 'lock'],
                        menu_icon="app-indicator", default_index=0,
                        styles={
        "container": {"padding": "5!important", "background-color": "#f0f2f6"},
        "icon": {"color": "orange", "font-size": "28px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#2C3845"}