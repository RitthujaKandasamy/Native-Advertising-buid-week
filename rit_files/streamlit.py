import streamlit as st
import torch
import base64
from training import model
from streamlit_option_menu import option_menu
import streamlit.components.v1 as stc














# to store the data
# st.set_option('deprecation.showfileUploaderEncoding', False)
# @st.cache(allow_output_mutation=True)



# load model
model.load_state_dict(torch.load('new_trained_model'))



# Menu
app_mode = option_menu(menu_title=None,
                       options=["HOME", "PREDICTOR", "ABOUT", "LOGOUT"],
                       icons=['house', 'app', 'person-circle', 'lock'],
                       menu_icon="app-indicator",
                       default_index=0,
                       orientation="horizontal",
                       styles={
                           "container": {"padding": "5!important", "background-color": "#f0f2f6"},
                           "icon": {"color": "orange", "font-size": "28px"},
                           "nav-link": {"font-size": "18px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
                           "nav-link-selected": {"background-color": "#2C3845"}
                       })






# Home page
if app_mode == 'HOME':

    # title
    st.markdown('\n')
    st.markdown('\n')
    st.markdown('\n')
    st.markdown('\n')

    HTML_BANNER = """
    <center><div style="background-color:#3a7ff0;padding:10px;border-radius:10px;width:800px">
    <h1 style="color:white;text-align:center;">Native Advertising </h1>
    </div></center>
    """
    stc.html(HTML_BANNER)

    st.markdown('###')

    # Gif from local file
    file_ = open("HIP1.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<center><img src="data:image/gif;base64,{data_url}" alt="test gif"></center>',
        unsafe_allow_html=True
    )

    # Description
    st.markdown('###')
    st.header("About this app    -->")
    st.markdown('###')
    st.markdown('The Intel Image Classification contains 25k images of size 150x150 distributed under 6 categories: buildings, forest, glacier, mountain, sea, street.')
    st.markdown('The Train, Test and Prediction data is separated in each zip files. There are around 14k images in Train, 3k in Test and 7k in Prediction.')
    st.markdown("This project is about creating a CNN model able to classify an image. Additionally, the app can receive a folder of mixed images, creates subfolders into which to classify the images.")
    st.markdown("...")
    st.markdown('\n')
    st.markdown('\n')
    st.markdown('\n')
    st.markdown('\n')
    






# # app
# elif app_mode == 'Predictor':

#     # title
#     st.markdown('\n')
#     st.markdown('\n')
#     st.markdown('\n')
#     st.markdown('\n')

#     HTML_BANNER = """
#     <center><div style="background-color:#3a7ff0;padding:10px;border-radius:10px;width:800px">
#     <h1 style="color:white;text-align:center;">Natural Scenes Images Classification </h1>
#     </div></center>
#     """
#     stc.html(HTML_BANNER)

#     def main():

#         model = load_model()
#         categories = load_labels()
#         image = load_image()
#         result = st.button('Run on image')

#         if result:
#             predict(model, categories, image)

#     if __name__ == '__main__':
#         main()





# # logout
# else:

#     # Gif from local file
#     file_ = open("logout.gif", "rb")
#     contents = file_.read()
#     data_url = base64.b64encode(contents).decode("utf-8")
#     file_.close()

#     st.markdown(
#         f'<center><img src="data:image/gif;base64,{data_url}" alt="test gif"></center>',
#         unsafe_allow_html=True
#     )

