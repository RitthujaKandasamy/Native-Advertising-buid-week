import streamlit as st
# import torch
import base64
# import spacy 
# from model import Classifier
from streamlit_option_menu import option_menu
import streamlit.components.v1 as stc


# from prediction import prediction  
import torch 
import spacy 
from torchtext.vocab import FastText
import requests 
from bs4 import BeautifulSoup 
from model import Classifier


###########################################################################################################
###########################################################################################################


nlp = spacy.load("en_core_web_sm") 
fasttext = FastText("simple")

def preprocessing(sentence):
    doc = nlp(sentence)
    tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_stop]
    return tokens

def token_encoder(token, vec):
    if token == "<pad>":
        return 1
    else:
        try:
            return vec.stoi[token]
        except:
            return 0

def encoder(tokens, vec):
    return [token_encoder(token, vec) for token in tokens]

def padding(list_of_indexes, max_seq_len, padding_index=1):
    output = list_of_indexes + (max_seq_len - len(list_of_indexes))*[padding_index]
    return output[:max_seq_len]



def prediction(url):
    emb_dim = 300
    max_seq_length = 64 

    model = Classifier(max_seq_length, 300, 128)
    # model = Classifier_3(max_seq_length, 300, 128)

    train_modeled = torch.load('C:/Users/zorve/OneDrive/Documents/GitHub/Native-Advertising-buid-week/new_trained_model')
    # train_modeled = torch.load('new_trained_model_2')

    model_state = train_modeled['model_state']

    model = Classifier(max_seq_length, 300, 128)
    model.load_state_dict(model_state)

    fasttext = FastText("simple")  

    request = requests.get(url)
    soup = BeautifulSoup(request.text, 'html.parser')

    results = soup.find_all('p')
    reviews = [result.text for result in results]

    reviews = [result.text.replace('\n', ' ').replace(',', '').replace('\t', ' ').replace("'", ' ').replace("*", ' ').replace("'", ' ').strip().replace('   ', '').replace('  ', ' ') for result in results]


    one_sentence = ''

    for item in reviews:
        item = item.strip().replace('\n', ' ').replace(',', '')
        one_sentence = one_sentence + item + ' '


    features = padding(encoder(preprocessing(one_sentence), fasttext), max_seq_length) 


    embeddings = [fasttext.vectors[el] for el in features]
    inputs = torch.stack(  embeddings )


    with torch.no_grad():
        prediction = model.forward(inputs.resize_(inputs.size()[0], 64 * emb_dim))
        prediction_classes = torch.argmax(prediction, dim=1)

    prediction_max = int(prediction_classes.max())


    if prediction_max == 0: 
        results_outcome = 'It is TRULY NATIVE.'
    else: 
        results_outcome = 'It is NOT TRULY NATIVE.'

    return results_outcome

###########################################################################################################
###########################################################################################################




# to store the data
# st.set_option('deprecation.showfileUploaderEncoding', False)
# @st.cache(allow_output_mutation=True)



# # load model
# def load_model():

#     model_state = torch.load('')
#     model = Classifier(64, 300, 64)
#     model.load_state_dict(model_state)
#     return model


# def predict(model, text):
#     nlp = spacy.load("en_core_web_sm") 
#     doc = nlp(text)
#     tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_stop]

#     sentences = tokens(text)

#     model.eval()
#     with torch.no_grad():
#         sentences.resize_(sentences.size()[0], 64 * 300)

#         prediction_test = model.forward(sentences)
#         prediction_class = torch.argmax(prediction_test, dim = 1)

#         if prediction_class == 0:
#             st.write("It is truly native 🏆")
#         else:
#             st.write("It is not truly native.")


# def load_text():
#     # upload_url = st.file_uploader(label = "Upload an a URL")
#     # if upload_url is not None:
#     #     text_data = upload_url
#     #     # st.write(text_data, caption= "Input Text")
#     #     return text_data

#     uploaded_files = st.file_uploader("Upload a URL", accept_multiple_files = True)
#     for uploaded_file in uploaded_files:
#         data = uploaded_file
#         return data

#     else:
#         st.write('Waiting for upload....')
#         return None


 



# Menu
app_mode = option_menu(menu_title=None,
                       options=["HOME", "PREDICTOR", "ABOUT", "LOGOUT"],
                       icons=['house', 'book', 'person-circle', 'lock'],
                       menu_icon = "app-indicator",
                       default_index = 0,
                       orientation = "horizontal",
                       styles={
                           "container": {"padding": "5!important", "background-color": "#f0f2f6"},
                           "icon": {"color": "orange", "font-size": "28px"},
                           "nav-link": {"font-size": "18px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
                           "nav-link-selected": {"background-color": "#2C3845"}
                       })






# Home page
if app_mode == 'HOME':

    st.markdown('\n')
    st.markdown('\n')
    st.markdown('\n')
    st.markdown('\n')

    # title
    HTML_BANNER = """
    <center><div style="background-color:#3a7ff0;padding:10px;border-radius:10px;width:800px">
    <h1 style="color:white;text-align:center;"> Truly Native </h1>
    </div></center>
    """
    stc.html(HTML_BANNER)

    st.markdown('###')

    # Gif from local file
    # file_ = open("Downloads\\HIP1.gif", "rb")
    # file_ = open("HIP1.gif", "rb")

    file_ = open('C:/Users/zorve/OneDrive/Documents/GitHub/Native-Advertising-buid-week/HIP1.gif', "rb")
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
    st.markdown("For unsespecting eyes, native ads can appear as part of the legitimate website content they desperately searched for.")
    st.markdown("However, despite their unpredictable disguise, they are what they are - Advertisements.")
    st.markdown("The goal of this project is to identify such black sheep, I mean to predict if a website contains sponsored content.")
    st.markdown("Here we are provided with a dataset of over 300,000 raw HTML files containing text, links, and downloadable images.")
    st.markdown("...")
    st.markdown('\n')
    st.markdown('\n')
    st.markdown('\n')
    st.markdown('\n')
    

    # Team members
    st.header('Our Team')
    st.markdown("1. Peter Zorve")
    st.markdown("2. Ritthuja")
    st.markdown("3. Islom")




# app
elif app_mode == 'PREDICTOR':

    # title
    st.markdown('\n')
    st.markdown('\n')
    st.markdown('\n')
    st.markdown('\n')

    HTML_BANNER = """
    <center><div style="background-color:#3a7ff0;padding:10px;border-radius:10px;width:800px">
    <h1 style="color:white;text-align:center;"> Truly Native </h1>
    </div></center>
    """
    stc.html(HTML_BANNER)

    # def main():

    #     model = load_model()
    #     text = load_text()
    #     result = st.button('Run on image')

    #     if result:
    #         predict(model, text)

    # if __name__ == '__main__':
    #     main()

    # text_data = load_text()
    user_input = st.text_input("Enter URL link", 'https://www.yelp.com/biz/social-brew-cafe-pyrmont')
    # url = 'https://www.yelp.com/biz/social-brew-cafe-pyrmont'

    st.title(prediction(user_input))



# about app
elif app_mode == 'ABOUT':
    
    st.markdown('\n')
    st.markdown('\n')
    st.markdown('\n')
    st.markdown('\n')

    # title
    HTML_BANNER = """
    <center><div style="background-color:#3a7ff0;padding:10px;border-radius:10px;width:800px">
    <h1 style="color:white;text-align:center;"> Native advertising </h1>
    </div></center>
    """
    stc.html(HTML_BANNER)

    
    # Description
    st.markdown('###')
    st.subheader("Advertisements have been around for a very long time. But they started appearing on the internet just a few years back. But the first generation of internet advertising, Internet advertisement 1.0, was not that popular among website users. To tell the truth, it was actually annoying - The annoying pop-ups blocking the screen, the intrusive advertisements that would prevent the user from scrolling, and the list goes on.")
    st.subheader("But today, internet advertising has evolved. There is a new beast on the menu, and it is called Native Advertising. Native advertisements are nothing like their predecessors. They can blend in smoothly with the website's content, not just optically but thematically as well. So, they don't distract or annoy the user; They don't make the user want to leave the website immediately. As a result, they create a win-win situation for all the parties involved - The publisher, the advertiser, and the user. The publisher doesn't lose the website visitors because the ads are not intrusive to them. The advertiser doesn't lose revenue because of users being forced to click the ad by mistake. The user benefits because he is shown ads that are relevant to his needs.")
    st.markdown('\n')
    st.markdown('\n')
    st.markdown('\n')
    st.markdown('\n')
    
    
    # image
    # st.image("Downloads\\image.jpg")
    # st.image("image.jpg")
    st.image("C:/Users/zorve/OneDrive/Documents/GitHub/Native-Advertising-buid-week/image.jpg")
    # image.jpg




# logout
else:
    
    st.markdown('\n')
    st.markdown('\n')
    st.markdown('\n')
    st.markdown('\n')

    # Gif from local file
    # file_ = open("logout.gif", "rb")
    file_ = open("C:/Users/zorve/OneDrive/Documents/GitHub/Native-Advertising-buid-week/logout.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<center><img src="data:image/gif;base64,{data_url}" alt="test gif"></center>',
        unsafe_allow_html=True
    )







