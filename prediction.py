
import torch
import spacy 
from model import Classifier, Classifier_3
# from helper_functions import padding, encoder, preprocessing
from torchtext.vocab import FastText
import requests 
from bs4 import BeautifulSoup 

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

###########################################################################################################
###########################################################################################################


def prediction(url):
    emb_dim = 300
    max_seq_length = 64 

    model = Classifier(max_seq_length, 300, 128)
    # model = Classifier_3(max_seq_length, 300, 128)

    train_modeled = torch.load('new_trained_model')
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


    if prediction_max == 1: 
        results_outcome = 'the website is SPONSORED'
    else: 
        results_outcome = 'the website is NOT SPONSORED'

    return results_outcome

print(prediction('https://www.yelp.com/biz/social-brew-cafe-pyrmont'))



