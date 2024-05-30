from openai import OpenAI
from flask import Flask, request, jsonify, render_template
import os
import spacy
import numpy as np
import warnings

# gensim: simple_preprocess for easy tokenization & convert to a python list;
# also contain a list of common stopwords
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS as swords

# nltk Lemmatize and stemmer, for lemmatization and stem, I will talk about it later
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.porter import *

# You might need to run the next two line if you don't have those come with your NLTK package
# nltk.download('wordnet')
# nltk.download('stopwords')
from nltk.corpus import stopwords as st
from nltk.corpus import wordnet
from nltk.corpus import stopwords

warnings.filterwarnings("ignore")
nlp = spacy.load("en_core_web_sm")

open_AI_key = os.environ.get("OPENAI")
client = OpenAI(api_key=open_AI_key)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/ask', methods=['POST'])
def ai_response():

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Describe Pokemon Pikachu"}],
        stream=True,
    )

    description = []
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            description.append(chunk.choices[0].delta.content)

    string_description = "".join(description)
    structured_data = extract_structured_data(string_description)

    return jsonify({'response': string_description, 'structured_data': structured_data})


def extract_structured_data(text):
    data = [text]

    # Remove Emails, web links
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]  # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]  # Remove distracting single quotes
    data = [re.sub("\'", '', sent) for sent in data]  # Gensim simple_preprocess function can be your friend with tokenization

    def sent_to_words(sentences):
        for sentence in sentences:
            yield gensim.utils.simple_preprocess(str(sentence), deacc=True)  # deacc=True removes punctuations

    data_words = list(sent_to_words(data))
    print(data_words)

    # Gensim stopwords list
    stop_words = st.words('english')
    # Expand by adding NLTK stopwords list
    stop_words.append(stopwords.words('english'))
    # extend stopwords by your choices
    stop_words.extend(['https', 'co', 'dont'] + list(swords))  # Put that in a function

    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    data_words_nostops = remove_stopwords(data_words)
    print(data_words_nostops)

    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    # Take the POS tag and use NLTK lemmatizer to apply lemmatization
    def lemmatize(texts):
        return [
            [WordNetLemmatizer().lemmatize(word, pos=get_wordnet_pos(word)) for word in simple_preprocess(str(doc)) if
             word not in stop_words] for doc in texts]

    data_lemmatized = lemmatize(data_words_nostops)
    print(data_lemmatized)
    flat_data = [x for xs in data_lemmatized for x in xs]

    doc = nlp(flat_data)

    # Extrahovat popis vzhledu
    appearance = ""
    for token in doc:
        if token.text.lower() == "creature" and token.dep_ == "attr":
            appearance_start = text.find("Pikachu is") + len("Pikachu is ")
            appearance_end = text.find(".", appearance_start)
            appearance = text[appearance_start:appearance_end]
            break
    data["appearance"] = appearance

    # Extrahovat schopnosti a útoky
    abilities = []
    for token in doc:
        if token.text in ["Thunderbolt", "Thunder Shock"]:
            abilities.append(token.text)
    data["abilities"] = abilities

    # Další informace
    data["role"] = "mascot" if "mascot" in text.lower() else ""
    data["popularity"] = "popular worldwide" if "popular worldwide" in text.lower() else ""

    return data


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
