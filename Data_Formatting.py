from openai import OpenAI
from flask import Flask, request, jsonify, render_template
import os
from dotenv import load_dotenv, dotenv_values
import spacy
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

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI"))

app = Flask(__name__)


# @app.route('/')
# def home():
#     return render_template("index.html")


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

    # Remove newline characters single quotes nad other characters
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
    data = [re.sub('\s+', ' ', sent) for sent in data]
    data = [re.sub("\'", '', sent) for sent in data]

    # Split text by words
    def sent_to_words(sentences):
        for sentence in sentences:
            yield gensim.utils.simple_preprocess(str(sentence), deacc=True)

    data_words = list(sent_to_words(data))
    print(data_words)

    # Create stopword list (prepositions etc.)
    stop_words = st.words('english')
    stop_words.append(stopwords.words('english'))
    stop_words.extend(['dont'] + list(swords))  # Add custom stopwords

    # Remove stopwords from text
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    data_words_nostops = remove_stopwords(data_words)
    print(data_words_nostops)

    # Convert words to its dictionary form (lemmatization)
    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    # Lemmatize words
    def lemmatize(texts):
        return [
            [WordNetLemmatizer().lemmatize(word, pos=get_wordnet_pos(word)) for word in simple_preprocess(str(doc)) if
             word not in stop_words] for doc in texts]

    data_lemmatized = lemmatize(data_words_nostops)
    print(data_lemmatized)

    flat_data = [x for xs in data_lemmatized for x in xs]
    doc = nlp(flat_data)

    # Sort data to JSON file
    # Appearance
    appearance = ""
    for token in doc:
        if token.text.lower() == "creature" and token.dep_ == "attr":
            appearance_start = text.find("Pikachu is") + len("Pikachu is ")
            appearance_end = text.find(".", appearance_start)
            appearance = text[appearance_start:appearance_end]
            break
    data["appearance"] = appearance

    # Abilities
    abilities = []
    for token in doc:
        if token.text in ["Thunderbolt", "Thunder Shock"]:
            abilities.append(token.text)
    data["abilities"] = abilities

    # Other attributes
    data["role"] = "mascot" if "mascot" in text.lower() else ""
    data["popularity"] = "popular worldwide" if "popular worldwide" in text.lower() else ""

    return data


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
