FROM python:3.9

ADD Data_Formatting.py .
ADD requirements.txt .
ADD README.md .
ADD .env .

RUN pip install -r requirements.txt
RUN python -m nltk.downloader wordnet
RUN python -m nltk.downloader stopwords
RUN python -m spacy download en

CMD ["python", "/Data_Formatting.py"]