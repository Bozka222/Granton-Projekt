FROM python:3.9

COPY Data_Formatting.py .
COPY requirements.txt .
COPY README.md .
COPY .env .

RUN pip install -r requirements.txt
RUN python -m nltk.downloader wordnet
RUN python -m nltk.downloader stopwords
RUN python -m spacy download en

CMD ["python", "/Data_Formatting.py"]