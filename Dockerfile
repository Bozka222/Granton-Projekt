FROM python:3.9

ADD Data_Formatting.py .
ADD README.md .

CMD ["python", "/Data_Formatting.py"]