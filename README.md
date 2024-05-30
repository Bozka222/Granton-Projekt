# Granton-Projekt
 To build docker file use command: <br>**docker build -t project-dockerization .**<br>
 To run docker file use command: <br>**docker run -p 5000:5000 -e OPENAI="API_KEY" project-dockerization**<br>
 Make API PUT request with Postman App:
 https://app.getpostman.com/join-team?invite_code=2093beb2eec6888a85832e674ef40eaa&target_code=8cd7ac232777c042f02327e0e2ae4f8b
 
I could not finish the main script to its final form but the main idea of it was to clean unstructured text in few steps:
1. remove all unwanted characters
2. create stopword list and remove all stop words like prepositions
3. convert words to its dictionary form
4. sort data to structured JSON file with regex or other similar function

Input to the ChatGPT is now hardcoded in message but could be done with the JavaScript 
code which makes input window and submit button at the homepage.

The PUT request is sent through the postman application.