import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath(r"C:\Users\jeyka\IBM\merged_intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
        
counter = 0

def main():
    global counter
    st.title("IT Professional Chatbot")
    
    # Sidebar Menu
    menu = ["Home", "Coding Challenges", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
        st.write("Welcome to the IT Professional Chatbot! Type a message and get responses related to IT, coding, and productivity.")
        
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])
        
        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")
        
        if user_input:
            user_input_str = str(user_input)
            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")
            
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])
            
            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Stay productive!")
                st.stop()
    
    elif choice == "Coding Challenges":
        challenges = [
            "Write a Python function to check if a number is prime.",
            "Reverse a linked list in Java.",
            "Optimize a SQL query for better performance.",
            "Implement a simple REST API using Flask.",
            "Debug a JavaScript async/await function."
        ]
        st.write("Today's coding challenge:")
        st.write(random.choice(challenges))
    
    elif choice == "Conversation History":
        st.header("Conversation History")
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")
    
    elif choice == "About":
        st.write("This chatbot is designed for IT professionals, providing coding tips, motivation, and stress relief.")
        st.subheader("Features:")
        st.write("- IT-related queries and responses")
        st.write("- Coding challenges for skill improvement")
        st.write("- Conversation history tracking")
        st.write("- Motivation and productivity tips")
        
if __name__ == '__main__':
    main()
