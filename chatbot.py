import os
import ssl
import random
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Resolve SSL issues
ssl._create_default_https_context = ssl._create_unverified_context

# Define intents for green technology
intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
        "responses": ["Hello! How can I help you with green technology today?", "Hi there! Let's talk about sustainability."]
    },
    {
        "tag": "renewable_energy",
        "patterns": ["Tell me about renewable energy", "What is renewable energy?", "Examples of renewable energy?"],
        "responses": [
            "Renewable energy comes from natural sources like sunlight, wind, rain, tides, and geothermal heat. Solar panels and wind turbines are common examples.",
            "Renewable energy is a sustainable alternative to fossil fuels and includes sources like solar, wind, and hydro power."
        ]
    },
    {
        "tag": "carbon_footprint",
        "patterns": ["How can I reduce my carbon footprint?", "Ways to lower carbon emissions?", "What is a carbon footprint?"],
        "responses": [
            "A carbon footprint measures the total greenhouse gases emitted by your activities. You can reduce it by using public transport, switching to renewable energy, and reducing waste.",
            "To lower your carbon footprint, adopt sustainable practices like eating less meat, conserving water, and using energy-efficient appliances."
        ]
    },
    {
        "tag": "sustainability",
        "patterns": ["What is sustainability?", "Why is sustainability important?", "Explain sustainability."],
        "responses": [
            "Sustainability means meeting our needs without compromising the ability of future generations to meet theirs. It involves balancing environmental, economic, and social factors.",
            "Sustainability is about using resources wisely to protect the planet while ensuring economic and social well-being."
        ]
    },
    {
        "tag": "goodbye",
        "patterns": ["Goodbye", "Bye", "See you later"],
        "responses": ["Goodbye! Remember to stay green and make eco-friendly choices.", "Bye! Keep working towards a sustainable future."]
    }
]

# Prepare data for training
tags = []
patterns = []

for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train model
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(patterns)
y = tags

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

clf = LogisticRegression(random_state=0, max_iter=10000)
clf.fit(x_train, y_train)

# Define the chatbot function
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    try:
        tag = clf.predict(input_text)[0]
        for intent in intents:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
    except:
        return "I'm sorry, I didn't understand that. Could you rephrase?"

# Streamlit chatbot UI
def main():
    st.title("Green Technology ChatBot")
    st.write("Welcome! I'm here to help you learn about green technology and sustainability.")
    
    user_input = st.text_input("You:", key="user_input")
    if user_input:
        response = chatbot(user_input)
        st.text_area("Chatbot:", value=response, height=100, max_chars=None, key="chatbot_response")
        
        if "goodbye" in response.lower() or "bye" in response.lower():
            st.write("Thank you for using me! Stay eco-friendly and protect our planet.")
            st.stop()

if __name__ == '__main__':
    main()
