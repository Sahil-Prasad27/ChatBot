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
    },
    {
        "tag": "solar_energy",
        "patterns": ["What is solar energy?", "Tell me about solar panels", "How does solar energy work?"],
        "responses": [
            "Solar energy is energy derived from the sun's radiation. It can be harnessed using solar panels or solar thermal systems to generate electricity or heat.",
            "Solar panels capture sunlight and convert it into electricity. They are an important part of the renewable energy revolution."
        ]
    },
    {
        "tag": "wind_energy",
        "patterns": ["What is wind energy?", "How do wind turbines work?", "Tell me about wind power."],
        "responses": [
            "Wind energy is generated by converting the kinetic energy of wind into electricity using wind turbines.",
            "Wind turbines capture the wind's energy to turn blades that generate electricity. It’s a clean and renewable energy source."
        ]
    },
    {
        "tag": "energy_efficiency",
        "patterns": ["What is energy efficiency?", "How can I improve energy efficiency?", "Why is energy efficiency important?"],
        "responses": [
            "Energy efficiency is the practice of using less energy to perform the same task. It's important for reducing greenhouse gas emissions and conserving resources.",
            "You can improve energy efficiency by using energy-efficient appliances, insulating your home, and switching to LED lighting."
        ]
    },
    {
        "tag": "green_building",
        "patterns": ["What is a green building?", "How can buildings be eco-friendly?", "Tell me about sustainable buildings."],
        "responses": [
            "A green building is designed to reduce its environmental impact. This includes energy efficiency, water conservation, and using sustainable materials.",
            "Sustainable buildings use eco-friendly construction methods, renewable energy sources, and innovative designs to minimize waste and energy consumption."
        ]
    },
    {
        "tag": "electric_vehicles",
        "patterns": ["Tell me about electric vehicles", "What are electric cars?", "How do electric vehicles work?"],
        "responses": [
            "Electric vehicles (EVs) run on electricity stored in batteries instead of gasoline or diesel. They emit fewer greenhouse gases and reduce air pollution.",
            "Electric vehicles are powered by electricity, usually from renewable sources, and are a cleaner alternative to traditional vehicles."
        ]
    },
    {
        "tag": "carbon_offset",
        "patterns": ["What is carbon offset?", "How does carbon offsetting work?", "How can I offset my carbon emissions?"],
        "responses": [
            "Carbon offsetting is the practice of compensating for emissions by funding projects that reduce or capture an equivalent amount of greenhouse gases.",
            "You can offset your carbon emissions by investing in renewable energy projects, tree planting initiatives, or carbon capture technologies."
        ]
    },
    {
        "tag": "climate_change",
        "patterns": ["What is climate change?", "How does climate change affect the planet?", "What can we do to stop climate change?"],
        "responses": [
            "Climate change refers to long-term changes in temperature and weather patterns caused by human activity, particularly the burning of fossil fuels.",
            "To fight climate change, we can reduce emissions, use renewable energy, protect forests, and support policies that promote sustainability."
        ]
    },
    {
        "tag": "plastic_pollution",
        "patterns": ["What is plastic pollution?", "How can we reduce plastic waste?", "Why is plastic pollution a problem?"],
        "responses": [
            "Plastic pollution is the accumulation of plastic waste in the environment. It harms wildlife, ecosystems, and contributes to long-term environmental degradation.",
            "We can reduce plastic pollution by reducing plastic use, recycling, and opting for sustainable alternatives like biodegradable materials."
        ]
    },
    {
        "tag": "water_conservation",
        "patterns": ["How can I conserve water?", "Why is water conservation important?", "Tell me about water conservation methods."],
        "responses": [
            "Water conservation involves reducing water wastage and using water efficiently. You can conserve water by fixing leaks, using water-efficient appliances, and reducing water use in landscaping.",
            "Saving water helps to ensure that this precious resource remains available for future generations and supports the health of our environment."
        ]
    },
    {
        "tag": "sustainable_food",
        "patterns": ["What is sustainable food?", "How can I eat sustainably?", "Why is sustainable food important?"],
        "responses": [
            "Sustainable food involves choosing foods that are produced with minimal environmental impact. This includes organic, locally grown, and plant-based foods.",
            "Eating sustainably means consuming foods that are produced in ways that protect the environment, support fair trade, and promote animal welfare."
        ]
    },
    {
        "tag": "green_transportation",
        "patterns": ["What is green transportation?", "How can I travel sustainably?", "Tell me about eco-friendly transportation options."],
        "responses": [
            "Green transportation includes options that have a lower environmental impact, such as electric vehicles, cycling, walking, and using public transport.",
            "By choosing green transportation, we can reduce our carbon footprint, improve air quality, and support cleaner, more sustainable cities."
        ]
    },
    {
        "tag": "sustainable_fashion",
        "patterns": ["What is sustainable fashion?", "How can I shop sustainably?", "Tell me about eco-friendly fashion brands."],
        "responses": [
            "Sustainable fashion focuses on producing clothing in ways that are kind to the environment and workers. It involves using sustainable materials, ethical production methods, and reducing waste.",
            "Eco-friendly fashion includes brands that use recycled materials, promote fair labor practices, and reduce waste by encouraging upcycling and repair."
        ]
    },
    {
        "tag": "green_energy_innovation",
        "patterns": ["What are the latest green energy innovations?", "Tell me about new renewable energy technologies.", "What are the upcoming trends in green energy?"],
        "responses": [
            "Recent innovations in green energy include advanced solar cells, wind energy storage solutions, and bioenergy technologies like algae-based fuels.",
            "Emerging trends in green energy involve floating wind farms, enhanced geothermal systems, and breakthrough solar technologies like perovskite solar cells."
        ]
    },
    {
        "tag": "biofuels",
        "patterns": ["What are biofuels?", "Tell me about biofuel production.", "How do biofuels work?"],
        "responses": [
            "Biofuels are renewable fuels made from organic materials, such as plant or animal waste. They can replace fossil fuels in transportation and energy production.",
            "Biofuels can be made from crops like corn or algae and serve as a cleaner alternative to gasoline and diesel, reducing overall greenhouse gas emissions."
        ]
    },
    {
        "tag": "sustainable_agriculture",
        "patterns": ["What is sustainable agriculture?", "How can farming be eco-friendly?", "Tell me about sustainable farming methods."],
        "responses": [
            "Sustainable agriculture is farming that maintains or improves soil health, conserves water, reduces pollution, and promotes biodiversity.",
            "Methods like crop rotation, organic farming, and reduced pesticide use help make agriculture more sustainable and less damaging to the environment."
        ]
    },
    {
        "tag": "green_finance",
        "patterns": ["What is green finance?", "How does green finance work?", "Tell me about sustainable investment options."],
        "responses": [
            "Green finance involves investments in projects that have environmental benefits, such as renewable energy, green infrastructure, and low-carbon technologies.",
            "Green bonds, impact investing, and sustainable mutual funds are examples of financial products that support environmentally friendly initiatives."
        ]
    },
    {
        "tag": "climate_resilience",
        "patterns": ["What is climate resilience?", "How can communities build climate resilience?", "What are some strategies for climate adaptation?"],
        "responses": [
            "Climate resilience refers to the ability of a system or community to anticipate, prepare for, and respond to climate impacts, such as floods, droughts, and extreme weather events.",
            "Building resilience involves strategies like creating flood-resistant infrastructure, planting climate-resilient crops, and protecting natural ecosystems like wetlands."
        ]
    },
    {
        "tag": "green_jobs",
        "patterns": ["What are green jobs?", "How can I find a job in green technology?", "Tell me about eco-friendly careers."],
        "responses": [
            "Green jobs involve work that contributes to preserving or restoring the environment. These careers include roles in renewable energy, environmental conservation, and sustainable construction.",
            "To find a green job, consider pursuing careers in solar panel installation, environmental engineering, green building design, or sustainability consulting."
        ]
    },
    {
        "tag": "smart_grids",
        "patterns": ["What is a smart grid?", "How do smart grids work?", "Tell me about the future of electrical grids."],
        "responses": [
            "A smart grid uses digital technology to monitor and manage electricity flow, enabling better integration of renewable energy and more efficient distribution.",
            "Smart grids allow for real-time communication between utilities and consumers, improving grid reliability and helping reduce energy waste."
        ]
    },
    {
        "tag": "circular_economy",
        "patterns": ["What is the circular economy?", "How does the circular economy work?", "Tell me about the benefits of a circular economy."],
        "responses": [
            "The circular economy is a system where resources are reused, repaired, and recycled to create a closed-loop system, minimizing waste and conserving resources.",
            "In a circular economy, products are designed for durability, repairability, and recyclability, reducing the need for new raw materials."
        ]
    },
    {
        "tag": "greenwashing",
        "patterns": ["What is greenwashing?", "How can I avoid greenwashing?", "What should I watch out for in eco-friendly claims?"],
        "responses": [
            "Greenwashing refers to misleading claims made by companies to appear environmentally friendly when their practices are not truly sustainable.",
            "To avoid greenwashing, look for third-party certifications, such as organic labels or energy-efficient certifications, and do some research on the company’s actual practices."
        ]
    },
    {
        "tag": "zero_waste_living",
        "patterns": ["What is zero waste living?", "How can I live a zero waste lifestyle?", "Tell me about zero waste practices."],
        "responses": [
            "Zero waste living is a lifestyle that aims to reduce the amount of waste sent to landfills by reusing, recycling, and composting as much as possible.",
            "Practices for zero waste living include avoiding single-use plastics, buying in bulk, and using reusable containers and bags."
        ]
    },
    {
        "tag": "green_chemistry",
        "patterns": ["What is green chemistry?", "How does green chemistry help the environment?", "Tell me about sustainable chemistry."],
        "responses": [
            "Green chemistry is the design of chemical products and processes that reduce or eliminate the use and generation of hazardous substances.",
            "Green chemistry helps reduce pollution by using sustainable raw materials, energy-efficient processes, and minimizing toxic byproducts."
        ]
    },
    {
        "tag": "environmental_education",
        "patterns": ["Why is environmental education important?", "How can I learn more about environmental issues?", "Tell me about eco-friendly education programs."],
        "responses": [
            "Environmental education helps people understand environmental issues, sustainability practices, and ways to protect natural resources.",
            "Programs like eco-schools and community sustainability workshops offer practical knowledge to individuals looking to reduce their environmental impact."
        ]
    },
    {
        "tag": "sustainable_transportation_infrastructure",
        "patterns": ["What is sustainable transportation infrastructure?", "How can cities become more sustainable?", "Tell me about eco-friendly transportation systems."],
        "responses": [
            "Sustainable transportation infrastructure includes public transport systems, bike lanes, pedestrian paths, and electric vehicle charging stations designed to reduce carbon emissions.",
            "Cities can become more sustainable by promoting public transit, building bike-friendly infrastructure, and investing in electric vehicle infrastructure."
        ]
    },
    {
        "tag": "eco_building_materials",
        "patterns": ["What are eco-friendly building materials?", "Tell me about sustainable construction materials.", "How can I build using green materials?"],
        "responses": [
            "Eco-friendly building materials include bamboo, reclaimed wood, recycled steel, and low-VOC paints. These materials reduce environmental impact and improve energy efficiency.",
            "Using sustainable construction materials helps reduce carbon emissions, conserve natural resources, and minimize waste during construction."
        ]
    },
    {
        "tag": "eco_tourism",
        "patterns": ["What is eco-tourism?", "How can I travel sustainably?", "Tell me about eco-friendly travel options."],
        "responses": [
            "Eco-tourism is responsible travel that focuses on conservation and supporting local communities while minimizing environmental impact.",
            "Sustainable travel practices include choosing eco-friendly accommodations, reducing energy consumption, and participating in activities that promote conservation."
        ]
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
    try:
        for intent in intents:
            if any(pattern.lower() in input_text.lower() for pattern in intent["patterns"]):
                return random.choice(intent["responses"])
    except:
        return "I'm sorry, I didn't understand that. Could you rephrase?"

# Streamlit chatbot UI
def main():
    st.title("Green Technology ChatBot")
    st.write("Welcome! I'm here to help you learn about green technology and sustainability.")
    
    # Initialize session state for chat history if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # User input
    user_input = st.text_input("You:", key="user_input")

    if user_input:
        # Get the chatbot response
        response = chatbot(user_input)

        # Add the user input and chatbot response to the session history
        st.session_state.chat_history = [(f"You: {user_input}", f"ChatBot: {response}")]

        # Display the latest chat history (only the most recent message)
        for user_msg, bot_msg in st.session_state.chat_history:
            st.write(user_msg)
            st.write(bot_msg)

        # Check if response includes "goodbye" or "bye"
        if "goodbye" in response.lower() or "bye" in response.lower():
            st.write("Thank you for using me! Stay eco-friendly and protect our planet.")
            st.stop()

if __name__ == '__main__':
    main()
