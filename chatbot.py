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
        "tag": "recycling",
        "patterns": ["What can I recycle?", "How does recycling help the environment?", "Why should we recycle?"],
        "responses": [
            "Recycling helps reduce waste, conserve resources, and prevent pollution. Items like paper, plastic, glass, and metals can often be recycled.",
            "By recycling, we save energy and reduce the demand for raw materials, helping protect ecosystems and reduce carbon emissions."
        ]
    },
    {
        "tag": "climate_change",
        "patterns": ["What is climate change?", "How does climate change affect us?", "What are the causes of climate change?"],
        "responses": [
            "Climate change refers to long-term shifts in temperatures and weather patterns, mainly caused by human activities like burning fossil fuels.",
            "Climate change leads to rising sea levels, extreme weather events, and biodiversity loss. Reducing emissions and using renewable energy can help mitigate its effects."
        ]
    },
    {
        "tag": "solar_power",
        "patterns": ["How does solar power work?", "Benefits of solar energy?", "What are solar panels?"],
        "responses": [
            "Solar power converts sunlight into electricity using solar panels made of photovoltaic cells. It's a clean and renewable energy source.",
            "The benefits of solar power include reducing electricity bills, lowering carbon footprints, and providing a sustainable energy source."
        ]
    },
    {
        "tag": "water_conservation",
        "patterns": ["Why conserve water?", "How can I save water?", "Tips for water conservation?"],
        "responses": [
            "Water conservation is essential to ensure future availability and protect ecosystems. Simple steps like fixing leaks and using water-efficient appliances can help.",
            "To save water, take shorter showers, collect rainwater, and use native plants in your garden to reduce irrigation needs."
        ]
    },
    {
        "tag": "electric_vehicles",
        "patterns": ["What are electric vehicles?", "Benefits of EVs?", "Why switch to electric cars?"],
        "responses": [
            "Electric vehicles (EVs) run on electricity instead of fossil fuels, reducing air pollution and greenhouse gas emissions.",
            "Switching to electric cars helps combat climate change, reduces dependency on oil, and offers lower maintenance costs compared to traditional vehicles."
        ]
    },
    {
        "tag": "green_buildings",
        "patterns": ["What are green buildings?", "Benefits of sustainable architecture?", "How to make buildings eco-friendly?"],
        "responses": [
            "Green buildings are designed to minimize environmental impact by using energy-efficient materials and renewable energy sources.",
            "Sustainable architecture reduces energy consumption, improves air quality, and promotes the use of renewable resources like solar panels and rainwater harvesting systems."
        ]
    },
    {
        "tag": "eco_friendly_products",
        "patterns": ["What are eco-friendly products?", "Examples of sustainable products?", "Why use eco-friendly products?"],
        "responses": [
            "Eco-friendly products are made with minimal environmental impact, often using biodegradable or recycled materials. Examples include reusable bags, bamboo utensils, and compostable packaging.",
            "Using sustainable products reduces waste and supports companies committed to protecting the environment."
        ]
    },
    {
        "tag": "renewable_energy_advantages",
        "patterns": ["Why use renewable energy?", "Benefits of green energy?", "Is renewable energy reliable?"],
        "responses": [
            "Renewable energy is sustainable, reduces greenhouse gas emissions, and lowers dependency on fossil fuels. It's also cost-effective in the long run.",
            "Green energy is reliable when combined with technologies like energy storage, ensuring consistent power even during low sunlight or wind periods."
        ]
    },
    {
        "tag": "plastic_pollution",
        "patterns": ["What is plastic pollution?", "How does plastic harm the environment?", "Ways to reduce plastic waste?"],
        "responses": [
            "Plastic pollution occurs when plastic waste accumulates in the environment, harming wildlife and ecosystems. It's a major global issue.",
            "To reduce plastic waste, avoid single-use plastics, use reusable items, and recycle properly. Supporting biodegradable alternatives also helps."
        ]
    },
    {
        "tag": "energy_efficiency",
        "patterns": ["What is energy efficiency?", "How to improve energy efficiency?", "Benefits of energy-efficient appliances?"],
        "responses": [
            "Energy efficiency means using less energy to perform the same tasks, reducing energy waste and lowering costs.",
            "You can improve energy efficiency by using LED lights, energy-efficient appliances, and insulating your home to minimize heating and cooling needs."
        ]
    },
    {
        "tag": "deforestation",
        "patterns": ["What is deforestation?", "Effects of deforestation?", "How to stop deforestation?"],
        "responses": [
            "Deforestation is the clearing of forests for agriculture, urban development, or logging. It leads to habitat loss and contributes to climate change.",
            "To combat deforestation, support sustainable forestry, reduce paper usage, and advocate for policies protecting natural habitats."
        ]
    },
    {
        "tag": "composting",
        "patterns": ["What is composting?", "How does composting work?", "Why is composting important?"],
        "responses": [
            "Composting is the process of decomposing organic waste like food scraps and yard waste into nutrient-rich soil. It's great for gardening and reduces landfill waste.",
            "By composting, you create natural fertilizer, reduce methane emissions from landfills, and promote healthier soil."
        ]
    },
    {
        "tag": "greenhouse_effect",
        "patterns": ["What is the greenhouse effect?", "How does the greenhouse effect impact the planet?", "Explain the greenhouse effect."],
        "responses": [
            "The greenhouse effect occurs when gases like CO2 trap heat in the Earth's atmosphere, leading to global warming. It's a natural process, but human activities have intensified it.",
            "While the greenhouse effect keeps Earth warm enough for life, excessive greenhouse gases are causing climate change."
        ]
    },
    {
        "tag": "biodegradable_materials",
        "patterns": ["What are biodegradable materials?", "Why use biodegradable materials?", "Examples of biodegradable materials?"],
        "responses": [
            "Biodegradable materials naturally break down into non-toxic substances, reducing waste. Examples include paper, cotton, and certain bioplastics.",
            "Using biodegradable materials helps minimize environmental impact and supports sustainable waste management."
        ]
    },
    {
        "tag": "sustainable_fashion",
        "patterns": ["What is sustainable fashion?", "How can I support eco-friendly clothing?", "Why is sustainable fashion important?"],
        "responses": [
            "Sustainable fashion focuses on environmentally friendly production, ethical labor practices, and reducing waste. It includes using organic fabrics and second-hand clothing.",
            "Supporting sustainable fashion reduces water consumption, pollution, and the exploitation of workers in the fashion industry."
        ]
    },
    {
        "tag": "wind_energy",
        "patterns": ["What is wind energy?", "How do wind turbines work?", "Benefits of wind energy?"],
        "responses": [
            "Wind energy uses turbines to convert wind into electricity. It's a clean, renewable energy source with minimal environmental impact.",
            "Wind energy reduces carbon emissions, provides sustainable power, and supports energy independence."
        ]
    },
    {
        "tag": "ocean_conservation",
        "patterns": ["Why is ocean conservation important?", "How can we protect marine life?", "What is ocean pollution?"],
        "responses": [
            "Ocean conservation is vital to protect marine ecosystems, combat pollution, and sustain biodiversity. Healthy oceans also regulate the Earth's climate.",
            "Protect marine life by reducing plastic waste, supporting sustainable fishing, and advocating for marine protected areas."
        ]
    },
    {
        "tag": "clean_air",
        "patterns": ["How can we improve air quality?", "What are the benefits of clean air?", "What causes air pollution?"],
        "responses": [
            "Clean air is essential for health and the environment. To improve air quality, reduce vehicle emissions, use renewable energy, and plant trees.",
            "Air pollution is caused by industrial emissions, vehicle exhaust, and burning fossil fuels. Reducing these activities leads to cleaner air."
        ]
    },
    {
        "tag": "eco_tourism",
        "patterns": ["What is eco-tourism?", "Why choose eco-tourism?", "Benefits of sustainable travel?"],
        "responses": [
            "Eco-tourism focuses on responsible travel to natural areas, conserving the environment, and benefiting local communities.",
            "By choosing eco-tourism, you support conservation efforts, promote sustainable economies, and reduce the impact of travel on ecosystems."
        ]
    },
    {
        "tag": "sustainable_agriculture",
        "patterns": ["What is sustainable agriculture?", "How does sustainable farming work?", "Benefits of sustainable agriculture?"],
        "responses": [
            "Sustainable agriculture uses eco-friendly farming methods that conserve resources, protect biodiversity, and ensure long-term food security.",
            "Practices like crop rotation, organic farming, and water conservation are examples of sustainable agriculture."
        ]
    },
    {
        "tag": "zero_waste_lifestyle",
        "patterns": ["What is a zero-waste lifestyle?", "How can I reduce waste?", "Tips for zero waste?"],
        "responses": [
            "A zero-waste lifestyle aims to reduce trash sent to landfills by reusing, recycling, and composting. It focuses on sustainability and reducing consumption.",
            "To live a zero-waste lifestyle, use reusable products, buy in bulk, and repair items instead of discarding them."
        ]
    },
    {
        "tag": "renewable_energy_jobs",
        "patterns": ["What jobs are available in renewable energy?", "Careers in green technology?", "How can I work in renewable energy?"],
        "responses": [
            "Renewable energy careers include roles like solar panel installers, wind turbine technicians, environmental scientists, and energy consultants.",
            "Green technology jobs are growing rapidly as the world transitions to sustainable energy. Consider studying engineering, environmental science, or sustainable business."
        ]
    },
    {
        "tag": "hydro_power",
        "patterns": ["What is hydro power?", "How does hydroelectric energy work?", "Is hydro power renewable?"],
        "responses": [
            "Hydro power generates electricity by using flowing water to spin turbines. It's a renewable energy source that relies on rivers and dams.",
            "Hydroelectric energy is reliable and produces no emissions, but it requires careful management to minimize environmental impact."
        ]
    },
    {
        "tag": "green_innovation",
        "patterns": ["What are green innovations?", "Examples of sustainable technologies?", "How does innovation help the environment?"],
        "responses": [
            "Green innovations are technologies designed to reduce environmental impact, such as electric vehicles, vertical farming, and carbon capture systems.",
            "Sustainable technologies improve energy efficiency, reduce waste, and create solutions for global environmental challenges."
        ]
    },
    {
        "tag": "sustainable_living",
        "patterns": ["What is sustainable living?", "How can I live sustainably?", "Examples of sustainable habits?"],
        "responses": [
            "Sustainable living involves reducing your ecological footprint by using fewer resources and adopting eco-friendly practices like recycling and conserving energy.",
            "Habits like eating locally sourced food, reducing waste, and using renewable energy are key to sustainable living."
        ]
    },
    {
        "tag": "solar_panels",
        "patterns": ["How do solar panels work?", "Are solar panels worth it?", "Tell me about solar panels"],
        "responses": [
            "Solar panels convert sunlight into electricity using photovoltaic cells. They are a sustainable and cost-effective way to generate energy.",
            "Solar panels are a great investment for reducing energy bills and lowering your carbon footprint."
        ]
    },
    {
        "tag": "electric_cars",
        "patterns": ["Are electric cars eco-friendly?", "How do electric cars help the environment?", "Should I switch to an electric car?"],
        "responses": [
            "Electric cars produce zero emissions while driving, helping to reduce air pollution and dependence on fossil fuels.",
            "Switching to an electric car is a great way to support green transportation and lower your carbon footprint."
        ]
    },
    {
        "tag": "tree_planting",
        "patterns": ["Why is tree planting important?", "Benefits of planting trees?", "How does planting trees help the environment?"],
        "responses": [
            "Planting trees helps absorb carbon dioxide, provide oxygen, prevent soil erosion, and support biodiversity.",
            "Trees are vital for combating climate change and creating healthier ecosystems."
        ]
    },
    {
        "tag": "recycling_plastics",
        "patterns": ["How can I recycle plastic?", "Why is recycling plastic important?", "What happens to recycled plastic?"],
        "responses": [
            "Recycling plastic reduces waste in landfills and oceans. It can be reused to create new products like clothing, furniture, or containers.",
            "By recycling plastic, you help conserve natural resources and reduce environmental pollution."
        ]
    },
    {
        "tag": "normal_conversation",
        "patterns": ["What's your name?", "Who are you?", "What do you do?"],
        "responses": [
            "I'm your green technology assistant, here to help you learn about sustainability!",
            "I’m a bot designed to assist you with green technology and eco-friendly living. How can I help?"
        ]
    },
    {
        "tag": "eco_friendly_homes",
        "patterns": ["What is an eco-friendly home?", "How to make my home eco-friendly?", "Tips for a sustainable house?"],
        "responses": [
            "Eco-friendly homes are designed to be energy-efficient and sustainable, often using solar panels, smart thermostats, and sustainable materials.",
            "Start by using energy-efficient appliances, insulating your home, and switching to renewable energy sources."
        ]
    },
    {
        "tag": "normal_how_are_you",
        "patterns": ["How are you?", "How's it going?", "Are you doing well?"],
        "responses": [
            "I'm great! Excited to help you with green technology questions.",
            "I'm doing well, thanks! How can I assist you today?"
        ]
    },
    {
        "tag": "organic_farming",
        "patterns": ["What is organic farming?", "Benefits of organic farming?", "How does organic farming work?"],
        "responses": [
            "Organic farming avoids synthetic pesticides and fertilizers, focusing on natural methods to grow food sustainably.",
            "It helps improve soil health, reduce pollution, and produce healthier food for consumption."
        ]
    },
    {
        "tag": "sustainable_energy_storage",
        "patterns": ["What are sustainable energy storage methods?", "How is energy stored sustainably?", "Examples of green energy storage?"],
        "responses": [
            "Sustainable energy storage includes battery systems like lithium-ion, flow batteries, and even hydrogen storage.",
            "These methods store renewable energy efficiently for future use, reducing waste and improving grid reliability."
        ]
    },
    {
        "tag": "water_conservation",
        "patterns": ["How can I conserve water?", "Tips to save water?", "Why is water conservation important?"],
        "responses": [
            "Conserve water by fixing leaks, using water-efficient appliances, and collecting rainwater for reuse.",
            "Water conservation ensures future availability and reduces energy consumption for water processing."
        ]
    },
    {
        "tag": "green_jobs",
        "patterns": ["What are green jobs?", "How do I get a green job?", "Examples of green careers?"],
        "responses": [
            "Green jobs focus on environmental sustainability, like renewable energy engineers, conservationists, and eco-designers.",
            "Explore green careers in sectors like clean energy, sustainable architecture, and environmental science."
        ]
    },
    {
        "tag": "normal_weather",
        "patterns": ["How's the weather?", "What's the weather like?", "Is it sunny outside?"],
        "responses": [
            "I don’t have real-time weather updates, but I hope it’s nice where you are!",
            "I'm not sure about the weather, but it’s always a good time to discuss green technology!"
        ]
    },
    {
        "tag": "eco_transport",
        "patterns": ["What are eco-friendly transportation options?", "How can I travel sustainably?", "Examples of green transport?"],
        "responses": [
            "Eco-friendly transport includes electric cars, bicycles, public transportation, and carpooling.",
            "Travel sustainably by walking, biking, or using electric scooters for short distances."
        ]
    },
    {
        "tag": "energy_efficient_appliances",
        "patterns": ["What are energy-efficient appliances?", "How do I choose energy-efficient devices?", "Examples of green appliances?"],
        "responses": [
            "Energy-efficient appliances use less electricity and water, reducing utility bills and environmental impact. Look for Energy Star ratings.",
            "Examples include LED bulbs, low-flow showerheads, and energy-efficient refrigerators."
        ]
    },
    {
        "tag": "climate_change_mitigation",
        "patterns": ["How can we mitigate climate change?", "What are solutions to climate change?", "Ways to combat climate change?"],
        "responses": [
            "Mitigate climate change by reducing emissions, switching to renewable energy, and supporting reforestation projects.",
            "Simple actions like conserving energy, eating sustainably, and using public transport also make a difference."
        ]
    },
    {
        "tag": "normal_favorite_color",
        "patterns": ["What's your favorite color?", "Do you like green?", "What color do you like?"],
        "responses": [
            "Green, of course! It's the color of sustainability and the environment.",
            "I love green—it reminds me of nature and all things eco-friendly!"
        ]
    },
    {
        "tag": "biofuels",
        "patterns": ["What are biofuels?", "How are biofuels made?", "Are biofuels sustainable?"],
        "responses": [
            "Biofuels are made from organic materials like crops and waste. They are a renewable energy source used in transportation.",
            "Biofuels can reduce carbon emissions but must be produced sustainably to avoid competing with food crops."
        ]
    },
    {
        "tag": "energy_audit",
        "patterns": ["What is an energy audit?", "How does an energy audit work?", "Why get an energy audit?"],
        "responses": [
            "An energy audit assesses how energy is used in a building to identify ways to save energy and reduce costs.",
            "Energy audits help improve efficiency, lower bills, and reduce environmental impact."
        ]
    },
    {
        "tag": "geothermal_energy",
        "patterns": ["What is geothermal energy?", "How does geothermal energy work?", "Is geothermal energy renewable?"],
        "responses": [
            "Geothermal energy uses heat from the Earth to generate electricity or provide heating. It's a clean and renewable resource.",
            "It works by tapping into underground reservoirs of steam or hot water to produce power."
        ]
    },
    {
        "tag": "normal_hobbies",
        "patterns": ["What are your hobbies?", "What do you like to do?", "Do you have any hobbies?"],
        "responses": [
            "I love helping people learn about sustainability and answering questions about green technology.",
            "My favorite activity is promoting eco-friendly living and inspiring change!"
        ]
    },
    {
        "tag": "solar_panels",
        "patterns": ["How do solar panels work?", "Explain the working of solar panels.", "Benefits of solar panels?"],
        "responses": [
            "Solar panels convert sunlight into electricity using photovoltaic cells. They're a clean and renewable energy source.",
            "By harnessing the power of the sun, solar panels help reduce greenhouse gas emissions and lower electricity bills."
        ]
    },
    {
        "tag": "geothermal_energy",
        "patterns": ["What is geothermal energy?", "How does geothermal energy work?", "Is geothermal energy renewable?"],
        "responses": [
            "Geothermal energy comes from heat stored beneath the Earth's surface. It's renewable and provides a consistent energy supply.",
            "This type of energy is used for heating, cooling, and electricity generation with minimal environmental impact."
        ]
    },
    {
        "tag": "ocean_energy",
        "patterns": ["What is ocean energy?", "How is ocean energy harnessed?", "Examples of ocean energy?"],
        "responses": [
            "Ocean energy includes tidal, wave, and thermal energy from the sea. It's a promising renewable energy source.",
            "Tidal power plants and wave converters are examples of technologies used to harness ocean energy."
        ]
    },
    {
        "tag": "carbon_neutral",
        "patterns": ["What does carbon-neutral mean?", "How to become carbon-neutral?", "Importance of carbon neutrality?"],
        "responses": [
            "Carbon neutrality means balancing carbon emissions by removing or offsetting an equivalent amount of carbon dioxide.",
            "You can achieve carbon neutrality by using renewable energy, planting trees, and supporting carbon offset programs."
        ]
    },
    {
        "tag": "green_building",
        "patterns": ["What are green buildings?", "Features of green buildings?", "Why are green buildings important?"],
        "responses": [
            "Green buildings are designed to be environmentally friendly and resource-efficient throughout their lifecycle.",
            "Features include energy-efficient systems, renewable energy use, and sustainable materials to minimize environmental impact."
        ]
    },
    {
        "tag": "electric_vehicles",
        "patterns": ["What are electric vehicles?", "Benefits of electric vehicles?", "How do EVs work?"],
        "responses": [
            "Electric vehicles (EVs) run on electricity stored in batteries instead of fossil fuels, reducing carbon emissions.",
            "EVs are cost-efficient, environmentally friendly, and contribute to reducing air pollution in urban areas."
        ]
    },
    {
        "tag": "eco_friendly_products",
        "patterns": ["What are eco-friendly products?", "Examples of eco-friendly products?", "How to choose eco-friendly items?"],
        "responses": [
            "Eco-friendly products are made with sustainable materials and processes, reducing harm to the environment.",
            "Reusable bags, bamboo toothbrushes, and solar-powered devices are examples of eco-friendly items."
        ]
    },
    {
        "tag": "composting",
        "patterns": ["What is composting?", "How does composting work?", "Benefits of composting?"],
        "responses": [
            "Composting involves recycling organic waste into nutrient-rich soil for plants. It's a great way to reduce waste.",
            "By composting food scraps and garden waste, you can reduce landfill contributions and support healthier soil."
        ]
    },
    {
        "tag": "urban_gardening",
        "patterns": ["What is urban gardening?", "How to start urban gardening?", "Benefits of urban gardening?"],
        "responses": [
            "Urban gardening involves growing plants in city spaces, like rooftops, balconies, or small gardens.",
            "It promotes sustainability by reducing food transportation and providing fresh, local produce."
        ]
    },
    {
        "tag": "green_jobs",
        "patterns": ["What are green jobs?", "Examples of green jobs?", "How to get into green careers?"],
        "responses": [
            "Green jobs focus on sustainability, such as renewable energy, conservation, and environmental protection roles.",
            "Solar panel installation, wind turbine maintenance, and sustainability consulting are examples of green jobs."
        ]
    },
    {
        "tag": "climate_action",
        "patterns": ["What is climate action?", "How to take climate action?", "Importance of climate action?"],
        "responses": [
            "Climate action involves efforts to reduce greenhouse gas emissions and adapt to climate change impacts.",
            "Simple actions like reducing waste, conserving energy, and supporting green policies can make a big difference."
        ]
    },
    {
        "tag": "recycling",
        "patterns": ["What can be recycled?", "How does recycling work?", "Why is recycling important?"],
        "responses": [
            "Recycling involves converting waste into reusable materials, helping reduce landfill waste and conserve resources.",
            "Materials like paper, plastic, glass, and metals can often be recycled to create new products."
        ]
    },
    {
        "tag": "tree_planting",
        "patterns": ["Why is tree planting important?", "How does tree planting help the environment?", "Benefits of trees?"],
        "responses": [
            "Trees absorb carbon dioxide, provide oxygen, and support biodiversity. Planting trees helps combat climate change.",
            "By planting trees, you can improve air quality, reduce urban heat, and create habitats for wildlife."
        ]
    },
    {
        "tag": "sustainable_transport",
        "patterns": ["What is sustainable transport?", "Examples of sustainable transport?", "Why is sustainable transport important?"],
        "responses": [
            "Sustainable transport focuses on reducing emissions through methods like cycling, walking, and using electric vehicles.",
            "Public transport, carpooling, and biking are eco-friendly ways to travel and reduce carbon footprints."
        ]
    },
    {
        "tag": "energy_efficiency",
        "patterns": ["What is energy efficiency?", "How to improve energy efficiency?", "Why is energy efficiency important?"],
        "responses": [
            "Energy efficiency means using less energy to perform the same tasks, reducing waste and saving money.",
            "Upgrading appliances, insulating homes, and using LED lighting are ways to improve energy efficiency."
        ]
    },
    {
        "tag": "sustainable_fashion",
        "patterns": ["What is sustainable fashion?", "How to support sustainable fashion?", "Why is fast fashion harmful?"],
        "responses": [
            "Sustainable fashion focuses on ethical production, eco-friendly materials, and reducing waste in the clothing industry.",
            "Choosing quality over quantity, supporting ethical brands, and recycling clothes are steps toward sustainable fashion."
        ]
    },
    {
        "tag": "renewable_innovation",
        "patterns": ["What are new innovations in renewable energy?", "Latest technologies in green energy?", "How is green technology evolving?"],
        "responses": [
            "Innovations like floating solar panels, advanced wind turbines, and energy storage solutions are transforming renewable energy.",
            "Green technology is evolving to become more efficient and accessible, driving global sustainability efforts."
        ]
    },
    {
        "tag": "normal_conversation",
        "patterns": ["How are you?", "What's your name?", "Can you help me?", "Tell me something interesting.", "What can you do?"],
        "responses": [
            "I'm here to help you with green technology and sustainability tips. Ask me anything!",
            "I'm your green assistant, ready to guide you toward an eco-friendly lifestyle. How can I assist?"
        ]
    },
    {
        "tag": "climate_change_policy",
        "patterns": ["What is climate change policy?", "Why are climate change policies important?", "How do policies impact the climate?"],
        "responses": [
            "Climate change policies are laws and regulations aimed at reducing carbon emissions and mitigating the effects of climate change.",
            "These policies are vital in guiding nations and organizations toward sustainable development and environmental protection."
        ]
    },
    {
        "tag": "greenhouse_gases",
        "patterns": ["What are greenhouse gases?", "How do greenhouse gases affect the environment?", "Examples of greenhouse gases?"],
        "responses": [
            "Greenhouse gases trap heat in the atmosphere and contribute to global warming. Common examples include carbon dioxide, methane, and nitrous oxide.",
            "These gases are primarily emitted through human activities like burning fossil fuels and deforestation."
        ]
    },
    {
        "tag": "clean_energy",
        "patterns": ["What is clean energy?", "Examples of clean energy?", "How can we transition to clean energy?"],
        "responses": [
            "Clean energy refers to energy sources that do not pollute the environment, such as solar, wind, and hydro power.",
            "Transitioning to clean energy involves investing in renewable energy, improving energy efficiency, and reducing reliance on fossil fuels."
        ]
    },
    {
        "tag": "zero_waste_living",
        "patterns": ["What is zero waste living?", "How to live a zero waste lifestyle?", "Why is zero waste important?"],
        "responses": [
            "Zero waste living involves reducing waste by reusing, recycling, and composting as much as possible.",
            "It helps decrease the amount of waste sent to landfills, conserves natural resources, and reduces pollution."
        ]
    },
    {
        "tag": "sustainable_agriculture",
        "patterns": ["What is sustainable agriculture?", "How does sustainable farming work?", "Why is sustainable agriculture important?"],
        "responses": [
            "Sustainable agriculture focuses on farming practices that preserve the environment, promote soil health, and reduce pesticide use.",
            "This method supports biodiversity and ensures that future generations can continue farming without depleting natural resources."
        ]
    },
    {
        "tag": "carbon_capture",
        "patterns": ["What is carbon capture?", "How does carbon capture technology work?", "Why is carbon capture important?"],
        "responses": [
            "Carbon capture involves capturing carbon dioxide emissions at their source and storing them underground to prevent them from entering the atmosphere.",
            "It plays a critical role in combating climate change by reducing greenhouse gas concentrations."
        ]
    },
    {
        "tag": "green_tech_innovation",
        "patterns": ["What are the latest innovations in green technology?", "New developments in green technology?", "How is green technology evolving?"],
        "responses": [
            "Recent innovations include floating solar panels, next-generation wind turbines, and carbon-neutral manufacturing technologies.",
            "Green tech is evolving to make energy systems more efficient and affordable while reducing environmental impact."
        ]
    },
    {
        "tag": "eco_conscious_communities",
        "patterns": ["What are eco-conscious communities?", "How can communities become more eco-friendly?", "Examples of eco-conscious communities?"],
        "responses": [
            "Eco-conscious communities focus on sustainable living through initiatives like renewable energy, local food production, and waste reduction.",
            "Some communities have adopted solar power grids, zero-waste programs, and sustainable transportation to minimize their environmental impact."
        ]
    },
    {
        "tag": "eco_tourism",
        "patterns": ["What is eco-tourism?", "Benefits of eco-tourism?", "How does eco-tourism help the environment?"],
        "responses": [
            "Eco-tourism promotes travel to natural areas that conserves the environment and improves the well-being of local communities.",
            "It encourages responsible travel practices, minimizes environmental impact, and supports conservation efforts."
        ]
    },
    {
        "tag": "water_conservation",
        "patterns": ["What is water conservation?", "How can I conserve water?", "Why is water conservation important?"],
        "responses": [
            "Water conservation is the practice of using water efficiently and reducing waste.",
            "You can conserve water by fixing leaks, using low-flow appliances, and collecting rainwater for irrigation."
        ]
    },
    {
        "tag": "environmental_education",
        "patterns": ["What is environmental education?", "Why is environmental education important?", "How can I get involved in environmental education?"],
        "responses": [
            "Environmental education raises awareness about environmental issues and teaches sustainable practices.",
            "It empowers individuals and communities to make informed decisions to protect the environment."
        ]
    },
    {
        "tag": "air_quality",
        "patterns": ["What is air quality?", "How does air quality affect health?", "How can we improve air quality?"],
        "responses": [
            "Air quality refers to the cleanliness of the air and the concentration of pollutants.",
            "We can improve air quality by reducing emissions from vehicles and factories and promoting clean energy."
        ]
    },
    {
        "tag": "sustainable_fishing",
        "patterns": ["What is sustainable fishing?", "Why is sustainable fishing important?", "How can we practice sustainable fishing?"],
        "responses": [
            "Sustainable fishing involves catching fish in ways that preserve fish populations and protect marine ecosystems.",
            "It includes measures like limiting fishing quotas, using eco-friendly fishing gear, and preventing overfishing."
        ]
    },
    {
        "tag": "circular_economy",
        "patterns": ["What is the circular economy?", "How does the circular economy work?", "Why is the circular economy important?"],
        "responses": [
            "The circular economy focuses on reducing waste by reusing, recycling, and regenerating products and materials.",
            "It helps reduce environmental impact and ensures that resources are used sustainably."
        ]
    },
    {
        "tag": "electric_bikes",
        "patterns": ["What are electric bikes?", "How do electric bikes work?", "Why should I use an electric bike?"],
        "responses": [
            "Electric bikes are bicycles powered by a battery, making them easier to ride over long distances and hilly terrains.",
            "They are a sustainable alternative to traditional transportation and can reduce your carbon footprint."
        ]
    },
    {
        "tag": "sustainable_materials",
        "patterns": ["What are sustainable materials?", "Examples of sustainable materials?", "Why are sustainable materials important?"],
        "responses": [
            "Sustainable materials are those that are produced with minimal environmental impact and are renewable or recyclable.",
            "Examples include bamboo, hemp, and recycled metals and plastics."
        ]
    },
    {
        "tag": "green_financing",
        "patterns": ["What is green financing?", "How does green financing work?", "Why is green financing important?"],
        "responses": [
            "Green financing refers to investments in projects that have positive environmental outcomes, such as renewable energy or energy efficiency projects.",
            "It supports the transition to a low-carbon economy by funding sustainable initiatives."
        ]
    },
    {
        "tag": "greenwashing",
        "patterns": ["What is greenwashing?", "How to identify greenwashing?", "Why is greenwashing harmful?"],
        "responses": [
            "Greenwashing is when companies falsely claim their products or practices are environmentally friendly to attract consumers.",
            "It misleads consumers and undermines genuine environmental efforts."
        ]
    },
    {
        "tag": "sustainable_development_goals",
        "patterns": ["What are the sustainable development goals?", "Why are the sustainable development goals important?", "How do the SDGs relate to climate change?"],
        "responses": [
            "The SDGs are a set of 17 global goals adopted by the United Nations to address social, environmental, and economic challenges.",
            "They aim to create a more sustainable, equitable, and prosperous world for all."
        ]
    },
    {
        "tag": "green_building_materials",
        "patterns": ["What are green building materials?", "Examples of green building materials?", "Why are green building materials important?"],
        "responses": [
            "Green building materials are eco-friendly materials that are sustainable, energy-efficient, and reduce environmental impact.",
            "Examples include recycled steel, bamboo, and low-impact insulation materials."
        ]
    },
    {
        "tag": "solar_farming",
        "patterns": ["What is solar farming?", "How does solar farming work?", "Benefits of solar farming?"],
        "responses": [
            "Solar farming involves using large areas of land to install solar panels to generate renewable energy.",
            "It provides a sustainable energy source while also creating economic opportunities for landowners and communities."
        ]
    },
    {
        "tag": "eco_friendly_transportation",
        "patterns": ["What is eco-friendly transportation?", "Examples of eco-friendly transportation?", "Why should I use eco-friendly transportation?"],
        "responses": [
            "Eco-friendly transportation includes modes like electric vehicles, cycling, and public transport.",
            "It reduces air pollution, conserves resources, and helps mitigate climate change."
        ]
    },
    {
        "tag": "clean_water",
        "patterns": ["What is clean water?", "How can we ensure clean water?", "Why is clean water important?"],
        "responses": [
            "Clean water is essential for health and sanitation. It can be ensured by reducing pollution and investing in water treatment technologies.",
            "Clean water is crucial for human life, agriculture, and ecosystems, and its preservation is key to sustainable development."
        ]
    },
    {
        "tag": "forest_conservation",
        "patterns": ["Why is forest conservation important?", "How can we conserve forests?", "What are the benefits of forests?"],
        "responses": [
            "Forests play a crucial role in absorbing carbon dioxide and providing habitats for wildlife. Conserving forests helps fight climate change.",
            "We can conserve forests by reducing deforestation, supporting sustainable logging practices, and protecting forested areas."
        ]
    },
    {
        "tag": "sustainable_urban_planning",
        "patterns": ["What is sustainable urban planning?", "How does sustainable urban planning work?", "Why is sustainable urban planning important?"],
        "responses": [
            "Sustainable urban planning focuses on creating cities that are energy-efficient, environmentally friendly, and socially equitable.",
            "It includes promoting green spaces, improving public transportation, and using sustainable building materials."
        ]
    },
    {
        "tag": "organic_farming",
        "patterns": ["What is organic farming?", "How is organic farming different?", "Why is organic farming better for the environment?"],
        "responses": [
            "Organic farming avoids the use of synthetic pesticides and fertilizers, focusing on natural methods to protect the environment.",
            "It helps preserve soil health, reduces water pollution, and promotes biodiversity."
        ]
    },
    {
        "tag": "green_business_practices",
        "patterns": ["What are green business practices?", "How can businesses become more sustainable?", "Why are green business practices important?"],
        "responses": [
            "Green business practices involve reducing waste, conserving energy, and adopting sustainable supply chain methods.",
            "By going green, businesses can reduce their environmental impact, attract eco-conscious consumers, and save money in the long run."
        ]
    },
    {
        "tag": "energy_efficiency",
        "patterns": ["What is energy efficiency?", "How can I improve energy efficiency?", "Why is energy efficiency important?"],
        "responses": [
            "Energy efficiency refers to using less energy to perform the same tasks, such as using LED lights or energy-efficient appliances.",
            "Improving energy efficiency helps reduce energy consumption, lower bills, and decrease environmental impact."
        ]
    },
    {
        "tag": "green_transportation_infrastructure",
        "patterns": ["What is green transportation infrastructure?", "Examples of green transportation infrastructure?", "Why is green transportation important?"],
        "responses": [
            "Green transportation infrastructure includes bike lanes, electric vehicle charging stations, and public transit systems.",
            "It helps reduce reliance on fossil fuels and decreases air pollution, promoting a more sustainable transport system."
        ]
    },
    {
        "tag": "carbon_neutrality",
        "patterns": ["What is carbon neutrality?", "How can I achieve carbon neutrality?", "Why is carbon neutrality important?"],
        "responses": [
            "Carbon neutrality means balancing the amount of carbon dioxide emitted with the amount removed from the atmosphere.",
            "Individuals and organizations can achieve carbon neutrality by reducing emissions and offsetting remaining emissions through projects like tree planting."
        ]
    },
    {
        "tag": "waste_management",
        "patterns": ["What is waste management?", "How can we improve waste management?", "Why is waste management important?"],
        "responses": [
            "Waste management involves collecting, treating, and disposing of waste in ways that minimize environmental impact.",
            "Improved waste management reduces pollution, conserves resources, and supports recycling and composting initiatives."
        ]
    },
    {
        "tag": "energy_storage",
        "patterns": ["What is energy storage?", "Why is energy storage important?", "How does energy storage work?"],
        "responses": [
            "Energy storage involves saving excess energy for later use, typically through batteries or pumped hydro storage.",
            "It helps balance energy supply and demand, supports renewable energy integration, and ensures a reliable power supply."
        ]
    },
    {
        "tag": "sustainable_design",
        "patterns": ["What is sustainable design?", "Why is sustainable design important?", "Examples of sustainable design?"],
        "responses": [
            "Sustainable design focuses on creating products, buildings, and systems that minimize environmental impact and use resources efficiently.",
            "Examples include energy-efficient buildings, water-saving technologies, and the use of sustainable materials."
        ]
    },
    {
        "tag": "clean_technology_startups",
        "patterns": ["What are clean technology startups?", "How do clean technology startups help the environment?", "Examples of clean technology startups?"],
        "responses": [
            "Clean technology startups focus on developing innovative solutions to reduce environmental impact and promote sustainability.",
            "Examples include companies working on renewable energy technologies, electric vehicles, and sustainable materials."
        ]
    },
    {
        "tag": "sustainable_financing",
        "patterns": ["What is sustainable financing?", "How does sustainable financing work?", "Why is sustainable financing important?"],
        "responses": [
            "Sustainable financing involves investing in projects that contribute to environmental sustainability and social responsibility.",
            "It helps fund initiatives like renewable energy, waste management, and sustainable agriculture."
        ]
    },
    {
        "tag": "green_retail",
        "patterns": ["What is green retail?", "How can retail businesses be more sustainable?", "Examples of green retail practices?"],
        "responses": [
            "Green retail involves adopting eco-friendly practices in the retail industry, such as reducing packaging and offering sustainable products.",
            "Examples include stores that use recyclable materials, reduce waste, and support ethical production."
        ]
    },
    {
        "tag": "green_cities",
        "patterns": ["What are green cities?", "How can we create green cities?", "Why are green cities important?"],
        "responses": [
            "Green cities focus on sustainable urban development, including green spaces, energy-efficient buildings, and renewable energy systems.",
            "They improve quality of life, reduce environmental impact, and help mitigate climate change."
        ]
    },
    {
        "tag": "solar_energy_investment",
        "patterns": ["Why should I invest in solar energy?", "What are the benefits of investing in solar energy?", "How can I invest in solar energy?"],
        "responses": [
            "Investing in solar energy helps reduce energy costs, provides long-term savings, and supports the transition to renewable energy.",
            "You can invest in solar through rooftop installations or by purchasing shares in solar energy companies."
        ]
    },
    {
        "tag": "green_energy_jobs",
        "patterns": ["What are green energy jobs?", "How do I get a job in the green energy sector?", "What skills are needed for green energy jobs?"],
        "responses": [
            "Green energy jobs involve working in industries related to renewable energy, energy efficiency, and environmental conservation.",
            "Skills in engineering, renewable energy technologies, and environmental science are highly valuable in this sector."
        ]
    },
    {
        "tag": "clean_air_initiatives",
        "patterns": ["What are clean air initiatives?", "How can we reduce air pollution?", "Why are clean air initiatives important?"],
        "responses": [
            "Clean air initiatives focus on reducing air pollution through measures like reducing emissions from vehicles, industry, and energy production.",
            "They help improve public health, reduce respiratory diseases, and combat climate change."
        ]
    },
    {
        "tag": "green_jobs",
        "patterns": ["What are green jobs?", "How can I get a green job?", "Why are green jobs important?"],
        "responses": [
            "Green jobs are positions that contribute to environmental sustainability, such as in renewable energy, recycling, or environmental protection.",
            "These jobs help drive the transition to a low-carbon economy and promote environmental well-being."
        ]
    },
    {
        "tag": "sustainable_waste_technologies",
        "patterns": ["What are sustainable waste technologies?", "How can technology help with waste management?", "Examples of sustainable waste technologies?"],
        "responses": [
            "Sustainable waste technologies include methods like composting, waste-to-energy, and recycling technologies.",
            "They help reduce waste sent to landfills and minimize the environmental impact of waste disposal."
        ]
    },
    {
        "tag": "carbon_footprint_calculator",
        "patterns": ["How do I calculate my carbon footprint?", "What is a carbon footprint calculator?", "Why should I calculate my carbon footprint?"],
        "responses": [
            "A carbon footprint calculator helps estimate the amount of greenhouse gases your activities produce.",
            "Calculating your carbon footprint can help you identify ways to reduce emissions and live more sustainably."
        ]
    },
    {
        "tag": "alternative_transportation",
        "patterns": ["What are alternative transportation options?", "Examples of alternative transportation?", "Why should I use alternative transportation?"],
        "responses": [
            "Alternative transportation includes modes like cycling, walking, and electric vehicles, which reduce reliance on fossil fuels.",
            "They help reduce carbon emissions, conserve energy, and improve air quality."
        ]
    },
    {
        "tag": "green_investment",
        "patterns": ["What is green investment?", "Why should I invest in green companies?", "How does green investment work?"],
        "responses": [
            "Green investment focuses on funding projects and companies that promote sustainability and environmentally-friendly practices.",
            "It includes investing in renewable energy, sustainable agriculture, and eco-friendly businesses to create long-term environmental benefits."
        ]
    },
    {
        "tag": "eco_tourism",
        "patterns": ["What is eco-tourism?", "How does eco-tourism help the environment?", "Examples of eco-tourism destinations?"],
        "responses": [
            "Eco-tourism promotes responsible travel to natural areas, helping conserve the environment and improve the well-being of local communities.",
            "It includes activities like wildlife watching, hiking, and staying at eco-friendly accommodations."
        ]
    },
    {
        "tag": "recyclable_materials",
        "patterns": ["What materials are recyclable?", "How can I recycle materials?", "Why is recycling important?"],
        "responses": [
            "Common recyclable materials include paper, cardboard, plastic, glass, and metals like aluminum.",
            "Recycling helps conserve resources, reduce pollution, and decrease the amount of waste sent to landfills."
        ]
    },
    {
        "tag": "sustainable_cooking",
        "patterns": ["What is sustainable cooking?", "How can I cook sustainably?", "Why should I cook sustainably?"],
        "responses": [
            "Sustainable cooking focuses on using locally-sourced ingredients, minimizing food waste, and reducing energy consumption in the kitchen.",
            "It helps reduce your environmental impact while promoting health and supporting local economies."
        ]
    },
    {
        "tag": "green_technology_innovation",
        "patterns": ["What is green technology innovation?", "How can technology help the environment?", "Examples of green technology innovations?"],
        "responses": [
            "Green technology innovation involves developing new technologies that reduce environmental harm and promote sustainability.",
            "Examples include advancements in renewable energy, energy-efficient appliances, and carbon capture technologies."
        ]
    },
    {
        "tag": "electric_bikes",
        "patterns": ["What are electric bikes?", "How do electric bikes work?", "Benefits of electric bikes?"],
        "responses": [
            "Electric bikes are bicycles with a battery-powered motor that assists with pedaling.",
            "They provide an eco-friendly alternative to traditional transportation, reduce traffic congestion, and promote healthier commuting."
        ]
    },
    {
        "tag": "solar_power_storage",
        "patterns": ["What is solar power storage?", "How does solar power storage work?", "Why is solar power storage important?"],
        "responses": [
            "Solar power storage involves storing excess solar energy in batteries for use during periods without sunlight.",
            "It allows for more reliable use of solar energy and helps balance the grid by providing power when needed."
        ]
    },
    {
        "tag": "green_buildings",
        "patterns": ["What are green buildings?", "How do green buildings help the environment?", "Examples of green buildings?"],
        "responses": [
            "Green buildings are designed with energy efficiency, sustainable materials, and environmental conservation in mind.",
            "Examples include buildings with solar panels, energy-efficient lighting, and water-saving systems."
        ]
    },
    {
        "tag": "plastic_pollution",
        "patterns": ["What is plastic pollution?", "How can we reduce plastic pollution?", "Why is plastic pollution a problem?"],
        "responses": [
            "Plastic pollution occurs when plastic waste accumulates in the environment, harming wildlife and ecosystems.",
            "We can reduce plastic pollution by using alternatives like reusable bags and bottles, recycling, and reducing plastic production."
        ]
    },
    {
        "tag": "low_carbon_lifestyles",
        "patterns": ["What is a low-carbon lifestyle?", "How can I live a low-carbon lifestyle?", "Why should I adopt a low-carbon lifestyle?"],
        "responses": [
            "A low-carbon lifestyle involves reducing your carbon emissions by using energy-efficient appliances, minimizing waste, and choosing sustainable transportation options.",
            "It helps mitigate climate change and supports a sustainable, eco-friendly future."
        ]
    },
    {
        "tag": "green_energy_credits",
        "patterns": ["What are green energy credits?", "How do green energy credits work?", "Why are green energy credits important?"],
        "responses": [
            "Green energy credits are certificates representing the environmental benefits of renewable energy generation.",
            "They help encourage the adoption of renewable energy by allowing individuals and organizations to offset their carbon emissions."
        ]
    },
    {
        "tag": "water_conservation_techniques",
        "patterns": ["What are water conservation techniques?", "How can I conserve water?", "Why is water conservation important?"],
        "responses": [
            "Water conservation techniques include using water-efficient appliances, fixing leaks, and reducing water usage in daily activities.",
            "It helps preserve this precious resource, reduce costs, and protect ecosystems from water scarcity."
        ]
    },
    {
        "tag": "carbon_offsetting",
        "patterns": ["What is carbon offsetting?", "How can I offset my carbon emissions?", "Why is carbon offsetting important?"],
        "responses": [
            "Carbon offsetting involves compensating for your carbon emissions by supporting projects that reduce or remove carbon from the atmosphere.",
            "It helps mitigate climate change by funding projects like reforestation and renewable energy."
        ]
    },
    {
        "tag": "sustainable_fishing",
        "patterns": ["What is sustainable fishing?", "How can fishing be more sustainable?", "Why is sustainable fishing important?"],
        "responses": [
            "Sustainable fishing practices focus on maintaining fish populations and protecting marine ecosystems.",
            "It involves methods like reducing overfishing, using eco-friendly fishing gear, and supporting marine conservation efforts."
        ]
    },
    {
        "tag": "green_chemicals",
        "patterns": ["What are green chemicals?", "Why are green chemicals better for the environment?", "Examples of green chemicals?"],
        "responses": [
            "Green chemicals are substances that are designed to be less harmful to the environment and human health.",
            "They include biodegradable cleaning products, sustainable pesticides, and eco-friendly solvents."
        ]
    },
    {
        "tag": "eco-friendly_products",
        "patterns": ["What are eco-friendly products?", "How can I identify eco-friendly products?", "Why should I use eco-friendly products?"],
        "responses": [
            "Eco-friendly products are those that have a minimal impact on the environment, such as products made from sustainable materials or those that are energy-efficient.",
            "Using eco-friendly products helps reduce waste, conserve resources, and promote sustainability."
        ]
    },
    {
        "tag": "reforestation",
        "patterns": ["What is reforestation?", "How does reforestation help the environment?", "Why is reforestation important?"],
        "responses": [
            "Reforestation involves planting trees in areas that have been depleted of forests.",
            "It helps absorb carbon dioxide, restore biodiversity, and improve air and water quality."
        ]
    },
    {
        "tag": "eco_friendly_transportation",
        "patterns": ["What is eco-friendly transportation?", "How can I travel more sustainably?", "Examples of eco-friendly transportation?"],
        "responses": [
            "Eco-friendly transportation includes options like electric vehicles, biking, walking, and public transit.",
            "It helps reduce air pollution, greenhouse gas emissions, and dependence on fossil fuels."
        ]
    },
    {
        "tag": "energy_smart_homes",
        "patterns": ["What are energy-smart homes?", "How do energy-smart homes work?", "Why are energy-smart homes important?"],
        "responses": [
            "Energy-smart homes use technologies like smart thermostats, solar panels, and energy-efficient appliances to reduce energy consumption.",
            "They help lower utility bills, reduce environmental impact, and promote energy independence."
        ]
    },
    {
        "tag": "green_finance",
        "patterns": ["What is green finance?", "Why is green finance important?", "How can I invest in green finance?"],
        "responses": [
            "Green finance involves funding projects that benefit the environment, such as renewable energy, waste management, and sustainable agriculture.",
            "It helps mobilize capital for projects that promote environmental sustainability and combat climate change."
        ]
    },
    {
        "tag": "zero_waste_lifestyle",
        "patterns": ["What is a zero waste lifestyle?", "How can I live a zero waste lifestyle?", "Why should I adopt a zero waste lifestyle?"],
        "responses": [
            "A zero waste lifestyle focuses on minimizing waste by reusing, recycling, and composting materials.",
            "It helps reduce landfill waste, conserve resources, and reduce pollution."
        ]
    },
    {
        "tag": "alternative_energy_sources",
        "patterns": ["What are alternative energy sources?", "Examples of alternative energy sources?", "Why should we use alternative energy?"],
        "responses": [
            "Alternative energy sources include solar, wind, geothermal, and tidal energy.",
            "They help reduce reliance on fossil fuels, decrease greenhouse gas emissions, and promote a sustainable energy future."
        ]
    },
    {
        "tag": "clean_technology",
        "patterns": ["What is clean technology?", "How does clean technology work?", "Examples of clean technology?"],
        "responses": [
            "Clean technology focuses on developing products and services that use fewer resources and generate less waste or pollution.",
            "Examples include solar panels, electric vehicles, and energy-efficient appliances."
        ]
    },
    {
        "tag": "green_certification",
        "patterns": ["What is green certification?", "Why is green certification important?", "How can I get green certification for my business?"],
        "responses": [
            "Green certification is a recognition given to businesses that meet certain environmental standards in their operations.",
            "It helps businesses demonstrate their commitment to sustainability and can improve their marketability."
        ]
    },
    {
        "tag": "biodegradable_materials",
        "patterns": ["What are biodegradable materials?", "Why are biodegradable materials important?", "Examples of biodegradable materials?"],
        "responses": [
            "Biodegradable materials break down naturally over time and do not cause long-term harm to the environment.",
            "Examples include organic waste, paper, and certain types of plastics made from natural sources."
        ]
    },
    {
        "tag": "green_jobs",
        "patterns": ["What are green jobs?", "How do I find green jobs?", "Why are green jobs important?"],
        "responses": [
            "Green jobs focus on work that contributes to protecting the environment or promoting sustainability.",
            "They can be found in fields like renewable energy, waste management, and environmental conservation."
        ]
    },
    {
        "tag": "climate_justice",
        "patterns": ["What is climate justice?", "Why is climate justice important?", "How can we achieve climate justice?"],
        "responses": [
            "Climate justice addresses the unequal effects of climate change on vulnerable communities and aims for fair solutions.",
            "It includes ensuring that those who contribute the least to climate change are not disproportionately affected."
        ]
    },
    {
        "tag": "eco_labeling",
        "patterns": ["What is eco-labeling?", "How does eco-labeling help consumers?", "Examples of eco-labels?"],
        "responses": [
            "Eco-labeling is the practice of certifying products that meet specific environmental standards.",
            "It helps consumers make informed choices and supports businesses that prioritize sustainability."
        ]
    },
    {
        "tag": "green_transportation_infrastructure",
        "patterns": ["What is green transportation infrastructure?", "How does green transportation infrastructure work?", "Examples of green transportation infrastructure?"],
        "responses": [
            "Green transportation infrastructure includes elements like electric vehicle charging stations, bike lanes, and efficient public transit systems.",
            "It helps reduce traffic congestion, lowers emissions, and promotes sustainable travel options."
        ]
    },
    {
        "tag": "clean_water_initiatives",
        "patterns": ["What are clean water initiatives?", "How do clean water initiatives help the environment?", "Why is clean water important for sustainability?"],
        "responses": [
            "Clean water initiatives focus on providing safe, sustainable, and accessible water sources.",
            "They help protect aquatic ecosystems, reduce waterborne diseases, and promote human health."
        ]
    },
    {
        "tag": "green_infrastructure",
        "patterns": ["What is green infrastructure?", "How does green infrastructure work?", "Examples of green infrastructure?"],
        "responses": [
            "Green infrastructure uses natural systems to manage water and reduce environmental impact.",
            "Examples include green roofs, permeable pavements, and urban forests."
        ]
    },
    {
        "tag": "circular_economy",
        "patterns": ["What is a circular economy?", "How does a circular economy work?", "Why is a circular economy important?"],
        "responses": [
            "A circular economy focuses on reducing waste and reusing resources, unlike the traditional linear economy.",
            "It helps conserve resources, reduce pollution, and minimize waste."
        ]
    },
    {
        "tag": "energy_efficiency_in_buildings",
        "patterns": ["What is energy efficiency in buildings?", "How can I improve energy efficiency in my home?", "Why is energy efficiency important in buildings?"],
        "responses": [
            "Energy efficiency in buildings involves using less energy to provide the same level of comfort or service.",
            "It can be achieved through insulation, energy-efficient appliances, and better lighting systems."
        ]
    },
    {
        "tag": "green_cities",
        "patterns": ["What is a green city?", "How can a city become green?", "Examples of green cities?"],
        "responses": [
            "Green cities incorporate sustainable practices like renewable energy, green spaces, and efficient waste management.",
            "Examples include Copenhagen, Amsterdam, and Vancouver."
        ]
    },
    {
        "tag": "food_sustainability",
        "patterns": ["What is food sustainability?", "Why is food sustainability important?", "How can I eat sustainably?"],
        "responses": [
            "Food sustainability involves producing and consuming food in ways that are healthy for both people and the planet.",
            "It includes practices like eating locally grown food, reducing food waste, and supporting organic farming."
        ]
    },
    {
        "tag": "green_energy_investment",
        "patterns": ["What is green energy investment?", "How can I invest in green energy?", "Why should I invest in green energy?"],
        "responses": [
            "Green energy investment involves putting money into projects and companies that focus on renewable energy sources like solar, wind, and geothermal.",
            "It supports the transition to a sustainable energy future and can generate financial returns."
        ]
    },
    {
        "tag": "smart_grids",
        "patterns": ["What is a smart grid?", "How do smart grids work?", "Why are smart grids important?"],
        "responses": [
            "A smart grid is an electricity supply network that uses digital technology to manage the distribution of electricity more efficiently.",
            "It helps reduce energy consumption, improve reliability, and integrate renewable energy sources."
        ]
    },
    {
        "tag": "sustainable_manufacturing",
        "patterns": ["What is sustainable manufacturing?", "How can manufacturing be more sustainable?", "Why is sustainable manufacturing important?"],
        "responses": [
            "Sustainable manufacturing focuses on producing goods in ways that minimize environmental impact and conserve resources.",
            "It involves using energy-efficient processes, reducing waste, and sourcing materials responsibly."
        ]
    },
    {
        "tag": "green_financial_products",
        "patterns": ["What are green financial products?", "How can I invest in green financial products?", "Why should I consider green financial products?"],
        "responses": [
            "Green financial products are investment options that focus on supporting sustainable and eco-friendly businesses and projects.",
            "They can include green bonds, sustainable mutual funds, and socially responsible investment portfolios."
        ]
    },
    {
        "tag": "zero_emissions_transportation",
        "patterns": ["What is zero emissions transportation?", "Examples of zero emissions transportation?", "Why is zero emissions transportation important?"],
        "responses": [
            "Zero emissions transportation refers to vehicles that produce no direct emissions, such as electric vehicles (EVs) and hydrogen fuel cell vehicles.",
            "It helps reduce air pollution and reliance on fossil fuels."
        ]
    },
    {
        "tag": "green_building_design",
        "patterns": ["What is green building design?", "How can I design a green building?", "Why is green building design important?"],
        "responses": [
            "Green building design incorporates sustainable materials, energy-efficient systems, and waste-reducing features into building plans.",
            "It helps lower environmental impact, reduce energy use, and improve indoor air quality."
        ]
    },
    {
        "tag": "sustainable_farming",
        "patterns": ["What is sustainable farming?", "How does sustainable farming work?", "Why is sustainable farming important?"],
        "responses": [
            "Sustainable farming practices focus on producing food in ways that maintain environmental health, support farmers' livelihoods, and preserve biodiversity.",
            "It includes methods like crop rotation, organic farming, and integrated pest management."
        ]
    },
    {
        "tag": "environmental_education",
        "patterns": ["What is environmental education?", "Why is environmental education important?", "How can I get involved in environmental education?"],
        "responses": [
            "Environmental education focuses on teaching individuals and communities about the environment and sustainable practices.",
            "It helps raise awareness about environmental issues and empowers people to take action."
        ]
    },
    {
        "tag": "green_innovation",
        "patterns": ["What is green innovation?", "How can I innovate sustainably?", "Examples of green innovations?"],
        "responses": [
            "Green innovation involves developing new technologies, products, and practices that minimize environmental impact.",
            "Examples include electric vehicles, biodegradable packaging, and green energy solutions."
        ]
    },
    {
        "tag": "carbon_neutral",
        "patterns": ["What is carbon neutral?", "How can a company become carbon neutral?", "Why is being carbon neutral important?"],
        "responses": [
            "Carbon neutral means balancing the amount of carbon emitted with the amount removed from the atmosphere.",
            "Companies can achieve carbon neutrality by reducing emissions and investing in carbon offset programs."
        ]
    },
    {
        "tag": "energy_storage",
        "patterns": ["What is energy storage?", "How does energy storage work?", "Why is energy storage important?"],
        "responses": [
            "Energy storage involves storing energy for later use, typically through batteries or other technologies.",
            "It is crucial for managing renewable energy sources like solar and wind, which are intermittent."
        ]
    },
    {
        "tag": "green_building_materials",
        "patterns": ["What are green building materials?", "Examples of green building materials?", "Why are green building materials important?"],
        "responses": [
            "Green building materials are environmentally friendly products used in construction, such as recycled materials, bamboo, and low-emission paints.",
            "They help reduce energy consumption, waste, and the carbon footprint of buildings."
        ]
    },
    {
        "tag": "environmental_policy",
        "patterns": ["What is environmental policy?", "Why is environmental policy important?", "How do environmental policies promote sustainability?"],
        "responses": [
            "Environmental policy refers to the regulations and actions taken by governments or organizations to protect the environment.",
            "These policies can help promote sustainability by reducing pollution, conserving resources, and encouraging green practices."
        ]
    },
    {
        "tag": "eco_entrepreneurship",
        "patterns": ["What is eco-entrepreneurship?", "How can I start an eco-friendly business?", "Examples of eco-entrepreneurship?"],
        "responses": [
            "Eco-entrepreneurship involves starting and running businesses that focus on sustainability and environmental responsibility.",
            "Examples include creating eco-friendly products, sustainable fashion, and renewable energy solutions."
        ]
    },
    {
        "tag": "greenwashing",
        "patterns": ["What is greenwashing?", "How can I avoid greenwashing?", "Why is greenwashing a problem?"],
        "responses": [
            "Greenwashing is when companies falsely claim to be environmentally friendly to attract customers.",
            "It can mislead consumers and hinder the progress of genuine sustainability efforts."
        ]
    },
    {
        "tag": "green_technology_innovation",
        "patterns": ["What is green technology innovation?", "How can I innovate in green technology?", "Examples of green technology innovations?"],
        "responses": [
            "Green technology innovation focuses on developing new products and services that help reduce environmental harm.",
            "Examples include innovations in renewable energy, sustainable agriculture, and waste management technologies."
        ]
    },
    {
        "tag": "plastic_recycling",
        "patterns": ["What is plastic recycling?", "How can I recycle plastic?", "Why is plastic recycling important?"],
        "responses": [
            "Plastic recycling involves converting waste plastic into new products, reducing the need for new plastic production.",
            "It helps reduce plastic waste, lower pollution, and conserve natural resources."
        ]
    },
    {
        "tag": "green_data_centers",
        "patterns": ["What is a green data center?", "How do green data centers work?", "Why are green data centers important?"],
        "responses": [
            "A green data center uses energy-efficient technologies, renewable energy, and sustainable building materials to minimize environmental impact.",
            "They help reduce energy consumption and the carbon footprint of data storage operations."
        ]
    },
    {
        "tag": "green_energy_policies",
        "patterns": ["What are green energy policies?", "How do green energy policies support sustainability?", "Examples of green energy policies?"],
        "responses": [
            "Green energy policies aim to promote the use of renewable energy sources like solar, wind, and hydro power.",
            "Examples include tax incentives for renewable energy projects and regulations requiring renewable energy adoption."
        ]
    },
    {
        "tag": "sustainable_transport",
        "patterns": ["What is sustainable transport?", "How can transportation be more sustainable?", "Examples of sustainable transportation?"],
        "responses": [
            "Sustainable transport focuses on reducing the environmental impact of travel by using clean energy and promoting efficient modes of transport.",
            "Examples include electric cars, public transportation, cycling, and walking."
        ]
    },
    {
        "tag": "green_chemistry",
        "patterns": ["What is green chemistry?", "How does green chemistry help the environment?", "Examples of green chemistry?"],
        "responses": [
            "Green chemistry involves designing chemical processes that reduce or eliminate the use of hazardous substances and minimize environmental impact.",
            "Examples include using renewable feedstocks and developing non-toxic solvents."
        ]
    },
    {
        "tag": "eco_friendly_investments",
        "patterns": ["What are eco-friendly investments?", "How can I invest in eco-friendly companies?", "Why should I consider eco-friendly investments?"],
        "responses": [
            "Eco-friendly investments focus on companies and projects that prioritize sustainability and environmental responsibility.",
            "They can include green bonds, renewable energy companies, and sustainable agriculture ventures."
        ]
    },
    {
        "tag": "eco_tourism",
        "patterns": ["What is eco-tourism?", "How can I travel sustainably?", "Why is eco-tourism important?"],
        "responses": [
            "Eco-tourism promotes responsible travel to natural areas that conserves the environment and improves the well-being of local communities.",
            "It helps raise awareness about environmental issues and supports conservation efforts."
        ]
    },
    {
        "tag": "green_energy_storage",
        "patterns": ["What is green energy storage?", "How does green energy storage work?", "Why is energy storage important for green energy?"],
        "responses": [
            "Green energy storage involves storing energy produced by renewable sources like solar and wind for use when demand is high or supply is low.",
            "It helps smooth out the intermittent nature of renewable energy and ensures a reliable power supply."
        ]
    },
    {
        "tag": "sustainable_manufacturing_technologies",
        "patterns": ["What are sustainable manufacturing technologies?", "How can manufacturing become more sustainable?", "Examples of sustainable manufacturing technologies?"],
        "responses": [
            "Sustainable manufacturing technologies include energy-efficient machinery, waste reduction techniques, and the use of eco-friendly materials.",
            "Examples include 3D printing, lean manufacturing, and closed-loop systems."
        ]
    },
    {
        "tag": "clean_energy_investment",
        "patterns": ["What is clean energy investment?", "How can I invest in clean energy?", "Why should I invest in clean energy?"],
        "responses": [
            "Clean energy investment focuses on supporting companies and projects that use renewable energy sources like solar, wind, and hydro.",
            "It promotes sustainability and helps transition away from fossil fuels."
        ]
    },
    {
        "tag": "green_water_management",
        "patterns": ["What is green water management?", "How can water management be sustainable?", "Why is green water management important?"],
        "responses": [
            "Green water management involves using natural systems to manage water resources, reduce runoff, and prevent water waste.",
            "It helps conserve water, protect ecosystems, and mitigate flooding."
        ]
    },
    {
        "tag": "solar_energy_systems",
        "patterns": ["What are solar energy systems?", "How do solar energy systems work?", "Why are solar energy systems important?"],
        "responses": [
            "Solar energy systems capture sunlight and convert it into electricity or heat.",
            "They are important because they reduce reliance on fossil fuels and provide a clean source of renewable energy."
        ]
    },
    {
        "tag": "eco_labels",
        "patterns": ["What are eco-labels?", "Why should I look for eco-labels on products?", "Examples of eco-labels?"],
        "responses": [
            "Eco-labels indicate that a product meets specific environmental standards, such as being made from sustainable materials or produced with minimal energy use.",
            "Examples include the Energy Star label, Fair Trade certification, and Forest Stewardship Council (FSC) certification."
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


def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    try:
        tag = clf.predict(input_text)[0]
        for intent in intents:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
    except:
        return "I'm sorry, I didn't understand that. Could you rephrase?"


def chatbot(user_input):
   
    return f"Chatbot: You said: {user_input}"

def main():
    st.title("Green Technology ChatBot")
    st.write("Welcome! I'm here to help you learn about green technology and sustainability.")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        st.text_area(label="Chatbot:", value=msg, height=100, max_chars=None, key=f"chat_{msg[:10]}")  # Show previous messages
    
    user_input = st.text_input("You:", key="user_input")

    if user_input:
        response = chatbot(user_input)
        st.session_state.chat_history.append(f"You: {user_input}")
        st.session_state.chat_history.append(f"{response}")

        st.text_area("Chatbot:", value=response, height=100, max_chars=None, key="chatbot_response")

        if "goodbye" in response.lower() or "bye" in response.lower():
            st.write("Thank you for using me! Stay eco-friendly and protect our planet.")
            st.stop()

if __name__ == '__main__':
    main()
