import os
import chainlit as cl
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, function_tool, set_tracing_disabled
from openai.types.responses import ResponseTextDeltaEvent
import requests
import random
from dotenv import load_dotenv

load_dotenv()
set_tracing_disabled(disabled=True)

gemini_api_key = os.getenv("GEMINI_API_KEY")

provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider
)

@function_tool
def get_weather(city: str) -> str:
    try:
        result = requests.get(
            f"http://api.weatherapi.com/v1/current.json?key=8e3aca2b91dc4342a1162608252604&q={city}"
        )
        data = result.json()
        return f"The current weather in {city} is {data['current']['temp_c']}°C with {data['current']['condition']['text']}."
    except Exception as e:
        return f"Could not fetch weather: {e}"

@function_tool
def time_machine(year: int) -> str:
    events = {
        1969: "🚀 Man landed on the moon for the first time!",
        1971: "📧 The first email was sent.",
        1989: "🌐 Tim Berners-Lee invented the World Wide Web.",
        1994: "🕹️ The first PlayStation was released.",
        1997: "💻 Netflix was founded as a DVD rental company.",
        2000: "🎉 Y2K scare came and went, and nothing crashed.",
        2004: "📘 Facebook was launched.",
        2007: "📱 The very first iPhone was introduced by Apple.",
        2010: "📸 Instagram was launched.",
        2015: "🌍 The Paris Climate Agreement was signed.",
        2016: "💀 Pokémon Go took over the world with AR gaming.",
        2020: "🦠 COVID-19 changed the entire world.",
        2022: "🎬 ChatGPT became a global sensation.",
        2024: "🔭 NASA tested Artemis systems for future moon missions.",
    }
    if year in events:
        return f"In {year}: {events[year]}"
    elif year > 2025:
        return f"🔮 I can't predict the future (yet), but {year} sounds exciting!"
    else:
        return f"Hmm, I don't have info for {year}, but it was surely an important year!"

@function_tool
def food_recipe() -> str:
    recipes = [
        {
            "name": "Spaghetti Aglio e Olio",
            "ingredients": "Spaghetti, garlic, olive oil, red pepper flakes, parsley, salt",
            "description": "A simple yet flavorful Italian pasta tossed in garlic-infused olive oil with a hint of spice."
        },
        {
            "name": "Chicken Biryani",
            "ingredients": "Chicken, basmati rice, yogurt, onion, spices, saffron",
            "description": "A fragrant South Asian rice dish layered with marinated chicken and spices."
        },
        {
            "name": "Vegetable Stir Fry",
            "ingredients": "Bell peppers, carrots, broccoli, soy sauce, garlic, sesame oil",
            "description": "A quick and colorful mix of veggies sautéed in soy-garlic sauce — healthy and delicious."
        },
        {
            "name": "Pancakes",
            "ingredients": "Flour, eggs, milk, sugar, baking powder, butter",
            "description": "Fluffy and light breakfast pancakes — serve with maple syrup or fruit."
        },
        {
            "name": "Shawarma Wrap",
            "ingredients": "Chicken/beef, pita bread, garlic sauce, lettuce, tomato, pickles",
            "description": "A Middle Eastern favorite — grilled meat wrapped in pita with creamy garlic sauce and crunchy veggies."
        },
    ]
    recipe = random.choice(recipes)
    return f"🍴 Try This Recipe: {recipe['name']}\n\n📝 Ingredients: {recipe['ingredients']}\n📖 About: {recipe['description']}"

@function_tool
def game_suggestion() -> str:
    games = [
        "🎮 Try 'Minecraft'  Build, explore, and survive in a blocky world.",
        "🕹️ Try 'Among Us'  Find the imposter or be the imposter!",
        "🧩 Try 'Portal 2'  A genius puzzle game with humor and story.",
        "🛡️ Try 'Zelda: Breath of the Wild'  Open-world adventure at its finest.",
        "👾 Try 'Hollow Knight'  Stunning hand-drawn platformer."
    ]
    return random.choice(games)

agent = Agent(
    name="Assistant",
    instructions="Handle weather, history, recipes, and game suggestions using tools.",
    model=model,
    tools=[
        game_suggestion,
        get_weather,
        time_machine,
        food_recipe
    ]
)

@cl.on_chat_start
async def start_chat():
    await cl.Message(
        author="Assistant 🤖",
        content="""
👋 Welcome! I'm your AI Assistant.

You can ask me about:

- 🌤️ Weather → "What's the weather in Lahore?"
- 🕰️ History → "What happened in 2004?"
- 🍽️ Recipe → "Suggest me something to cook"
- 🎮 Game → "What game should I play?"
"""
    ).send()

@cl.on_message
async def handle_message(message: cl.Message):
    msg = cl.Message(content="")
    await msg.send()

    result = Runner.run_streamed(
        agent,
        input=message.content
    )

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)

    await msg.update()
