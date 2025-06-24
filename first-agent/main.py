import os
from agents import Agent , Runner , OpenAIChatCompletionsModel , AsyncOpenAI , RunConfig 
from openai.types.responses import ResponseTextDeltaEvent
from dotenv import load_dotenv
import chainlit as cl

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client

) 

config = RunConfig(
    model=model,
    model_provider =external_client,
    tracing_disabled =True

)

frontend_agent = Agent(
    name="Frontend Expert ",
    instructions="""
    You are a Frontend Expert. 
    You help with HTML, CSS, JavaScript, React, Next.js, Tailwind CSS, and UI/UX design.
    Always provide clear, helpful, and example-based answers.
    """
)


backend_agent = Agent(
    name="Backend Expert ",
    instructions="""
    You are a Backend Expert. 
    You help with backend development using Python (Flask, Django), Node.js (Express), REST APIs, database design (SQL, NoSQL), and performance optimization.
    Provide code samples and best practices.
    """
)

web_dev_agent = Agent(
    name="Full Stack Web Dev",
    instructions="""
    You are a Full Stack Web Developer.
    You understand both frontend and backend. You can assist with full project architecture, API integration, deployment (Netlify, Vercel, AWS), and debugging complex web apps.
    Guide the user step by step with clear explanations.
    """,
    handoffs=[frontend_agent,backend_agent]
)

@cl.on_chat_start
async def handle_start_chat():
    cl.user_session.set("history", [])
    await cl.Message(content="👋 Salaam! I'm Rayyan's Full Stack Coding Assistant.\n\nAsk me anything about programming, coding problems, or software development! 🚀").send()

@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")

    msg = cl.Message(content="")
    await msg.send()

    history.append({"role": "user", "content": message.content})

    result = Runner.run_streamed(
        web_dev_agent,
        input=history,
        run_config=config
    )

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)

    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)
