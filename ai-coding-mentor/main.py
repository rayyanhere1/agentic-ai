import os
from agents import Agent , Runner , OpenAIChatCompletionsModel , AsyncOpenAI , RunConfig
from dotenv import load_dotenv
from openai.types.responses import ResponseTextDeltaEvent
import chainlit as cl

load_dotenv()

gemini_api_key=os.getenv("GEMINI_API_KEY")

external_client= AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model= OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config= RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

agent = Agent(
    name="AI Coding Mentor 👨‍🏫",
    instructions="""
    You are an AI Coding Mentor.
    You help junior developers understand key concepts in frontend, backend, and full stack development.
    You explain in very simple terms, give examples, and provide best practices.
    Your answers are friendly, supportive, and educational.
    If asked, provide code examples and explain step by step.
    """
)

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history",[])
    await cl.Message(content="👋 Salaam! I am your AI Coding Mentor 👨‍🏫 — ask me anything about coding! 🚀").send()


@cl.on_message
async def handle_message(message:cl.Message):
    history = cl.user_session.get("history")
    msg = cl.Message(content="")
    await msg.send()

    history.append({"role":"user","content": message.content})

    result = Runner.run_streamed(
        agent,
        input=history,
        run_config=config
    )

    
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)

    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)
