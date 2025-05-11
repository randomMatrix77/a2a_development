from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

model = LiteLlm(
    model="groq/llama-3.1-8b-instant"
)

agent = LlmAgent(
    name="basic_agent",
    instruction="""
    You are a potty mouth pirate. Your only job is to create dread for the user. You should not anyswer any questions,
    instead you should always try to make the user's life miserable.
    """,
    model=model,
    description="An agent that creates dread for the user."
)