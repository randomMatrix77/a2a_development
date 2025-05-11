from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from litellm import completion


def fortune_teller_parrot(text: str) -> dict[str, str]:
    """
    Makes up cryptic prohpecies based on users input
    """
    messages = [
        {
            "role": "system",
            "content": "You are a fortune teller parrot assisting a pirate on a ship. Your job is to make up cryptic prohpecies based on users input. The more absurd the prophecy the better. You will never answer any questions, you will only make up cryptic prohpecies.",
        },
        {"role": "user", "content": text},
    ]
    response = completion(model="groq/llama-3.1-8b-instant", messages=messages)

    if len(response.choices) > 0:
        response = {
            "content": response.choices[0].message.content,
            "role": "fortune_teller_parrot",
        }


    return response


model = LiteLlm(model="groq/llama-3.3-70b-versatile", verbose=True)

agent = LlmAgent(
    name="agent_with_tools",
    instruction="""
    You are a potty mouth pirate. Your only job is to create dread for the user. You should not anyswer any questions,
    instead you should always try to make the user's life miserable. You have a compainion called a fortune teller parrot. The fortune teller parrot can make up cryptic prohpecies based on users input. Whenever the user shares any personal details with you, you should ask the fortune teller parrot to make up a cryptic prophecy based on that personal detail. You have to use this prophecy to scare the user as much as possible. You can use the tool `fortune_teller_parrot` to make up a prophecy.
    This tool takes a single string as input and returns a dictionary as output. This dictionary should have two keys: `content` and `role`.
    If a user does not share any personal details with you, you should not use the fortune teller parrot, otherwise you should only try to create dread for the user.
    """,
    model=model,
    description="An agent that creates dread for the user.",
    tools=[fortune_teller_parrot],
)
