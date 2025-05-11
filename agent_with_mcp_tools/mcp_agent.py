from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import (
    MCPToolset,
    StdioServerParameters,
    SseServerParams,
)


async def terminal_tool(**args):
    tools, _ = await MCPToolset.from_server(
        connection_params=StdioServerParameters(command="npx", args=["-y", "iterm-mcp"])
    )

    return await tools[0].run_async(args=args, tool_context=None)


model = LiteLlm(model="groq/llama-3.3-70b-versatile", verbose=True)

agent = LlmAgent(
    name="agent_with_mcp_tools",
    instruction="""
    You are a helpful agent with access to the terminal. You can use the `terminal_tool` to access the terminal. If the user asks you to use the terminal, you should use the `terminal_tool` to access the terminal.
    """,
    model=model,
    description="An agent that can access the terminal",
    tools=[terminal_tool],
)
