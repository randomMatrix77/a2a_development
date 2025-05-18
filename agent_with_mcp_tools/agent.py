from contextlib import AsyncExitStack
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

model = LiteLlm(model="groq/llama-3.3-70b-versatile", verbose=True)


async def create_agent():
    common_exit_stack = AsyncExitStack()

    local_tools, _ = await MCPToolset.from_server(
        connection_params=StdioServerParameters(
            command="npx", args=["-y", "iterm-mcp"]
        ),
        async_exit_stack=common_exit_stack,
    )

    agent = LlmAgent(
        model=model,
        name="agent_with_mcp_tools",
        instruction="""You are a helpful agent with access to the terminal. You can use the `terminal_tool` to access the terminal. If the user asks you to use the terminal, you should use the `terminal_tool` to access the terminal.""",
        tools=[*local_tools],
    )
    return agent, common_exit_stack


root_agent = create_agent()
