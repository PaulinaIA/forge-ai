"""Forge Agent — LangGraph-powered ML assistant."""

from __future__ import annotations

import os
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from forge_ai.agent.llm import PROVIDERS, LLMConfig
from forge_ai.prompts.system import FORGE_SYSTEM_PROMPT
from forge_ai.tools import get_all_tools


class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[list[BaseMessage], add_messages]


class ForgeAgent:
    """Main agent orchestrator for the ML assistant using LangGraph."""

    def __init__(
        self,
        provider: str = "groq",
        model: str | None = None,
        temperature: float = 0.1,
        memory_window: int = 10,
        verbose: bool = False,
    ) -> None:
        self.config = LLMConfig(provider=provider, model=model, temperature=temperature)
        self._verbose = verbose
        self._memory_window = max(int(memory_window), 0)
        self._chat_history: list[BaseMessage] = []

        provider_cfg = PROVIDERS[self.config.provider]
        env_key = str(provider_cfg["env_key"])
        api_key = os.getenv(env_key)
        if not api_key:
            raise ValueError(f"Missing API key. Set `{env_key}` in your environment or .env file.")

        self._llm = ChatOpenAI(
            api_key=api_key,
            base_url=str(provider_cfg["base_url"]),
            model=str(self.config.model),
            temperature=temperature,
            max_tokens=self.config.max_tokens,
        )

        self._tools = get_all_tools()
        self._model_with_tools = self._llm.bind_tools(self._tools)
        self._graph = self._build_graph()

    def _call_model(self, state: AgentState) -> dict:
        """Node for calling the LLM."""
        messages = list(state["messages"])
        
        # Workaround for Gemini's OpenAI compatibility endpoint:
        # It throws an error if SystemMessages are mixed with ToolMessages.
        # We inject the system prompt into the first HumanMessage instead.
        if messages and isinstance(messages[0], HumanMessage):
            # Ensure we only prepend once
            if not str(messages[0].content).startswith("You are Forge"):
                messages[0] = HumanMessage(
                    content=f"{FORGE_SYSTEM_PROMPT}\n\n{messages[0].content}"
                )
        
        response = self._model_with_tools.invoke(messages)
        # LangGraph add_messages handles appending to state
        return {"messages": [response]}

    def _should_continue(self, state: AgentState) -> str:
        """Determine whether to continue or end the graph."""
        messages = state["messages"]
        last_message = messages[-1]
        
        # If there are tool calls, route to the tools node
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        
        # Otherwise, end
        return END

    def _build_graph(self):
        """Build the state graph."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", ToolNode(self._tools))

        # Add edges
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {"tools": "tools", END: END}
        )
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    def run(self, query: str, **kwargs: Any) -> dict:
        """Run the agent on a user query."""
        if "file_path" in kwargs and kwargs["file_path"]:
            query = f"[Dataset loaded: {kwargs['file_path']}]\n\n{query}"

        # Combine history + new query
        input_messages = self._chat_history + [HumanMessage(content=query)]
        
        # Run graph
        result = self._graph.invoke({"messages": input_messages})
        
        # final state representation as sequence of messages
        final_messages = result["messages"]
        
        # Extract steps (tool calls) since our last prompt 
        # to match the old public API
        new_msgs_only = final_messages[len(input_messages):]
        
        output = ""
        steps = []
        
        for msg in new_msgs_only:
            if isinstance(msg, AIMessage):
                if msg.content:
                    output += "\n" + str(msg.content) if output else str(msg.content)
                    
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        steps.append({
                            "id": tc.get("id"),
                            "tool": tc.get("name"),
                            "input": str(tc.get("args", {}))[:200],
                            "output": ""
                        })
            
            elif isinstance(msg, ToolMessage):
                tc_id = msg.tool_call_id
                for s in steps:
                    if s.get("id") == tc_id:
                        s["output"] = str(msg.content)[:500]

        for s in steps:
            s.pop("id", None)

        self._chat_history = final_messages
        
        if self._memory_window > 0:
            max_msgs = self._memory_window * 2
            if len(self._chat_history) > max_msgs:
                self._chat_history = self._chat_history[-max_msgs:]

        return {
            "output": output.strip(),
            "steps": steps,
        }

    def reset_memory(self) -> None:
        """Clear conversation history."""
        self._chat_history = []

    @property
    def tools(self) -> list:
        return self._tools

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self._tools]

