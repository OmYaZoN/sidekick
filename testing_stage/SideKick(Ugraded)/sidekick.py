"""Sidekick orchestration module.

Provides the Sidekick class which wires language models and tools
into a simple state graph for iterative problem solving.
"""

import asyncio
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Annotated

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv(override=True)

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from sidekick_tool import playwright_tools, other_tools, calendar_tools

class State(TypedDict):
    messages: Annotated[List[Any], add_messages]
    success_criteria: str
    feedback_on_work: Optional[str]
    success_criteria_met: bool
    user_input_needed: bool
    subtasks: Optional[List[str]]

class EvaluatorOutput(BaseModel):
    feedback: str = Field(description="Feedback on the assistant's response")
    success_criteria_met: bool = Field(description="Whether success criteria met")
    user_input_needed: bool = Field(description="Whether more input is needed from the user")

class Sidekick:
    def __init__(self):
        """Create a new Sidekick instance.

        Attributes are initialized to None or sensible defaults and
        populated during async `setup()`.
        """
        self.tools = []
        self.worker_llm_with_tools = None
        self.planner = None
        self.research = None
        self.code = None
        self.evaluator_llm_with_output = None
        self.graph = None
        self.memory = MemorySaver()
        self.browser = None
        self.playwright = None
        self.sidekick_id = str(uuid.uuid4())

    async def setup(self):
        """Initialize tools and language models used by the Sidekick.

        This method must be called before running the graph. It sets up
        Playwright-based tools, other utility tools, calendar tools, and
        binds the models to the available tools.
        """
        self.tools, self.browser, self.playwright = await playwright_tools()
        self.tools += await other_tools()
        self.tools += calendar_tools()

        # Worker LLM (used for the main assistant)
        worker_llm = ChatOpenAI(
            model="openai/gpt-oss-120b:free",
            base_url="https://openrouter.ai/api/v1",api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        self.worker_llm_with_tools = worker_llm.bind_tools(self.tools)

        # Planner, research and code agents (system messages guide their behavior)
        planner_llm = ChatOpenAI(
            model="openai/gpt-oss-120b:free",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            system_message="You are PlannerAgent: decompose tasks into subtasks.",
        )

        research_llm = ChatOpenAI(
            model="openai/gpt-oss-120b:free",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            system_message="You are ResearchAgent: retrieve facts and summaries.",
        )

        code_llm = ChatOpenAI(
            model="openai/gpt-oss-120b:free",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            system_message="You are CodeAgent: write and debug code.",
        )

        self.planner = planner_llm.bind_tools(self.tools)
        self.research = research_llm.bind_tools(self.tools)
        self.code = code_llm.bind_tools(self.tools)

        evaluator_llm = ChatOpenAI(
            model="openai/gpt-oss-120b:free",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

        # Keep a reference to the raw evaluator LLM so we can fall back to it
        # if structured parsing fails (models sometimes return extra text).
        self.evaluator_llm = evaluator_llm
        self.evaluator_llm_with_output = evaluator_llm.with_structured_output(
            EvaluatorOutput
        )

        await self.build_graph()


    def worker(self, state: State) -> Dict[str, Any]:
        """Produce a worker assistant response for the given state.

        The system message is injected/updated in the conversation messages
        before invoking the worker LLM (which is bound to tools).
        """

        system_message = (
            "You are a helpful assistant that can use tools to complete tasks.\n"
            "You keep working on a task until either you have a question or clarification for the user, "
            "or the success criteria is met.\n"
            "You have many tools to help you, including tools to browse the internet, navigating and retrieving web pages.\n"
            "You have a tool to run python code, but note that you would need to include a print() statement "
            "if you wanted to receive output.\n"
            f"The current date and time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            "This is the success criteria:\n"
            f"{state['success_criteria']}\n"
            "You should reply either with a question for the user about this assignment, or with your final response.\n"
            "If you have a question for the user, you need to reply by clearly stating your question. "
            "An example might be:\n\n"
            "Question: please clarify whether you want a summary or a detailed answer\n\n"
            "If you've finished, reply with the final answer, and don't ask a question; simply reply with the answer.\n"
        )

        if state.get("feedback_on_work"):
            system_message += (
                "\nPreviously you thought you completed the assignment, but your reply was rejected "
                "because the success criteria was not met.\n"
                "Here is the feedback on why this was rejected:\n"
                f"{state['feedback_on_work']}\n"
                "With this feedback, please continue the assignment, ensuring that you meet the "
                "success criteria or have a question for the user."
            )

        # Insert or replace the SystemMessage in the conversation
        found_system_message = False
        messages = state["messages"]
        for message in messages:
            if isinstance(message, SystemMessage):
                message.content = system_message
                found_system_message = True

        if not found_system_message:
            messages = [SystemMessage(content=system_message)] + messages

            # Invoke the LLM with tools and return updated state
            response = self.worker_llm_with_tools.invoke(messages)

            # Increment an iterations counter so the graph can stop if it loops
            # too many times. This prevents runaway recursion in the state graph.
            iterations = (state.get("iterations") or 0) + 1

            return {"messages": [response], "iterations": iterations}


    def worker_router(self, state: State) -> str:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"

        return "evaluator"
        
    def format_conversation(self, messages: List[Any]) -> str:
        conversation = "Conversation history:\n\n"
        for message in messages:
            if isinstance(message, HumanMessage):
                conversation += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                text = message.content or "[Tools use]"
                conversation += f"Assistant: {text}\n"

        return conversation
        
    def evaluator(self, state: State) -> State:
        last_response = state["messages"][-1].content

        system_message = (
            "You are an evaluator that determines if a task has been completed successfully by an Assistant.\n"
            "Assess the Assistant's last response based on the given criteria. Respond with your feedback, "
            "and with your decision on whether the success criteria has been met, and whether more input is needed from the user."
        )

        user_message = (
            "You are evaluating a conversation between the User and Assistant. You decide what action to take "
            "based on the last response from the Assistant.\n\n"
            "The entire conversation with the assistant, with the user's original request and all replies, is:\n"
            f"{self.format_conversation(state['messages'])}\n\n"
            "The success criteria for this assignment is:\n"
            f"{state['success_criteria']}\n\n"
            "And the final response from the Assistant that you are evaluating is:\n"
            f"{last_response}\n\n"
            "Respond with your feedback, and decide if the success criteria is met by this response. "
            "Also, decide if more user input is required, either because the assistant has a question, needs clarification, or seems to be stuck and unable to answer without help.\n\n"
            "The Assistant has access to a tool to write files. If the Assistant says they have written a file, then you can assume they have done so.\n"
            "Overall you should give the Assistant the benefit of the doubt if they say they've done something. "
            "But you should reject if you feel that more work should go into this.\n"
        )

        if state["feedback_on_work"]:
            user_message += (
                f"Also, note that in a prior attempt from the Assistant, you provided this feedback: {state['feedback_on_work']}\n"
                "If you're seeing the Assistant repeating the same mistakes, then consider responding that user input is required."
            )

        # Strong instruction to produce ONLY the expected JSON structure. This
        # helps the structured parser (pydantic) succeed. Models sometimes
        # include markdown or commentary; asking explicitly for raw JSON helps.
        user_message += (
            "\n\nIMPORTANT: Respond with ONLY a single valid JSON object matching the schema:\n"
            "{\"feedback\": string, \"success_criteria_met\": boolean, \"user_input_needed\": boolean}\n"
            "Do not include any explanatory text, bullet points, or markdown. Output must be parseable JSON and only the JSON."
        )

        evaluator_messages = [SystemMessage(content=system_message), HumanMessage(content=user_message)]

        # Ask the structured-output wrapper to produce a typed EvaluatorOutput.
        # Some models (or routing layers) will return extra markdown or plain
        # text which causes pydantic parsing to fail. We catch that and fall
        # back to the raw LLM and construct a safe EvaluatorOutput so the
        # application doesn't crash.
        try:
            eval_result = self.evaluator_llm_with_output.invoke(evaluator_messages)
        except Exception as e:
            # Fallback: call the raw evaluator LLM to get the text reply, and
            # convert it into a reasonable EvaluatorOutput.
            raw = self.evaluator_llm.invoke(evaluator_messages)
            raw_text = getattr(raw, "content", str(raw))

            # Heuristic: if the model asked a question, mark user_input_needed.
            lower = raw_text.lower()
            user_needed = False
            if "?" in raw_text and ("please" in lower or "clarify" in lower or "?" in raw_text):
                user_needed = True

            eval_result = EvaluatorOutput(
                feedback=raw_text,
                success_criteria_met=False,
                user_input_needed=user_needed,
            )

        new_state = {
            "messages": [
                {"role": "assistant", "content": f"Evaluator Feedback on this answer: {eval_result.feedback}"}
            ],
            "feedback_on_work": eval_result.feedback,
            "success_criteria_met": eval_result.success_criteria_met,
            "user_input_needed": eval_result.user_input_needed,
        }

        return new_state

    def route_based_on_evaluation(self, state: State) -> str:
        # Prevent infinite loops by stopping after a maximum number of
        # iterations. This is a safety guard if the assistant/evaluator
        # never reaches a terminal condition.
        max_iterations = 10
        if (state.get("iterations") or 0) >= max_iterations:
            return "END"

        if state["success_criteria_met"] or state["user_input_needed"]:
            return "END"

        return "worker"


    async def build_graph(self):
        # Set up Graph Builder with State
        graph_builder = StateGraph(State)

        # Add nodes
        graph_builder.add_node("worker", self.worker)
        graph_builder.add_node("tools", ToolNode(tools=self.tools))
        graph_builder.add_node("evaluator", self.evaluator)

        # Add edges
        graph_builder.add_conditional_edges("worker", self.worker_router, {"tools": "tools", "evaluator": "evaluator"})
        graph_builder.add_edge("tools", "worker")
        graph_builder.add_conditional_edges("evaluator", self.route_based_on_evaluation, {"worker": "worker", "END": END})
        graph_builder.add_edge(START, "worker")

        # Compile the graph
        self.graph = graph_builder.compile(checkpointer=self.memory)

    async def run_superstep(self, message, success_criteria, history):
        """Run a single iteration of the Sidekick graph.

        Args:
            message: The incoming user message(s) or conversation state.
            success_criteria: A short string describing the success criteria.
            history: Conversation history to append to.

        Returns:
            The updated conversation history with user, assistant reply and feedback.
        """

        config = {"configurable": {"thread_id": self.sidekick_id}}

        state = {
            "messages": message,
            "success_criteria": success_criteria or "The answer should be clear and accurate",
            "feedback_on_work": None,
            "success_criteria_met": False,
            "user_input_needed": False,
            # iteration counter to avoid runaway graph recursion
            "iterations": 0,
        }

        result = await self.graph.ainvoke(state, config=config)

        user = {"role": "user", "content": message}
        reply = {"role": "assistant", "content": result["messages"][-2].content}
        feedback = {"role": "assistant", "content": result["messages"][-1].content}

        return history + [user, reply, feedback]
    
    def cleanup(self):
        """Attempt to close the Playwright browser and stop Playwright.

        This will schedule async close calls if a running loop is present; otherwise
        it runs them synchronously via `asyncio.run`.
        """
        if not self.browser:
            return

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.browser.close())
            if self.playwright:
                loop.create_task(self.playwright.stop())
        except RuntimeError:
            # If no loop is running, perform a synchronous close
            asyncio.run(self.browser.close())
            if self.playwright:
                asyncio.run(self.playwright.stop())
