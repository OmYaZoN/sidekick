"""
Sidekick agent implementation using LangGraph.

This module defines the Sidekick class, which implements a multi-agent system
with a worker agent and an evaluator agent that work together to complete tasks.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from sidekick_tools import other_tools, playwright_tools

load_dotenv(override=True)


# ============================================================================
# State and Data Models
# ============================================================================

class State(TypedDict):
    """State structure for the LangGraph workflow."""
    messages: Annotated[List[Any], add_messages]
    success_criteria: str
    feedback_on_work: Optional[str]
    success_criteria_met: bool
    user_input_needed: bool
    step_count: int


class EvaluatorOutput(BaseModel):
    """Structured output from the evaluator agent."""
    feedback: str = Field(
        description="Feedback on the assistant's response"
    )
    success_criteria_met: bool = Field(
        description="Whether the success criteria have been met"
    )
    user_input_needed: bool = Field(
        description="True if more input is needed from the user, or clarifications, or the assistant is stuck"
    )


# ============================================================================
# Sidekick Class
# ============================================================================

class Sidekick:
    """Main Sidekick agent class that orchestrates worker and evaluator agents."""

    def __init__(self):
        """Initialize a new Sidekick instance."""
        self.worker_llm_with_tools = None
        self.evaluator_llm_with_output = None
        self.tools = None
        self.llm_with_tools = None
        self.graph = None
        self.sidekick_id = str(uuid.uuid4())
        self.memory = MemorySaver()
        self.browser = None
        self.playwright = None

    async def setup(self):
        """Set up the Sidekick agent with tools and LLMs."""
        # Initialize browser tools
        self.tools, self.browser, self.playwright = await playwright_tools()
        self.tools += await other_tools()

        # Initialize worker LLM with tools
        worker_llm = ChatOpenAI(
            model="openai/gpt-oss-120b:free",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        self.worker_llm_with_tools = worker_llm.bind_tools(self.tools)

        # Initialize evaluator LLM with structured output
        evaluator_llm = ChatOpenAI(
            model="openai/gpt-oss-120b:free",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        self.evaluator_llm_with_output = evaluator_llm.with_structured_output(
            EvaluatorOutput
        )

        # Build the LangGraph workflow
        await self.build_graph()

    def worker(self, state: State) -> Dict[str, Any]:
        """
        Worker node that processes tasks using available tools.
        
        Args:
            state: Current state of the workflow.
            
        Returns:
            Dict containing updated messages.
        """
        system_message = f"""You are a helpful assistant that can use tools to complete tasks.
        You keep working on a task until either you have a question or clarification for the user, or the success criteria is met.
        You have many tools to help you, including tools to browse the internet, navigating and retrieving web pages.
        You have a tool to run python code, but note that you would need to include a print() statement if you wanted to receive output.
        The current date and time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

        This is the success criteria:
        {state["success_criteria"]}
        You should reply either with a question for the user about this assignment, or with your final response.
        If you have a question for the user, you need to reply by clearly stating your question. An example might be:

        Question: please clarify whether you want a summary or a detailed answer

        If you've finished, reply with the final answer, and don't ask a question; simply reply with the answer.
        """

        if state.get("feedback_on_work"):
            system_message += f"""
        Previously you thought you completed the assignment, but your reply was rejected because the success criteria was not met.
        Here is the feedback on why this was rejected:
        {state["feedback_on_work"]}
        With this feedback, please continue the assignment, ensuring that you meet the success criteria or have a question for the user.
        """

        # Add or update the system message
        found_system_message = False
        messages = state["messages"]
        for message in messages:
            if isinstance(message, SystemMessage):
                message.content = system_message
                found_system_message = True

        if not found_system_message:
            messages = [SystemMessage(content=system_message)] + messages

        # Invoke the LLM with tools
        response = self.worker_llm_with_tools.invoke(messages)

        # Return updated state
        return {
            "messages": [response],
        }

    def worker_router(self, state: State) -> str:
        """
        Route from worker node based on whether tools were called.
        
        Args:
            state: Current state of the workflow.
            
        Returns:
            str: Next node name ("tools" or "evaluator").
        """
        last_message = state["messages"][-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        else:
            return "evaluator"

    def format_conversation(self, messages: List[Any]) -> str:
        """
        Format conversation messages into a readable string.
        
        Args:
            messages: List of message objects.
            
        Returns:
            str: Formatted conversation string.
        """
        conversation = "Conversation history:\n\n"
        for message in messages:
            if isinstance(message, HumanMessage):
                conversation += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                text = message.content or "[Tools use]"
                conversation += f"Assistant: {text}\n"
        return conversation

    def evaluator(self, state: State) -> State:
        """
        Evaluator node that assesses whether the task has been completed successfully.
        You are an evaluator.

        Args:
            state: Current state of the workflow.
            
        Returns:
            State: Updated state with evaluation results.
        """
        last_response = state["messages"][-1].content

        


        system_message = """You are an evaluator that determines if a task has been completed successfully by an Assistant.
        Assess the Assistant's last response based on the given criteria. Respond with your feedback, and with your decision on whether the success criteria has been met,
        and whether more input is needed from the user.

        You MUST respond ONLY with valid JSON.
            NO markdown.
            NO explanations.
            NO extra text.

            Return EXACTLY this JSON schema:

            {
                "feedback": "string",
                "success_criteria_met": true or false,
                "user_input_needed": true or false
            }
        """

        user_message = f"""You are evaluating a conversation between the User and Assistant. You decide what action to take based on the last response from the Assistant.

        The entire conversation with the assistant, with the user's original request and all replies, is:
        {self.format_conversation(state["messages"])}

        The success criteria for this assignment is:
        {state["success_criteria"]}

        And the final response from the Assistant that you are evaluating is:
        {last_response}

        Respond with your feedback, and decide if the success criteria is met by this response.
        Also, decide if more user input is required, either because the assistant has a question, needs clarification, or seems to be stuck and unable to answer without help.

        The Assistant has access to a tool to write files. If the Assistant says they have written a file, then you can assume they have done so.
        Overall you should give the Assistant the benefit of the doubt if they say they've done something. But you should reject if you feel that more work should go into this.
        """

        if state["feedback_on_work"]:
            user_message += f"Also, note that in a prior attempt from the Assistant, you provided this feedback: {state['feedback_on_work']}\n"
            user_message += "If you're seeing the Assistant repeating the same mistakes, then consider responding that user input is required."

        evaluator_messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message),
        ]

        try:
            eval_result = self.evaluator_llm_with_output.invoke(evaluator_messages)
        except Exception as e:
            # Defensive fallback to prevent graph crash
            eval_result = EvaluatorOutput(
                feedback="Evaluator failed to produce valid JSON output.",
                success_criteria_met=False,
                user_input_needed=True,
            )

        step_count = state.get("step_count", 0) + 1

        new_state = {
            "messages": [
                AIMessage(content=f"Evaluator Feedback: {eval_result.feedback}")
            ],
            "feedback_on_work": eval_result.feedback,
            "success_criteria_met": eval_result.success_criteria_met,
            "user_input_needed": eval_result.user_input_needed,
            "step_count": step_count, 
        }
        return new_state

    def route_based_on_evaluation(self, state: State) -> str:
        """
        Route from evaluator node based on evaluation results.
        
        Args:
            state: Current state of the workflow.
            
        Returns:
            str: Next node name ("worker" or "END").
        """
        if state["step_count"] >= 5:
            return "END"

        if state["success_criteria_met"] or state["user_input_needed"]:
            return "END"
        else:
            return "worker"

    async def build_graph(self):
        """Build and compile the LangGraph workflow."""
        # Set up Graph Builder with State
        graph_builder = StateGraph(State)

        # Add nodes
        graph_builder.add_node("worker", self.worker)
        graph_builder.add_node("tools", ToolNode(tools=self.tools))
        graph_builder.add_node("evaluator", self.evaluator)

        # Add edges
        graph_builder.add_conditional_edges(
            "worker",
            self.worker_router,
            {"tools": "tools", "evaluator": "evaluator"}
        )
        graph_builder.add_edge("tools", "worker")
        graph_builder.add_conditional_edges(
            "evaluator",
            self.route_based_on_evaluation,
            {"worker": "worker", "END": END}
        )
        graph_builder.add_edge(START, "worker")

        # Compile the graph
        self.graph = graph_builder.compile(checkpointer=self.memory)

    async def run_superstep(self, message, success_criteria, history):
        """
        Run a single superstep of the workflow.
        
        Args:
            message: User's message/request.
            success_criteria: Criteria for task completion.
            history: Previous conversation history.
            
        Returns:
            list: Updated conversation history with user message, reply, and feedback.
        """
        config = {"configurable": {"thread_id": self.sidekick_id}}

        state = {
            "messages": [HumanMessage(content=message)],
            "success_criteria": success_criteria or "The answer should be clear and accurate",
            "feedback_on_work": None,
            "success_criteria_met": False,
            "user_input_needed": False,
            "step_count": 0,
        }
        result = await self.graph.ainvoke(state, config=config)
        
        user = {"role": "user", "content": message}
        assistant_msgs = [
            m for m in result["messages"] if isinstance(m, AIMessage)
        ]

        reply = {"role": "assistant", "content": assistant_msgs[-2].content}
        feedback = {"role": "assistant", "content": assistant_msgs[-1].content}
        
        return history + [user, reply, feedback]

    def cleanup(self):
        """Clean up browser and playwright resources."""
        if self.browser:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.browser.close())
                if self.playwright:
                    loop.create_task(self.playwright.stop())
            except RuntimeError:
                # If no loop is running, do a direct run
                asyncio.run(self.browser.close())
                if self.playwright:
                    asyncio.run(self.playwright.stop())
