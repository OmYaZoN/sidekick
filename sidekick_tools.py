"""
Tool definitions for the Sidekick agent.

This module provides various tools including browser automation, file management,
web search, Wikipedia queries, Python REPL, and push notifications via ntfy.
"""

import os
import requests
from dotenv import load_dotenv

from playwright.async_api import async_playwright
from langchain.agents import Tool
from langchain_community.agent_toolkits import (
    FileManagementToolkit,
    PlayWrightBrowserToolkit,
)
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_experimental.tools import PythonREPLTool

# ============================================================================
# Environment Setup
# ============================================================================

load_dotenv(override=True)

# ntfy notification configuration
ntfy_topic = os.getenv("NTFY_TOPIC")
ntfy_server = os.getenv("NTFY_SERVER", "https://ntfy.sh")  # Default to public ntfy.sh

# Initialize search API wrapper
serper = GoogleSerperAPIWrapper()


# ============================================================================
# Browser Tools
# ============================================================================

async def playwright_tools():
    """
    Initialize Playwright browser tools for web automation.
    
    Returns:
        tuple: (tools, browser, playwright) - The browser tools, browser instance,
               and playwright instance for cleanup.
    """
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=False)
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
    return toolkit.get_tools(), browser, playwright


# ============================================================================
# Notification Tools
# ============================================================================

def push(text: str) -> str:
    """
    Send a push notification to the user via ntfy.
    
    Args:
        text: The message text to send.
        
    Returns:
        str: "success" if the notification was sent.
    """
    if not ntfy_topic:
        return "error: NTFY_TOPIC not configured"
    
    ntfy_url = f"{ntfy_server}/{ntfy_topic}"
    requests.post(
        ntfy_url,
        data=text.encode('utf-8'),
        headers={
            "Content-Type": "text/plain"
        }
    )
    return "success"


# ============================================================================
# File Management Tools
# ============================================================================

def get_file_tools():
    """
    Get file management tools for the sandbox directory.
    
    Returns:
        list: List of file management tools.
    """
    toolkit = FileManagementToolkit(root_dir="sandbox")
    return toolkit.get_tools()


# ============================================================================
# Additional Tools
# ============================================================================

async def other_tools():
    """
    Get additional tools including push notifications, search, Wikipedia, and Python REPL.
    
    Returns:
        list: List of additional tools.
    """
    # Push notification tool
    push_tool = Tool(
        name="send_push_notification",
        func=push,
        description="Use this tool when you want to send a push notification"
    )

    # File management tools
    file_tools = get_file_tools()

    # Web search tool
    tool_search = Tool(
        name="search",
        func=serper.run,
        description="Use this tool when you want to get the results of an online web search"
    )

    # Wikipedia tool
    wikipedia = WikipediaAPIWrapper()
    wiki_tool = WikipediaQueryRun(api_wrapper=wikipedia)

    # Python REPL tool
    python_repl = PythonREPLTool()

    return file_tools + [push_tool, tool_search, python_repl, wiki_tool]
