import os
import json
import requests
import smtplib
import uuid
from email.mime.text import MIMEText
from PIL import Image
import pytesseract
from ultralytics import YOLO
import tempfile

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, Optional
from typing_extensions import TypedDict

# Load environment variables
load_dotenv()
TAVILY_API_KEY      = os.getenv("TAVILY_API_KEY")
GOOGLE_API_KEY      = os.getenv("GOOGLE_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GMAIL_ADDRESS       = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD  = os.getenv("GMAIL_APP_PASSWORD")

# Utility functions

def get_weather_raw_json(city: str) -> dict:
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    )
    resp = requests.get(url)
    if resp.status_code != 200:
        return {"error": resp.text}
    return resp.json()


def build_weather_prompt(weather_json: dict) -> str:
    json_str = json.dumps(weather_json, indent=2)
    return (
        "You are a helpful assistant.\n\n"
        "Here is the raw JSON data from a weather API:\n\n"
        f"{json_str}\n\n"
        "Please summarize the weather in a single, human-readable sentence. Include:\n"
        "- Weather description\n"
        "- Temperature in Celsius\n"
        "- Humidity percentage\n"
        "- Wind speed in m/s\n"
        "- City name\n\n"
        "Example: 'The weather in London is clear sky with a temperature of 18.5°C,"
        " humidity at 56%, and wind speed of 3.6 m/s.'"
    )

# Tool implementations

@tool(description="Fetch current weather for a given city.")
def get_weather(city: str) -> str:
    if not OPENWEATHER_API_KEY:
        return "Error: OPENWEATHER_API_KEY not set."
    data = get_weather_raw_json(city)
    if "error" in data:
        return f"Failed to fetch weather: {data['error']}"
    desc = data["weather"][0]["description"]
    temp = data["main"]["temp"]
    return f"The weather in {city} is {desc} with a temperature of {temp}°C."

@tool(description="Summarize raw weather JSON into a human-readable sentence.")
def summarize_weather(city: str) -> str:
    weather_json = get_weather_raw_json(city)
    prompt = build_weather_prompt(weather_json)
    res = llm.invoke([{"role": "user", "content": prompt}])
    return res.content

@tool(description="Send an email: parameters are recipient, subject, and body.")
def send_email(recipient: str, subject: str, body: str) -> str:
    if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
        return "Error: GMAIL_ADDRESS or GMAIL_APP_PASSWORD not set."
    msg = MIMEText(body, "plain")
    msg["Subject"] = subject
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = recipient
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            smtp.send_message(msg)
        return f"Email sent successfully to {recipient}."
    except Exception as e:
        return f"Error sending email: {e}"

@tool(description="Generate a good looking single-page website using HTML, CSS, and JavaScript based on the provided specification.")
def generate_website(spec: str) -> str:
    prompt = (
        "Create a single-page website using HTML, CSS, and JavaScript based on the following specification:\n\n"
        f"{spec}\n\n"
        "Provide the complete code in a single response."
    )
    res = llm.invoke([{"role": "user", "content": prompt}])
    return res.content

# Initialize tools and memory
tool_search = TavilySearch(tavily_api_key=TAVILY_API_KEY, max_results=4)
tools = [tool_search, get_weather, summarize_weather, send_email, generate_website]
memory = MemorySaver()
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize LLM and bind tools
llm = init_chat_model("google_genai:gemini-2.0-flash")
llm_with_tools = llm.bind_tools(tools)

# Build LangGraph state graph
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def chatbot_node(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"]) ]}

graph_builder.add_node("chatbot", chatbot_node)

class BasicToolNode:
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        msgs = inputs.get("messages", [])
        if not msgs:
            raise ValueError("No messages in input")
        last = msgs[-1]
        out = []
        for call in last.tool_calls:
            result = self.tools_by_name[call["name"]].invoke(call["args"])
            out.append(
                ToolMessage(
                    content=json.dumps(result),
                    name=call["name"],
                    tool_call_id=call["id"],
                )
            )
        return {"messages": out}

tool_node = BasicToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

def route_tools(state: State):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END

graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", END: END},
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

graph = graph_builder.compile(checkpointer=memory)

def run_chat_through_graph(user_message: str, thread_id: Optional[str] = None) -> dict:
    system_msg = {
        "role": "system",
        "content": (
            "You are a helpful assistant with access to tools: get_weather, summarize_weather, send_email, generate_website."
        )
    }
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    msgs = [system_msg, {"role": "user", "content": user_message}]
    out = graph.invoke({"messages": msgs}, config={"configurable": {"thread_id": thread_id}})
    ai_text = None
    for m in reversed(out["messages"]):
        if isinstance(m, AIMessage):
            ai_text = m.content
            break
    if ai_text is None:
        ai_text = json.dumps([m.content for m in out["messages"]])
    return {"response": ai_text, "thread_id": thread_id}
