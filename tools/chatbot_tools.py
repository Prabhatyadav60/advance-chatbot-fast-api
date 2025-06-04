import os
import json
import re
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
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


load_dotenv()
TAVILY_API_KEY      = os.getenv("TAVILY_API_KEY")
GOOGLE_API_KEY      = os.getenv("GOOGLE_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GMAIL_ADDRESS       = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD  = os.getenv("GMAIL_APP_PASSWORD")


LOADED_TRANSCRIPTS = {} 


YOUTUBE_ID_REGEX = re.compile(
    r"""
    (?:https?://)?                # optional protocol
    (?:www\.)?                    # optional www
    (?:                           # beginning of host alternatives
      youtube\.com/watch\?v=      #   youtube.com/watch?v=VIDEO_ID
      |youtu\.be/                 #   youtu.be/VIDEO_ID
      |youtube\.com/embed/        #   youtube.com/embed/VIDEO_ID
      |youtube\.com/v/            #   youtube.com/v/VIDEO_ID
    )
    ([0-9A-Za-z_-]{11})           # capture group for the 11-character video ID
    """,
    re.VERBOSE,
)

def extract_video_id(url: str) -> str:
    """
    Extracts a YouTube video ID from various URL formats.
    Returns the 11-character ID if found, otherwise None.
    """
    match = YOUTUBE_ID_REGEX.search(url)
    return match.group(1) if match else None

def format_timestamp(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def get_weather_raw_json(city: str) -> dict:
    api_key = OPENWEATHER_API_KEY
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        return {"error": response.text}
    return response.json()

def build_weather_prompt(weather_json: dict) -> str:
    json_str = json.dumps(weather_json, indent=2)
    return f"""
You are a helpful assistant.

Here is the raw JSON data from a weather API:

{json_str}

Please summarize the weather in a single, human-readable sentence. Include:
- Weather description
- Temperature in Celsius
- Humidity percentage
- Wind speed in m/s
- City name

Example: "The weather in London is clear sky with a temperature of 18.5°C, humidity at 56%, and wind speed of 3.6 m/s."
"""




@tool(description="Fetch current weather for a given city.")
def get_weather(city: str) -> str:
    api_key = OPENWEATHER_API_KEY
    if not api_key:
        return "Error: OPENWEATHER_API_KEY not set."
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        return f"Failed to fetch weather: {response.text}"
    data = response.json()
    weather = data["weather"][0]["description"]
    temp = data["main"]["temp"]
    return f"The weather in {city} is {weather} with a temperature of {temp}°C."

@tool(description="Summarize raw weather JSON into a human-readable sentence.")
def summarize_weather(city: str) -> str:
    weather_json = get_weather_raw_json(city)
    prompt = build_weather_prompt(weather_json)
    llm_response = llm.invoke([{"role": "user", "content": prompt}])
    return llm_response.content

@tool(description="Send an email: parameters are recipient, subject, and body.")
def send_email(recipient: str, subject: str, body: str) -> str:
    sender = GMAIL_ADDRESS
    app_pass = GMAIL_APP_PASSWORD
    if not sender or not app_pass:
        return "Error: GMAIL_ADDRESS or GMAIL_APP_PASSWORD not set."
    msg = MIMEText(body, "plain")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender, app_pass)
            smtp.send_message(msg)
        return f"Email sent successfully to {recipient}."
    except Exception as e:
        return f"Error sending email: {e}"


@tool(description="Load a YouTube video’s transcript and return the full timestamped transcript.")
def get_youtube_transcript(video_url: str) -> str:
    video_id = extract_video_id(video_url)
    if not video_id:
        return "Error: Invalid YouTube URL."

    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript_obj = transcript_list.find_generated_transcript(["en"])
        except (NoTranscriptFound, TranscriptsDisabled):
            transcript_obj = transcript_list.find_transcript(["en"])
        transcript_data = transcript_obj.fetch()
    except TranscriptsDisabled:
        return "Error: Transcripts are disabled for this video."
    except NoTranscriptFound:
        return "Error: No transcripts found for this video."
    except Exception as e:
        return f"Error retrieving transcript: {e}"

    LOADED_TRANSCRIPTS[video_id] = transcript_data

    lines = []
    for entry in transcript_data:
        ts = format_timestamp(entry.start)
        text = entry.text.replace("\n", " ").strip()
        lines.append(f"[{ts}] {text}")
    return "\n".join(lines)

tool_search = TavilySearch(tavily_api_key=TAVILY_API_KEY, max_results=4)

tools = [
    tool_search,
    get_weather,
    summarize_weather,
    get_youtube_transcript,
    send_email,
]

memory = MemorySaver()
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


llm = init_chat_model("google_genai:gemini-2.0-flash")
llm_with_tools = llm.bind_tools(tools)

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def chatbot_node(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot_node)

class BasicToolNode:
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        messages = inputs.get("messages", [])
        if not messages:
            raise ValueError("No messages in input")
        last = messages[-1]
        outputs = []
        for tool_call in last.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

tool_node = BasicToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

def route_tools(state: State):
    messages = state.get("messages", [])
    if not messages:
        raise ValueError("No messages in input state")
    last = messages[-1]
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


def run_chat_through_graph(user_message: str, thread_id: str = None) -> dict:
    """
    Run a single user message through LangGraph, optionally using an existing thread_id.
    Prepends a system message so that the LLM knows to call our tools
    when it sees an image URL, YouTube URL, etc.
    Returns a dict with:
      - 'response': the final AI response
      - 'thread_id': the UUID used (either provided or newly generated)
    """
  
    system_msg = {
        "role": "system",
        "content": (
            "You are a helpful assistant that has access to the following tools:\n"
      
            "- get_youtube_transcript(video_url: str) → str\n"
            "  Use this tool when I provide a YouTube URL, to fetch the transcript or provide a summary of the video.\n"
            "- get_weather(city: str) → str\n"
            "  Use this tool when I ask about current weather.\n"
            "- summarize_weather(city: str) → str\n"
            "  Use this tool when I ask for a more detailed weather summary.\n"
            "- send_email(recipient: str, subject: str, body: str) → str\n"
            "  Use this tool when I ask you to send an email.\n\n"
            "Always return the final answer after any tool calls. "
            "If no tool is needed, just answer directly."
        )
    }

 
    prompt = f"{user_message}"
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    messages = [
        system_msg,
        {"role": "user", "content": prompt}
    ]

    output = graph.invoke(
        {"messages": messages},
        config=config,
    )

   
    ai_text = None
    for msg in reversed(output["messages"]):
        if isinstance(msg, AIMessage):
            ai_text = msg.content
            break

    if ai_text is None:
      
        ai_text = json.dumps([m.content for m in output["messages"]])

    return {"response": ai_text, "thread_id": thread_id}
