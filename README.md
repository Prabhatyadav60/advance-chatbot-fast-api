# Advance Bot (Adbot)

Advance Bot (Adbot) is an advanced chatbot built with FastAPI and powered by Google’s Gemini language model via the LangGraph framework. It integrates multiple tools—such as weather lookup, YouTube transcript retrieval, email sending, OCR, and image recognition—into a single, stateful conversational agent. The frontend is a responsive, clean chat interface served via an `index.html` page, allowing real-time interaction with the AI.
## Frontend (FEATURES:  Mail sending support, Real time News/general query, Youtube video analysis, Weather report, Iimage analysis)
![image](https://github.com/user-attachments/assets/fa7295bd-1bbe-436c-bbba-2547e22d24c7)
## Image Analysis 
![image](https://github.com/user-attachments/assets/b717e266-eb6e-4aab-a977-04024949fa03)
## Youtube-interaction
![image](https://github.com/user-attachments/assets/855ddfbe-8742-4b4b-96e7-6ecd451b3efd)

## Features

* **Gemini-Powered**: Leverages Google GenAI’s Gemini 2.0 Flash model for natural language understanding and generation.
* **LangGraph Workflow**: Uses LangGraph to orchestrate message flows, tool invocation, and memory persistence for multi-turn conversations.
* **Modular Tooling**:

  * **Weather Lookup** (`get_weather`, `summarize_weather`): Fetch current weather data and summarize it.
  * **YouTube Transcript** (`get_youtube_transcript`): Retrieve timestamped transcripts from YouTube videos.
  * **Email Sending** (`send_email`): Send emails via Gmail SMTP using an app-specific password.
  * **OCR & Image Recognition**: Includes Tesseract for OCR and YOLOv8 (via Ultralytics) for image detection (if extended in future).
* **Frontend UI**: A minimal, user-friendly chat interface (`index.html`) with:

  * Chat bubbles (user vs. AI), timestamps, typing indicators.
  * A "plus" button to toggle additional tool options (web search, news, weather, email, YouTube summary).
  * Responsive styling using modern CSS, optimized for clarity and ease of use.
* **CORS-Enabled FastAPI**: Serves API endpoints with CORS configuration to support any frontend origin.
* **Environment-Based Configuration**: All sensitive API keys (Gemini, OpenWeather, Tavily, Gmail) are managed via a `.env` file.
* **Thread Management**: Conversations are tracked via `thread_id`, ensuring context persists across multiple user exchanges.

## Folder Structure

```plaintext
├── main.py
├── requirements.txt
├── index.html
└── tools
    └── chatbot_tools.py
```

* `main.py`: The FastAPI application that exposes the `/chat` endpoint and wires LangGraph.
* `requirements.txt`: Python dependencies required to run the project.
* `index.html`: Frontend chat UI that interacts with the FastAPI backend.
* `tools/chatbot_tools.py`: Contains tool implementations (weather, YouTube transcript, email, OCR, etc.) and LangGraph graph definition.

## Prerequisites

* Python 3.9 or higher
* A Google Cloud project with access to the Gemini 2.0 Flash model (GenAI API)
* OpenWeatherMap API key
* Tavily API key (for Bing/alternative web search)
* Gmail address & App Password for SMTP
* (Optional) Tesseract OCR installed on your system for OCR capabilities
* (Optional) GPU or CPU with Ultralytics YOLOv8 installed for image detection (if used)

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/advance-bot.git
   cd advance-bot
   ```

2. **Set Up a Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate    # Windows
   ```

3. **Install Python Dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Install Tesseract (Optional)**

   * **Ubuntu/Debian**:

     ```bash
     sudo apt-get update
     sudo apt-get install tesseract-ocr
     ```
   * **macOS (Homebrew)**:

     ```bash
     brew install tesseract
     ```
   * **Windows**: Download the Tesseract installer from [https://github.com/tesseract-ocr/tesseract/releases](https://github.com/tesseract-ocr/tesseract/releases) and follow the setup instructions. Ensure the installation path is added to your `PATH` environment variable.

5. **Create a `.env` File**
   In the root directory, create a file named `.env` and add the following variables:

   ```dotenv
   TAVILY_API_KEY=your_tavily_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   OPENWEATHER_API_KEY=your_openweather_api_key_here
   GMAIL_ADDRESS=your_gmail_address_here
   GMAIL_APP_PASSWORD=your_gmail_app_password_here
   ```

## Environment Variables Explanation

* `TAVILY_API_KEY`: API key for Tavily/Bing (used in `TavilySearch` for web search tool).
* `GOOGLE_API_KEY`: API key for Google GenAI (used to authenticate the Gemini model).
* `OPENWEATHER_API_KEY`: API key for OpenWeatherMap (used by `get_weather` tool).
* `GMAIL_ADDRESS`: Your Gmail address used as sender for `send_email` tool.
* `GMAIL_APP_PASSWORD`: An app-specific password generated from your Google Account security settings. Used to authenticate SMTP.

## Usage

### 1. Run the FastAPI Backend

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

* The API will be accessible at `http://localhost:8000` (or your specified host/port).
* The main endpoint is:

  * `POST /chat` – Accepts a JSON payload with `{ "message": "<your_message>", "thread_id": "<optional_thread_id>" }` and returns `{ "response": "<ai_response>", "thread_id": "<thread_id>" }`.

### 2. Serve the Frontend

Since `index.html` is a static file, you can serve it with any static server or just open it directly in your browser. However, to avoid CORS issues, it’s recommended to serve it via a simple HTTP server:

```bash
# From the project root, run:
# Python 3.x built-in HTTP server
python -m http.server 8080
```

Then open your browser at `http://localhost:8080/index.html`. The frontend will connect to the FastAPI backend. If you deploy your backend elsewhere (e.g., Render, Heroku), update the `API_ENDPOINT` in the `<script>` section of `index.html` accordingly.

### 3. Interact with the Chatbot

* Type a message in the input box and click "Send" or press Enter.
* The AI’s response appears as a chat bubble with a timestamp.
* Click the "+" button to toggle additional tool options (e.g., send mail, get weather, summarize YouTube videos). Toggle an option on/off to let the model know which tools are active.
* The conversation context is preserved via `thread_id`. If you reload the page or reinitialize the frontend, it will start a fresh `thread_id` unless you store and reuse the previous one.

## Tools Overview (in `tools/chatbot_tools.py`)

* **`get_weather(city: str) -> str`**

  * Fetches current weather data for a given city using OpenWeatherMap.
  * Returns a string summary: "The weather in {city} is {description} with a temperature of {temp}°C."

* **`summarize_weather(city: str) -> str`**

  * Fetches raw JSON weather data, builds a prompt containing that JSON, and asks Gemini to produce a detailed, human-readable summary (includes humidity, wind speed, etc.).

* **`get_youtube_transcript(video_url: str) -> str`**

  * Extracts a YouTube video ID via regex, retrieves the transcript via `youtube_transcript_api`, and formats it with timestamps.
  * Caches transcripts in memory for repeated calls.

* **`send_email(recipient: str, subject: str, body: str) -> str`**

  * Sends an email via Gmail’s SMTP server using your `GMAIL_ADDRESS` and `GMAIL_APP_PASSWORD`.
  * Returns success/failure messages.

* **(Future Extensions)**: You can integrate OCR (via `pytesseract`), object detection (via Ultralytics YOLO), and other tools as needed. Utility functions for extracting YouTube IDs (`extract_video_id`) and formatting timestamps are provided.

## LangGraph Setup

1. **State Definition** (`State`) – Holds a list of messages with `add_messages` annotation.
2. **Nodes**:

   * `chatbot_node`: Sends the full message history (including system instructions) to Gemini, returning an `AIMessage` with potential `tool_calls`.
   * `BasicToolNode`: Iterates over `tool_calls` from the last LLM output, invokes the corresponding Python functions, and wraps results as `ToolMessage` for the next node.
3. **Edges**:

   * Starts at `chatbot` node, and if the LLM outputs any `tool_calls`, routes to `tools` node; otherwise, ends and returns the last `AIMessage` content.
4. **Memory**: `MemorySaver` is used for persistent memory across runs (e.g., storing conversation history or relevant data).
5. **Invocation**: `run_chat_through_graph(user_message, thread_id)` builds a system and user message list, generates/uses a `thread_id`, and invokes the compiled graph to produce a response and updated `thread_id`.

## Index.html Frontend Highlights

* **Header**: Branding with a logo and the title "Advance Bot (Adbot)".
* **Chat Area**: Dynamic insertion of chat bubbles for user/AI messages; includes fade-in animations, distinct styling, and timestamps.
* **Typing Indicator**: Displays animated dots while awaiting AI response.
* **Footer Input**: Text input for messages, a toggleable "+" button showing additional tool options, and a send button.
* **Options Dropdown**: Contains buttons for toggling specific tool contexts (web search, news, weather, send mail, YouTube summarize).
* **JavaScript Logic**:

  * Manages `threadId` state across requests.
  * Handles CORS-enabled fetch requests to the FastAPI endpoint.
  * Toggling of tool options adds an `active` class to buttons, allowing the LLM to know which tools are currently available in the conversation.
  * Enables/disables input and send button based on loading states.

## Deployment

1. **Backend**:

   * Deploy the FastAPI app to any cloud provider (e.g., Render, Heroku, AWS, GCP) with environment variables set.
   * Example (Render): Create a "Web Service" with the command: `uvicorn main:app --host 0.0.0.0 --port $PORT`.
   * Ensure the `.env` environment variables are configured in your deployment settings.

2. **Frontend**:

   * Serve `index.html` via a static hosting service (Netlify, Vercel, GitHub Pages) or alongside your backend (e.g., using FastAPI’s `StaticFiles` middleware).
   * Update the `API_ENDPOINT` in `index.html` to match your deployed backend URL (e.g., `https://your-domain.com/chat`).

3. **Domain & SSL**:

   * Configure a custom domain for both frontend and backend, ensuring HTTPS for secure API calls.

## Contributing

If you’d like to add new tools, improve prompts, or enhance the frontend:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature-name`.
3. Implement your changes.
4. Add documentation in `README.md` and update `requirements.txt` if adding new dependencies.
5. Submit a pull request with a clear description of your changes.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

*For any issues or support, please open an issue on GitHub or contact the maintainer directly.*
