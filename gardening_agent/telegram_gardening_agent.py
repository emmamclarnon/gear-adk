import os
import requests
import base64
from typing import Annotated, List, Dict, Any
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.error import BadRequest

load_dotenv()


# ============================================================================
# 1. State Schema & Tools
# ============================================================================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    location: str
    active_crops: List[str]


@tool
def get_weather_forecast(latitude: float = 53.8, longitude: float = -1.5) -> str:
    """Fetch 7-day weather forecast and rain totals from Open-Meteo API. Tags frost hazard."""
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={latitude}&longitude={longitude}"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
        f"&timezone=auto"
    )
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        daily = data.get("daily", {})

        forecast_summary = ["### 7-Day Local Weather Forecast:"]
        for i in range(len(daily.get("time", []))):
            date = daily["time"][i]
            t_max = daily["temperature_2m_max"][i]
            t_min = daily["temperature_2m_min"][i]
            precip = daily["precipitation_sum"][i]

            frost_warning = " ⚠️ FROST WARNING (<= 2°C)" if t_min <= 2.0 else ""
            forecast_summary.append(
                f"- **{date}**: Min {t_min}°C / Max {t_max}°C | Rain: {precip}mm{frost_warning}"
            )
        return "\n".join(forecast_summary)
    except Exception as e:
        return f"Error retrieving weather data: {str(e)}"


@tool
def query_companion_planting(crop: str) -> str:
    """Look up companion plants and poor plant neighbors."""
    knowledge_base = {
        "tomato": "Good companions: Basil, Marigold, Garlic, Nasturtiums. Avoid: Brassicas, Fennel, Potatoes.",
        "carrots": "Good companions: Lettuce, Radish, Onions, Rosemary, Sage. Avoid: Parsnips, Dill.",
        "garlic": "Good companions: Tomatoes, Peppers, Brassicas, Fruit Trees. Avoid: Beans, Peas, Asparagus.",
        "brassicas": "Good companions: Mint, Sage, Garlic, Onions, Potato. Avoid: Tomatoes, Strawberries, Pole Beans."
    }
    key = crop.lower().strip()
    return knowledge_base.get(
        key,
        f"No direct match for '{crop}'. Aromatic herbs and alliums usually repel pests; legumes fix soil nitrogen."
    )


@tool
def find_seed_suppliers(crop_or_variety: str) -> str:
    """Find reputable UK seed suppliers with low P&P costs or free shipping thresholds."""
    suppliers = [
        {"name": "GrowSeed", "p_and_p": "From £1.80", "free_delivery": "FREE over £15",
         "url": "https://www.growseed.co.uk"},
        {"name": "Premier Seeds Direct", "p_and_p": "~£1.50 - £2.20", "free_delivery": "Tiered rates",
         "url": "https://premierseedsdirect.com"},
        {"name": "Budget Seeds", "p_and_p": "Flat £2.50", "free_delivery": "N/A", "url": "https://budgetseeds.co.uk"}
    ]
    results = [f"### Recommended Low P&P UK Seed Merchants for '{crop_or_variety}':\n"]
    for supplier in suppliers:
        results.append(
            f"**[{supplier['name']}]({supplier['url']})**\n"
            f"- **P&P:** {supplier['p_and_p']} | **Free Post:** {supplier['free_delivery']}\n"
        )
    return "\n".join(results)


tools = [get_weather_forecast, query_companion_planting, find_seed_suppliers]

# ============================================================================
# 2. Agent Setup & Helpers
# ============================================================================
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=api_key
).bind_tools(tools)

SYSTEM_PROMPT = """You are an expert personal AI Gardening Assistant.
Always keep the user's garden context in mind:
- Location: {location}
- Active crops: {active_crops}

Use your available tools for weather, companion planting, and seed suppliers whenever appropriate.
When an image is provided, examine it carefully to identify plant diseases, pests, or deficiencies, and provide clear, practical advice.
"""


def agent_node(state: AgentState) -> Dict[str, Any]:
    formatted_prompt = SYSTEM_PROMPT.format(
        location=state.get("location", "Leeds, UK"),
        active_crops=", ".join(state.get("active_crops", ["Tomatoes", "Garlic", "Brassicas"]))
    )
    messages = [SystemMessage(content=formatted_prompt)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")


def extract_text_from_message(msg: AIMessage) -> str:
    """Safely extracts string text from AIMessage content across string, list, or dict types."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif "text" in block:
                    parts.append(block["text"])
        return "\n".join(parts)
    elif isinstance(content, dict):
        return content.get("text", "")
    return str(content)


async def send_safe_reply(update: Update, text: str):
    """Sends reply with Markdown, falling back to plain text if Markdown syntax error occurs."""
    try:
        await update.message.reply_text(text, parse_mode="Markdown")
    except BadRequest:
        await update.message.reply_text(text)


# ============================================================================
# 3. Telegram Message Handlers
# ============================================================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🌱 Hello! I'm your AI Gardening Assistant.\n\n"
        "Ask me about weather/frost forecasts, companion planting, cheap seed suppliers, "
        "or snap a photo of any plant/leaf issue and send it to me for visual diagnosis!"
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    chat_id = str(update.effective_chat.id)

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    config = {"configurable": {"thread_id": f"telegram_{chat_id}"}}
    input_state = {
        "messages": [HumanMessage(content=user_text)],
        "location": "Leeds, UK",
        "active_crops": ["Tomatoes", "Garlic", "Brassicas"]
    }

    with SqliteSaver.from_conn_string("gardening_agent_memory.db") as checkpointer:
        app = builder.compile(checkpointer=checkpointer)
        final_response = ""

        events = app.invoke(input_state, config=config)

        if "messages" in events:
            last_msg = events["messages"][-1]
            if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
                final_response = extract_text_from_message(last_msg)

    if final_response:
        await send_safe_reply(update, final_response)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    caption = update.message.caption or "Please diagnose this plant photo for any pests, diseases, or deficiencies."

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    photo_file = await update.message.photo[-1].get_file()
    image_bytes = await photo_file.download_as_bytearray()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    multimodal_message = HumanMessage(
        content=[
            {"type": "text", "text": caption},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            }
        ]
    )

    config = {"configurable": {"thread_id": f"telegram_{chat_id}"}}
    input_state = {"messages": [multimodal_message]}

    with SqliteSaver.from_conn_string("gardening_agent_memory.db") as checkpointer:
        app = builder.compile(checkpointer=checkpointer)
        final_response = ""

        events = app.invoke(input_state, config=config)

        if "messages" in events:
            last_msg = events["messages"][-1]
            if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
                final_response = extract_text_from_message(last_msg)

    if final_response:
        await send_safe_reply(update, final_response)


# ============================================================================
# 4. Main Entry Point
# ============================================================================
def main():
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not telegram_token:
        raise ValueError("TELEGRAM_BOT_TOKEN is missing from your .env file!")

    app_bot = ApplicationBuilder().token(telegram_token).build()

    app_bot.add_handler(CommandHandler("start", start_command))
    app_bot.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_text))
    app_bot.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    print("🤖 Gardening Telegram Bot is running! Open Telegram and message your bot.")
    app_bot.run_polling()


if __name__ == "__main__":
    main()