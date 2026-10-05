import os
import requests
from typing import Annotated, List, Dict, Any
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver


# ============================================================================
# 1. State Schema
# ============================================================================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    location: str
    active_crops: List[str]


# ============================================================================
# 2. Tools
# ============================================================================
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
        {
            "name": "GrowSeed",
            "p_and_p": "From £1.80 (2nd Class) / £2.00 (1st Class)",
            "free_delivery": "FREE on seed orders over £15",
            "highlights": "DEFRA certified, carbon-neutral shipping.",
            "url": "https://www.growseed.co.uk"
        },
        {
            "name": "Premier Seeds Direct",
            "p_and_p": "Standard Royal Mail letter rate (~£1.50 - £2.20)",
            "free_delivery": "Periodic discounts / tiered delivery",
            "highlights": "Minimal eco-packaging, low price per packet.",
            "url": "https://premierseedsdirect.com"
        },
        {
            "name": "Budget Seeds",
            "p_and_p": "Flat £2.50 per order (unlimited packet count)",
            "free_delivery": "N/A (Flat rate)",
            "highlights": "DEFRA registered, cheap seed catalog under £1 per packet.",
            "url": "https://budgetseeds.co.uk"
        }
    ]

    results = [f"### Recommended Low P&P UK Seed Merchants for '{crop_or_variety}':\n"]
    for supplier in suppliers:
        results.append(
            f"**[{supplier['name']}]({supplier['url']})**\n"
            f"- **P&P Fee:** {supplier['p_and_p']}\n"
            f"- **Free Post:** {supplier['free_delivery']}\n"
            f"- **Highlights:** {supplier['highlights']}\n"
        )
    return "\n".join(results)


tools = [get_weather_forecast, query_companion_planting, find_seed_suppliers]

# ============================================================================
# 3. Agent Node & System Prompt
# ============================================================================
import os

# Retrieve API key from environment (checking both GOOGLE_API_KEY and GEMINI_API_KEY)
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "API key missing! Please set GOOGLE_API_KEY or GEMINI_API_KEY in your environment, "
        "or pass it directly to ChatGoogleGenerativeAI(google_api_key='...')."
    )

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=api_key
).bind_tools(tools)


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
).bind_tools(tools)

SYSTEM_PROMPT = """You are an expert personal AI Gardening Assistant.
Always keep the user's garden context in mind:
- Location: {location}
- Active crops: {active_crops}

Use your available tools for weather, companion planting, and seed suppliers whenever appropriate.
"""


def agent_node(state: AgentState) -> Dict[str, Any]:
    formatted_prompt = SYSTEM_PROMPT.format(
        location=state.get("location", "Leeds, UK"),
        active_crops=", ".join(state.get("active_crops", ["Tomatoes", "Garlic", "Brassicas"]))
    )
    messages = [SystemMessage(content=formatted_prompt)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# ============================================================================
# 4. Graph Assembly
# ============================================================================
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")


# ============================================================================
# 5. Execution Test Run
# ============================================================================
def main():
    db_path = "gardening_agent_memory.db"

    # Use SqliteSaver as a context manager for safe connection management
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        app = builder.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "test_session_01"}}

        print("=" * 80)
        print("RUNNING TEST TURN 1: Weather & Frost Query")
        print("=" * 80)

        turn_1_input = {
            "messages": [HumanMessage(content="Is there any frost risk coming up in Leeds for my tomato plants?")],
            "location": "Leeds, UK",
            "active_crops": ["Tomatoes", "Garlic", "Brassicas"]
        }

        # Stream the graph execution
        for event in app.stream(turn_1_input, config=config, stream_mode="values"):
            if "messages" in event:
                last_msg = event["messages"][-1]
                if isinstance(last_msg, AIMessage) and last_msg.content:
                    print(f"\n[Agent Output]:\n{last_msg.content}")

        print("\n" + "=" * 80)
        print("RUNNING TEST TURN 2: Memory & Companion Planting Check")
        print("=" * 80)

        # Turn 2 omits location/crops to test if SqliteSaver recalls context from Turn 1
        turn_2_input = {
            "messages": [HumanMessage(
                content="What should I plant right next to my tomatoes, and where can I buy seeds cheaply?")]
        }

        for event in app.stream(turn_2_input, config=config, stream_mode="values"):
            if "messages" in event:
                last_msg = event["messages"][-1]
                if isinstance(last_msg, AIMessage) and last_msg.content:
                    print(f"\n[Agent Output]:\n{last_msg.content}")


if __name__ == "__main__":
    main()