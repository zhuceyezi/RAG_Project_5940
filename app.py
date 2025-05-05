import streamlit as st
from openai import OpenAI
import openai
from pydantic import BaseModel, Field
from utils import *
import PyPDF2
from dotenv import load_dotenv
import os
import base64
from utils import save_scene_graph_image

load_dotenv()  # Automatically loads from `.env` in current directory

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

# === Set up OpenAI client ===
client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

# === Set up session state ===
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

if "players" not in st.session_state or not isinstance(list(st.session_state.players.values())[0], Player):
    st.session_state.players = {
        "Aragorn": Player(name="Aragorn", max_hp=30, hp=30),
    }


scene_graph_agent = Agent(
    name="SceneGraphAgent",
    model="openai.gpt-4o",
    instructions="""
    You are a Dungeons & Dragons campaign analyst. Given a full script of a one-shot adventure, extract a structured scene map.

    For each scene, return:
    - scene_number: the number of the scene
    - title: short name for the scene
    - location: where the scene happens (village, forest, shrine, etc.)
    - description: short summary of what happens
    - connections: list of scene titles or numbers that logically follow this one

    Return a JSON array like:
    [
    {
        "scene_number": 1,
        "title": "The Village of Moonshade",
        "location": "Moonshade Valley",
        "description": "Party meets Mayor and learns of ghostly singing.",
        "connections": ["Misty Forest Path"]
    },
    ...
    ]
    Only include story-driving scenes (ignore rewards, stat blocks, etc).
    Don't include Scene number in connections. Only the name.
    """,
    tools=[],
    players={}
)

# === Streamlit UI ===
st.title("🧙 D&D Chat with Dice Rolls")
st.caption("Powered by Azure OpenAI + Tool Calling")


st.header("🧙 NPC Generator Test")

npc_count = st.number_input("How many NPCs do you want to generate?", min_value=1, max_value=10, value=3)

if st.button("Generate NPCs"):
    result = sample_npcs(npc_count)
    st.session_state["npcs"] = result["npcs"]

    st.success("NPCs Generated:")
    for npc in result["npcs"]:
        with st.expander(f"🧍 {npc.name} - {npc.char_class} ({npc.race})"):
            st.markdown(f"- **Personality:** {npc.personality}")
            st.markdown(f"- **Combat Role:** {npc.combat_role}")
            st.markdown(f"- **Quirks:** {npc.quirks}")
            st.markdown(f"- **Voice Style:** {npc.voice}")
            st.markdown(f"- **Unique Trait:** {npc.trait}")

# === Script Upload ===
st.header("📘 Upload Your Script")

uploaded_file = st.file_uploader("Upload your campaign script (PDF or TXT)", type=["pdf", "txt"])
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            plot = ""
            for page in pdf_reader.pages:
                plot += page.extract_text() or ""
        except Exception as e:
            st.error(f"❌ Could not read PDF: {e}")
            text = ""
    elif uploaded_file.type == "text/plain":
        plot = uploaded_file.read().decode("utf-8")
    else:
        st.warning("Unsupported file type")
        plot = ""
    st.session_state["script_text"] = plot
    st.success("✅ Scene map successfully extracted!")

    st.subheader("📜 Script Preview")
    st.text_area("Script contents:", plot, height=400)

else:
    st.info("Please upload a PDF or TXT file to begin setting up your world.")

if "scene_graph_img" not in st.session_state and "script_text" in st.session_state:

    messages = [{"role": "user", "content": plot}]
    scene_response, _ = run_full_turn(client, scene_graph_agent, messages)
    scene_data_raw = scene_response[-1].content if hasattr(scene_response[-1], "content") else scene_response[-1]["content"]

    # Strip ```json ... ``` if GPT wrapped it
    scene_data_raw = re.sub(r"^```(?:json)?|```$", "", scene_data_raw.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()

    scene_data = json.loads(scene_data_raw)

    st.session_state["scene_list"] = scene_data
    st.session_state["current_scene"] = scene_data[0]["title"] if scene_data else "Unknown"
    graph_path = save_scene_graph_image(scene_data, current_location=st.session_state["current_scene"])
    with open(graph_path, "rb") as f:
        encoded_img = base64.b64encode(f.read()).decode()

    # Store encoded image for display
    st.session_state["scene_graph_img"] = encoded_img


# === Define the agent using your helper class ===
npc_context = ""
if "npcs" in st.session_state:
    npc_context = build_npc_summary(st.session_state["npcs"])

instructions = (
    "You are a Dungeons & Dragons Dungeon Master. Narrate what happens. "
    "You can use the `roll_dice` tool when needed. "
    "Use vivid storytelling and refer to characters naturally.\n\n"
    f"The following NPCs are present:\n{npc_context}"
    f"Current script: {st.session_state["script_text"] if "script_text" in st.session_state else ''}"
    f"Available Scenes: {st.session_state["scene_list"] if "scene_list" in st.session_state else ''}"
)

agent = Agent(
    name="DungeonMaster",
    model="openai.gpt-4o",
    instructions=instructions,
    tools=[roll_dice, sample_npcs, move_to_scene, add_npc, remove_npc],
    players=st.session_state.players
)


# === Display chat log ===
for msg in st.session_state.chat_log:
    role = msg["role"] if isinstance(msg, dict) else msg.role
    content = msg["content"] if isinstance(msg, dict) else msg.content

    if content is None or role == "tool":
        continue  # hide tool result content

    with st.chat_message(role):
        st.markdown(content)


# === Chat input ===
if prompt := st.chat_input("What happens next?"):
    st.session_state.chat_log.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run GPT with tool support
    new_messages, tool_outputs = run_full_turn(client, agent, st.session_state.chat_log)
    st.session_state.chat_log.extend(new_messages)

    # Display assistant messages in bubbles
    for msg in new_messages:
        role = msg["role"] if isinstance(msg, dict) else msg.role
        content = msg["content"] if isinstance(msg, dict) else msg.content

        if content is None:
            continue  # Skip messages with no displayable content

        with st.chat_message(role):
            st.markdown(content)

    # Apply effects like healing or damage
    apply_hp_effects(client, new_messages, agent.players)

render_sidebar(agent.players)
if "npcs" in st.session_state:
    render_npcs(st.session_state["npcs"])
    
if "scene_graph_img" in st.session_state:
    render_scene_graph_bottom_right(st.session_state["scene_graph_img"])
    # render_scene_graph_right_panel(st.session_state["scene_graph_img"])
