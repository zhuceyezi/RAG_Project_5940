import streamlit as st
from openai import OpenAI
import openai
from pydantic import BaseModel, Field
from utils import *
import textwrap
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

# Apply custom theme
set_dnd_theme()

# === Set up session state ===
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

if "players" not in st.session_state or not isinstance(list(st.session_state.players.values())[0], Player):
    st.session_state.players = {
        "Aragorn": Player(name="Aragorn", max_hp=30, hp=30),
    }

# === Map font ===
if "map_font" not in st.session_state:
    st.session_state["map_font"] = "serif"  # Default font


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
st.title("🐉 D&D Adventure Master")
st.markdown('<div class="dnd-divider"></div>', unsafe_allow_html=True)

# === Script Upload Section ===
with st.container():
    st.header("📜 Adventure Script")
    st.markdown("Upload your campaign script to start your journey")

    uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])
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

        st.success("✅ Your adventure script has been loaded! The mystical maps are being drawn...")

        with st.expander("📖 Script Preview"):
            st.text_area("The tale unfolds...", plot, height=300)

# Extract scene map from script
if "scene_graph_img" not in st.session_state and "script_text" in st.session_state:
    # First, extract scene data
    messages = [{"role": "user", "content": plot}]
    scene_response, _ = run_full_turn(client, scene_graph_agent, messages)
    scene_data_raw = scene_response[-1].content if hasattr(scene_response[-1], "content") else scene_response[-1]["content"]

    # Strip ```json ... ``` if GPT wrapped it
    scene_data_raw = re.sub(r"^```(?:json)?|```$", "", scene_data_raw.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()
    scene_data = json.loads(scene_data_raw)

    # Store scene data
    st.session_state["scene_list"] = scene_data
    st.session_state["current_scene"] = scene_data[0]["title"] if scene_data else "Unknown"

    # No longer using map_description_agent - removed that section

    # Generate and save the map image directly from scene data
    with st.spinner("🗺️ Generating fantasy map..."):
        map_path = generate_map_image(client, scene_data)  # Updated signature
        if map_path:
            st.session_state["base_map_path"] = map_path

            # Create the scene graph with map background
            graph_path = save_scene_graph_with_map_background(
                scene_data,
                map_path,
                current_location=st.session_state["current_scene"],
                custom_font=st.session_state.get("map_font", "serif")
            )

            with open(graph_path, "rb") as f:
                encoded_img = base64.b64encode(f.read()).decode()
                st.session_state["scene_graph_img"] = encoded_img
                st.success("✅ Adventure map successfully generated!")
        else:
            # Fallback to regular scene graph if map generation fails
            graph_path = save_scene_graph_image(scene_data, current_location=st.session_state["current_scene"])
            with open(graph_path, "rb") as f:
                encoded_img = base64.b64encode(f.read()).decode()
                st.session_state["scene_graph_img"] = encoded_img

# === NPC Generator Section ===
st.markdown('<div class="dnd-divider"></div>', unsafe_allow_html=True)
with st.container():
    st.header("🧙 NPC Recruitment Hall")
    st.markdown("Recruit companions and antagonists for your quest")

    col1, col2 = st.columns([3, 1])
    with col1:
        npc_count = st.slider("How many NPCs would you like to recruit?", min_value=1, max_value=10, value=3)

    with col2:
        if st.button("Summon NPCs", use_container_width=True):
            with st.spinner("🔮 Summoning characters from the realms..."):
                result = sample_npcs(npc_count)
                st.session_state["npcs"] = result["npcs"]
                st.success(f"✨ {len(result['npcs'])} NPCs have joined your adventure!")

    # Display NPCs in a grid layout if they exist
    if "npcs" in st.session_state and st.session_state["npcs"]:
        st.markdown("### Your Companions & Foes")

        # Create a custom grid layout for NPCs
        npc_html = '<div class="npc-grid">'

        for npc in st.session_state["npcs"]:
            # Calculate HP percentage for the progress bar
            hp_percent = (npc.hp / npc.max_hp) * 100

            # Create NPC card with styling
            npc_html += textwrap.dedent(f"""
                <div class="character-card">
                    <h4>{npc.name}</h4>
                    <p><strong>{npc.race} {npc.char_class}</strong></p>
                    <div class="hp-bar-container">
                        <div class="hp-bar" style="width: {hp_percent}%;"></div>
                    </div>
                    <p style="text-align: center; margin: 0;">{npc.hp}/{npc.max_hp} HP</p>
                    <p><em>"{npc.personality}"</em></p>
                    <p><strong>Role:</strong> {npc.combat_role}</p>
                    <p><strong>Quirk:</strong> {npc.quirks}</p>
                    <p><strong>Voice:</strong> {npc.voice}</p>
                    <p><strong>Trait:</strong> {npc.trait}</p>
                </div>
            """)

        npc_html += '</div>'
        st.markdown(npc_html, unsafe_allow_html=True)

# === Chat Interface ===
st.markdown('<div class="dnd-divider"></div>', unsafe_allow_html=True)
st.header("⚔️ Adventure Journey")

# === Define the agent using your helper class ===
npc_context = ""
if "npcs" in st.session_state:
    npc_context = build_npc_summary(st.session_state["npcs"])

instructions = (
    "You are a Dungeons & Dragons Dungeon Master. Narrate what happens. "
    "You can use the `roll_dice` tool when needed. "
    "Use vivid storytelling and refer to characters naturally.\n\n"
    f"The following NPCs are present:\n{npc_context}"
    f"Current script: {st.session_state['script_text'] if 'script_text' in st.session_state else ''}"
    f"You should only use the available scenes when you decide to move. If no major scene change, you can stay at the same scene."
    f"Here are the available scenes: {st.session_state['scene_list'] if 'scene_list' in st.session_state else ''}"
    f"Try to refer to stats when applicable."
    "When an NPC’s attitude toward the party should change based on game events or history, "
    "call the `set_npc_alignment` tool with the NPC’s name and new alignment ('ally','enemy','neutral').\n"
    "For example: @tool set_npc_alignment(name=\"Goblin Chief\", alignment=\"enemy\").\n"
    "For combat, use calculate_attack and describe the results dramatically.\n"
    "Example: @tool calculate_attack(attacker_name=\"Aragorn\", defender_name=\"Goblin\")\n"
)

agent = Agent(
    name="DungeonMaster",
    model="openai.gpt-4o",
    instructions=instructions,
    tools=[roll_dice, sample_npcs, move_to_scene, add_npc, remove_npc, set_npc_alignment, calculate_attack],
    players=st.session_state.players
)

# Display chat interface with custom styling
chat_container = st.container()
with chat_container:
    # Display chat log
    for msg in st.session_state.chat_log:
        role = msg["role"] if isinstance(msg, dict) else msg.role
        content = msg["content"] if isinstance(msg, dict) else msg.content

        if content is None or role == "tool":
            continue  # hide tool result content

        with st.chat_message(role):
            st.markdown(content)

    # Chat input
    if prompt := st.chat_input("What is your next action, brave adventurer?"):
        st.session_state.chat_log.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Run GPT with tool support
        with st.spinner("🎲 The Dungeon Master is considering..."):
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

# === Enhanced Sidebar ===
render_sidebar(agent.players)
if "npcs" in st.session_state:
    render_npcs(st.session_state["npcs"])
    
# Display scene map if enabled
if st.session_state.get("show_map") and "scene_graph_img" in st.session_state:
    render_scene_graph_bottom_right(st.session_state["scene_graph_img"])
    # render_scene_graph_right_panel(st.session_state["scene_graph_img"])