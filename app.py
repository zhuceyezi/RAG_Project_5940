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

if "current_response" not in st.session_state:
    st.session_state.current_response = "Game is starting. What would you like to do?"

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

# Apply the medieval styling
render_medieval_css()

# === Configure the sidebar for tools ===
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h2>Game Master Tools</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # NPC Generator in sidebar
    st.header("🧙 NPC Generator")
    npc_count = st.number_input("How many NPCs?", min_value=1, max_value=10, value=3)
    
    if st.button("Generate NPCs"):
        result = sample_npcs(npc_count)
        st.session_state["npcs"] = result["npcs"]
        st.success(f"✅ {len(result['npcs'])} NPCs Generated!")
    
    # Script Upload in sidebar
    st.header("📘 Upload Script")
    
    uploaded_file = st.file_uploader("Upload campaign script (PDF/TXT)", type=["pdf", "txt"])
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            try:
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                plot = ""
                for page in pdf_reader.pages:
                    plot += page.extract_text() or ""
            except Exception as e:
                st.sidebar.error(f"❌ Could not read PDF: {e}")
                plot = ""
        elif uploaded_file.type == "text/plain":
            plot = uploaded_file.read().decode("utf-8")
        else:
            st.sidebar.warning("Unsupported file type")
            plot = ""
            
        st.session_state["script_text"] = plot
        
        # Process scene data if new script is uploaded
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
        
        st.sidebar.success("✅ Scene map extracted!")
        
    # Display NPC details in sidebar if they exist
    if "npcs" in st.session_state and st.session_state["npcs"]:
        st.header("🧙 NPCs Details")
        for npc in st.session_state["npcs"]:
            with st.expander(f"{npc.name} ({npc.char_class})"):
                st.markdown(f"**Race:** {npc.race}")
                st.markdown(f"**Personality:** {npc.personality}")
                st.markdown(f"**Combat Role:** {npc.combat_role}")
                st.markdown(f"**Quirks:** {npc.quirks}")
                st.markdown(f"**Voice:** {npc.voice}")
                st.markdown(f"**Trait:** {npc.trait}")
                
                # Display stats if available
                if hasattr(npc, "stats") and isinstance(npc.stats, dict):
                    st.markdown("**Stats:**")
                    for stat, value in npc.stats.items():
                        st.markdown(f"- {stat}: {value}")

# === Main Layout with Columns ===
# Create main columns: left for game, right for current situation
left_col, right_col = st.columns([3, 1])

with left_col:
    # Map Container
    st.markdown("""
    <div class="map-outer-container">
        <h1>Map</h1>
        <div class="map-inner-container">
            <!-- Map image will be inserted here -->
    """, unsafe_allow_html=True)

    # Insert map image if available
    if "scene_graph_img" in st.session_state:
        st.markdown(f"""
            <img src="data:image/png;base64,{st.session_state['scene_graph_img']}" 
                 class="map-image">
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="map-placeholder">
                No map available. Upload a script to generate one.
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("""
        </div>
        
        <!-- Text Input -->
        <div class="text-input-container">
    """, unsafe_allow_html=True)
    
    # Text input for commands
    prompt = st.text_input("", placeholder="Enter your action...", key="medieval_input")
    
    st.markdown("""
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Character cards row
    st.markdown("""<div class="character-row">""", unsafe_allow_html=True)
    
    # Create character cards using our custom renderer
    render_character_cards_v2(
        st.session_state.players, 
        st.session_state.get("npcs", [])
    )
    
    st.markdown("""</div>""", unsafe_allow_html=True)

# Right column for current situation
with right_col:
    st.markdown("""
    <div class="current-situation">
        <h3>Generated respond/current situation:</h3>
        <div class="situation-content">
    """, unsafe_allow_html=True)
    
    # Display current response
    st.markdown(f"""
        <p>{st.session_state.current_response}</p>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        </div>
        <div class="situation-notes">
            <p>Depends on how many npc were generated, we used 5 blocks here as example</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Process chat input if entered
if prompt:
    st.session_state.chat_log.append({"role": "user", "content": prompt})
    
    # Define the agent using your helper class
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
    )
    
    agent = Agent(
        name="DungeonMaster",
        model="openai.gpt-4o",
        instructions=instructions,
        tools=[roll_dice, sample_npcs, move_to_scene, add_npc, remove_npc],
        players=st.session_state.players
    )
    
    # Run GPT with tool support
    new_messages, tool_outputs = run_full_turn(client, agent, st.session_state.chat_log)
    st.session_state.chat_log.extend(new_messages)
    
    # Update the current response with the latest assistant message
    for msg in new_messages:
        role = msg["role"] if isinstance(msg, dict) else msg.role
        content = msg["content"] if isinstance(msg, dict) else msg.content
        
        if content is None or role == "tool":
            continue  # hide tool result content
        
        if role == "assistant":
            st.session_state.current_response = content
    
    # Apply effects like healing or damage
    apply_hp_effects(client, new_messages, agent.players)
    
    # Rerun to update the UI
    st.experimental_rerun()