import streamlit as st
from openai import OpenAI
import openai
from pydantic import BaseModel, Field
from utils import *
import PyPDF2

# === Set up OpenAI client ===
client = OpenAI(
    api_key="sk-xIvD5Mw4sc1_owNpoJcO8g",
    base_url="https://api.ai.it.cornell.edu/"
)

# === Set up session state ===
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

if "players" not in st.session_state or not isinstance(list(st.session_state.players.values())[0], Player):
    st.session_state.players = {
        "Aragorn": Player(name="Aragorn", max_hp=30, hp=30),
        "Frodo": Player(name="Frodo", max_hp=20, hp=20)
    }


# === Define the agent using your helper class ===
npc_context = ""
if "npcs" in st.session_state:
    npc_context = build_npc_summary(st.session_state["npcs"])

instructions = (
    "You are a Dungeons & Dragons Dungeon Master. Narrate what happens. "
    "You can use the `roll_dice` tool when needed. "
    "Use vivid storytelling and refer to characters naturally.\n\n"
    f"The following NPCs are present:\n{npc_context}"
)

agent = Agent(
    name="DungeonMaster",
    model="openai.gpt-4o",
    instructions=instructions,
    tools=[roll_dice, sample_npcs],
    players=st.session_state.players
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
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"❌ Could not read PDF: {e}")
            text = ""
    elif uploaded_file.type == "text/plain":
        text = uploaded_file.read().decode("utf-8")
    else:
        st.warning("Unsupported file type")
        text = ""

    st.session_state["script_text"] = text

    st.subheader("📜 Script Preview")
    st.text_area("Script contents:", text, height=400)

else:
    st.info("Please upload a PDF or TXT file to begin setting up your world.")

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
