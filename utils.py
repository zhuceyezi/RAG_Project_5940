import base64
import inspect
import json
import re
from pydantic import BaseModel, Field
from typing import Dict
import random
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import tempfile

class Player:
    def __init__(self, name, max_hp, hp):
        self.name = name
        self.max_hp = max_hp
        self.hp = hp

    @classmethod
    def from_dict(cls, d):
        return cls(
            name=d["name"],
            max_hp=d.get("max_hp", 20),
            hp=d.get("hp", d.get("max_hp", 20))
        )

    def take_damage(self, amount):
        self.hp = max(0, self.hp - amount)

    def heal(self, amount):
        self.hp = min(self.max_hp, self.hp + amount)

    def status(self):
        return f"{self.name}: {self.hp}/{self.max_hp} HP"

class NPCPlayer(Player):
    def __init__(self, name, max_hp, hp, race, char_class, personality, combat_role, quirks, voice, trait):
        super().__init__(name, max_hp, hp)
        self.race = race
        self.char_class = char_class
        self.personality = personality
        self.combat_role = combat_role
        self.quirks = quirks
        self.voice = voice
        self.trait = trait

    @classmethod
    def from_dict(cls, d):
        return cls(
            name=d["name"],
            max_hp=d.get("max_hp", 20),
            hp=d.get("hp", d.get("max_hp", 20)),
            race=d.get("race", "Unknown"),
            char_class=d.get("class", "Commoner"),
            personality=d.get("personality", "nondescript"),
            combat_role=d.get("combat_role", "None"),
            quirks=d.get("quirks", ""),
            voice=d.get("voice", ""),
            trait=d.get("trait", "")
        )

    def to_dict(self):
        return {
            "name": self.name,
            "max_hp": self.max_hp,
            "hp": self.hp,
            "race": self.race,
            "class": self.char_class,
            "personality": self.personality,
            "combat_role": self.combat_role,
            "quirks": self.quirks,
            "voice": self.voice,
            "trait": self.trait
        }


def load_npc_pool(path="npcs.json") -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def save_scene_graph_image(scene_data, current_location=None):
    import matplotlib.pyplot as plt
    import networkx as nx
    import tempfile
    from matplotlib.patches import Patch

    G = nx.DiGraph()
    labels = {}
    node_colors = []

    for scene in scene_data:
        title = scene["title"]
        G.add_node(title)
        labels[title] = f'{scene["scene_number"]}. {title}'
        for conn in scene.get("connections", []):
            G.add_edge(title, conn)

    pos = nx.spring_layout(G, seed=42)

    for node in G.nodes:
        if current_location and node.lower() == current_location.lower():
            node_colors.append("red")
        else:
            node_colors.append("lightblue")

    plt.figure(figsize=(6, 6))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2200)
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=20, edge_color="gray")
    nx.draw_networkx_labels(G, pos, labels, font_size=10)

    # Add legend below the graph
    legend_elements = [
        Patch(facecolor='red', edgecolor='gray', label='Current Location'),
        Patch(facecolor='lightblue', edgecolor='gray', label='Other Scenes')
    ]
    plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2)

    tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.axis("off")
    plt.tight_layout(pad=2)
    plt.savefig(tmpfile.name, format="png", bbox_inches="tight")
    plt.close()
    return tmpfile.name



def sample_npcs(n: int) -> dict:
    """Randomly sample N predefined NPCs and return them as NPCPlayer objects."""

    npc_pool = load_npc_pool()
    print(f"There are {len(npc_pool)} NPCs in the pool")
    sampled = random.sample(npc_pool, min(n, len(npc_pool)))
    print(f"Sampled {len(sampled)} NPCs")

    npc_players = [NPCPlayer.from_dict(npc) for npc in sampled]
    return {"npcs": npc_players}



def function_to_schema(func) -> dict:
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": (func.__doc__ or "").strip(),
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }


def run_full_turn(client, agent, messages):
    num_init_messages = len(messages)
    messages = messages.copy()
    tool_outputs = []

    while True:
        # Turn Python functions into OpenAI tool schemas
        tool_schemas = [function_to_schema(tool) for tool in agent.tools]
        tools_map = {tool.__name__: tool for tool in agent.tools}

        # === 1. Inject real-time game state into system context
        game_state_summary = build_game_state_summary(
            agent.players,
            st.session_state.get("npcs", [])
        )
        system_messages = [
            {"role": "system", "content": agent.instructions},
            {"role": "system", "content": game_state_summary}
        ]

        # === 2. Get assistant response
        response = client.chat.completions.create(
            model=agent.model,
            messages=system_messages + messages,
            tools=tool_schemas or None,
        )
        message = response.choices[0].message
        messages.append(message)

        if message.content:  # Assistant textual output
            print("Assistant:", message.content)

        if not message.tool_calls:  # No tool calls? Break the loop.
            break

        # === 3. Handle tool calls ===
        for tool_call in message.tool_calls:
            raw_result = execute_tool_call(tool_call, tools_map)

            # Ensure result is a JSON string
            if not isinstance(raw_result, str):
                raw_result = json.dumps(raw_result)

            # Add to messages for follow-up by model
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": raw_result
            })

            # Optional: store tool call result for inspection
            try:
                parsed_result = json.loads(raw_result)
                tool_outputs.append({
                    "tool": tool_call.function.name,
                    "args": json.loads(tool_call.function.arguments),
                    "result": parsed_result
                })
            except Exception:
                tool_outputs.append({
                    "tool": tool_call.function.name,
                    "args": json.loads(tool_call.function.arguments),
                    "result": raw_result
                })

    return messages[num_init_messages:], tool_outputs





def execute_tool_call(tool_call, tools_map):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    print(f"Assistant is calling: {name}({args})")

    result = tools_map[name](**args)

    # ✅ Safely handle NPCPlayer or other custom objects
    def safe_serialize(obj):
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        else:
            return str(obj)
    return json.dumps(result, default=safe_serialize) if isinstance(result, dict) else str(result)

def get_role(m):
    return m.get("role") if isinstance(m, dict) else m.role

def get_content(m):
    return m.get("content") if isinstance(m, dict) else m.content

def roll_dice(num_dice: int, sides: int) -> dict:
    """Rolls dice and returns structured result."""
    rolls = [random.randint(1, sides) for _ in range(num_dice)]
    total = sum(rolls)
    return {
        "rolls": rolls,
        "total": total,
        "text": f"Rolled {num_dice}d{sides}: {rolls} → total: {total}"
    }

class Agent(BaseModel):
    name: str = "Agent"
    model: str = "openai.gpt-4o-mini"
    instructions: str = "You are a helpful Agent."
    tools: list = []
    players: Dict[str, 'Player'] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

def build_npc_summary(npcs: list[NPCPlayer]) -> str:
    return "\n".join(
        f"{npc.name} is a {npc.race} {npc.char_class} who is {npc.personality.lower()}."
        f" They currently have {npc.hp}/{npc.max_hp} HP. Quirks: {npc.quirks} | Trait: {npc.trait}"
        for npc in npcs
    )

def build_game_state_summary(players: dict, npcs: list) -> str:
    lines = ["Here is the current status of all characters:"]

    for p in players.values():
        lines.append(f"{p.name}: {p.hp}/{p.max_hp} HP")

    if npcs:
        for npc in npcs:
            if hasattr(npc, "hp"):  # support both Player and NPCPlayer
                lines.append(f"{npc.name}: {npc.hp}/{npc.max_hp} HP")

    return "\n".join(lines)

def render_sidebar(players):
    with st.sidebar:
        st.header("🩸 Party HP")
        for player in players.values():
            bar = int(player.hp / player.max_hp * 20)
            st.markdown(f"**{player.name}**")
            st.markdown(f"{'🟥' * bar}{'⬛' * (20 - bar)}")
            st.caption(f"{player.hp}/{player.max_hp} HP")

def render_npcs(npcs: list):
    with st.sidebar:
        st.header("🧍 NPCs")

        if not npcs:
            st.info("No NPCs generated yet.")
            return

        for npc in npcs:
            bar = int(npc.hp / npc.max_hp * 20)
            st.markdown(f"{npc.name} ({npc.char_class} - {npc.race})")
            st.markdown(f"🩸 {'🟥' * bar}{'⬛' * (20 - bar)} ({npc.hp}/{npc.max_hp})")
            if isinstance(npc, NPCPlayer):
                with st.expander("Details"):
                    st.markdown(f"- **Personality:** {npc.personality}")
                    st.markdown(f"- **Combat Role:** {npc.combat_role}")
                    st.markdown(f"- **Quirks:** {npc.quirks}")
                    st.markdown(f"- **Voice Style:** {npc.voice}")
                    st.markdown(f"- **Unique Trait:** {npc.trait}")
            else:  # fallback
                with st.expander(f"{npc.name}"):
                    st.markdown(f"🩸 {'🟥' * bar}{'⬛' * (20 - bar)} ({npc.hp}/{npc.max_hp})")
                    
def render_scene_graph_bottom_right(encoded_image: str):
    st.markdown(
        f"""
        <div style='
            position: fixed;
            bottom: 0;
            right: 0;
            width: 20vw;
            height: 50vh;
            background: white;
            border: 1px solid #ccc;
            box-shadow: -2px -2px 10px rgba(0,0,0,0.1);
            z-index: 10000;
            padding: 12px;
            overflow: auto;
        '>
            <div style='font-weight: bold; margin-bottom: 8px; font-size: 16px;'>🗺️ Scene Map</div>
            <img src='data:image/png;base64,{encoded_image}' style='width: 100%; height: auto;' />
        </div>
        """,
        unsafe_allow_html=True
    )

    
def render_scene_graph_right_panel(encoded_image: str):
    st.markdown(
        f"""
        <div style='
            position: fixed;
            top: 0;
            right: 0;
            width: 20vw;
            height: 100vh;
            background: white;
            border-left: 2px solid #ccc;
            box-shadow: -4px 0 10px rgba(0,0,0,0.1);
            z-index: 10000;
            padding: 16px;
            overflow-y: auto;
        '>
            <div style='font-weight: bold; margin-bottom: 12px; font-size: 18px;'>🗺️ Scene Map</div>
            <img src='data:image/png;base64,{encoded_image}' style='width: 100%; height: auto;' />
        </div>
        """,
        unsafe_allow_html=True
    )

# @tool
def remove_npc(name: str):
    """Remove a player or NPC by name. Used when a character dies or should be removed."""
    if "npcs" not in st.session_state:
        return

    original_count = len(st.session_state["npcs"])
    st.session_state["npcs"] = [
        npc for npc in st.session_state["npcs"] if npc.name.lower() != name.lower()
    ]

    if len(st.session_state["npcs"]) < original_count:
        st.info(f"💀 NPC '{name}' has fallen and was removed from the sidebar.")


def apply_hp_effects(client, assistant_messages, players, model="openai.gpt-4o"):
    """Uses GPT to extract and apply multiple HP effects from assistant narration via run_full_turn."""
    # Combine all assistant text into one clean block
    assistant_text = "\n".join(
        m.content if hasattr(m, "content") else m["content"]
        for m in assistant_messages
        if (
            (getattr(m, "role", None) == "assistant" or (isinstance(m, dict) and m.get("role") == "assistant")) and
            (getattr(m, "content", None) is not None or (isinstance(m, dict) and m.get("content") is not None))
        )
    )

    all_names = list(players.keys())

    if "npcs" in st.session_state:
        all_names += [npc.name for npc in st.session_state["npcs"]]

    instructions = (f"...Only consider these valid targets: {all_names}")

    # Build the mini extractor agent
    extractor_agent = Agent(
        name="EffectExtractor",
        model=model,
        instructions=(
            "You are a game engine. Given a D&D narration, extract all HP effects.\n"
            "Return ONLY a JSON array like:\n"
            "[{\"target\": \"Frodo\", \"kind\": \"damage\", \"amount\": 3}, "
            "{\"target\": \"Frodo\", \"kind\": \"heal\", \"amount\": 4}]\n"
            "Return an empty list [] if no HP effects occurred.\n"
            f"Valid player names: {all_names}."
        ),
        tools=[],
        players={}
    )

    # Use run_full_turn with the extractor agent
    try:
        result_messages, _ = run_full_turn(
            client,
            extractor_agent,
            [{"role": "user", "content": assistant_text}]
        )
    except Exception as e:
        st.warning(f"❌ Failed to query GPT for HP effect: {e}")
        return

    # Extract GPT response
    try:
        response_msg = result_messages[-1]
        content = response_msg.content if hasattr(response_msg, "content") else response_msg.get("content", "")
        st.markdown(f"🔎 GPT returned:\n```json\n{content.strip()}\n```")
        effects = json.loads(content)
    except Exception as e:
        st.warning(f"❌ GPT could not parse HP effects: {e}")
        return

    if not effects:
        st.info("ℹ️ No HP effects detected.")
        return

    # Apply each effect
    for effect in effects:
        player = players.get(effect["target"])

        # Check for NPCs if not a player
        if not player and "npcs" in st.session_state:
            for npc in st.session_state["npcs"]:
                if npc.name.lower() == effect["target"].lower():
                    player = npc
                    break

        if not player:
            st.warning(f"❓ Character '{effect['target']}' not found among players or NPCs.")
            continue

        amount = effect["amount"]
        kind = effect["kind"]

        raw_amount = effect["amount"]
        kind = effect["kind"]

        # Handle "full", "half", or numeric strings
        # Normalize amount in case it's "full", "half", etc.
        raw_amount = effect["amount"]
        if isinstance(raw_amount, str):
            if raw_amount.lower() == "full":
                amount = player.max_hp - player.hp
            elif raw_amount.lower() == "half":
                amount = int(player.max_hp / 2)
            else:
                try:
                    amount = int(raw_amount)
                except ValueError:
                    st.warning(f"⚠️ Invalid amount: '{raw_amount}' for {player.name}")
                    continue
        else:
            amount = int(raw_amount)

        # Apply effect
        if kind == "heal":
            player.heal(amount)
            st.success(f"🟢 {player.name} heals {amount} → {player.status()}")
        elif kind == "damage":
            player.take_damage(amount)
            st.error(f"🔴 {player.name} takes {amount} damage → {player.status()}")
            # ✅ Remove if it's an NPC and HP hits zero
            if player.hp <= 0 and "npcs" in st.session_state and player in st.session_state["npcs"]:
                remove_npc(player.name)
        else:
            st.warning(f"⚠️ Unknown effect type: {kind}")


def extract_hp_effect_from_text(client, assistant_text: str, player_names: list, model="openai.gpt-4o") -> dict | None:
    """Use GPT to extract who took damage/healing, how much, and the type."""
    import streamlit as st
    prompt = (
        f"You are a game engine. Given this text:\n\n"
        f"\"{assistant_text}\"\n\n"
        f"Return a JSON object like:\n"
        f"{{\"target\": \"Frodo\", \"kind\": \"heal\", \"amount\": 8}}\n"
        f"or return null if no HP effect occurred.\n"
        f"Only consider these valid targets: {player_names}."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You extract structured HP effects from game narration."},
            {"role": "user", "content": prompt}
        ]
    )

    try:
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        st.warning(f"❌ GPT could not parse HP effect: {e}")
        return None

# @TOOL
def move_to_scene(scene_name):
    """Set the current scene to a known scene by title. Use this whenever players move to a place."""
    scene_data = st.session_state.get("scene_list")
    st.session_state["current_scene"] = scene_name

    if not scene_data:
        return

    graph_path = save_scene_graph_image(scene_data, current_location=scene_name)
    with open(graph_path, "rb") as f:
        encoded_img = base64.b64encode(f.read()).decode()

    st.session_state["scene_graph_img"] = encoded_img

# @tool
def add_npc(
    name: str,
    max_hp: int = 10,
    hp: int = None,
    race: str = "Unknown",
    class_: str = "Unknown",
    personality: str = "Unknown",
    combat_role: str = "None",
    quirks: str = "",
    voice: str = "",
    trait: str = "",
):
    """Add an NPC or enemy to the sidebar. Can be full character or simple enemy."""
    if hp is None:
        hp = max_hp

    npc = NPCPlayer(
        name=name,
        race=race,
        char_class=class_,
        max_hp=max_hp,
        hp=hp,
        personality=personality,
        combat_role=combat_role,
        quirks=quirks,
        voice=voice,
        trait=trait
    )

    if "npcs" not in st.session_state:
        st.session_state["npcs"] = []

    st.session_state["npcs"].append(npc)
    return f"✅ NPC or enemy '{name}' has been added with {hp}/{max_hp} HP."
