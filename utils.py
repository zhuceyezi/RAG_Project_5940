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
    def __init__(self, name, max_hp, hp, stats=None):
        self.name = name
        self.max_hp = max_hp
        self.hp = hp
        self.stats = stats or {
            "STR": 10,
            "DEX": 10,
            "CON": 10,
            "INT": 10,
            "WIS": 10,
            "CHA": 10
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            name=d["name"],
            max_hp=d.get("max_hp", 20),
            hp=d.get("hp", d.get("max_hp", 20)),
            stats=d.get("stats")
        )

    def status(self):
        return f"{self.name}: {self.hp}/{self.max_hp} HP"

    def take_damage(self, amount):
        self.hp = max(0, self.hp - amount)

    def heal(self, amount):
        self.hp = min(self.max_hp, self.hp + amount)


class NPCPlayer(Player):
    def __init__(self, name, max_hp, hp, race, char_class, personality, combat_role, quirks, voice, trait, stats=None):
        super().__init__(name, max_hp, hp, stats)
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
            trait=d.get("trait", ""),
            stats=d.get("stats")
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
            "trait": self.trait,
            "stats": self.stats
        }


def load_npc_pool(path="npcs.json") -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def save_scene_graph_image(scene_data, current_location=None):
    import matplotlib.pyplot as plt
    import networkx as nx
    import tempfile
    from matplotlib.patches import Patch

    # Set a D&D themed style for the graph
    plt.style.use('dark_background')
    
    G = nx.DiGraph()
    labels = {}
    node_colors = []
    edge_colors = []

    # Create nodes and edges
    for scene in scene_data:
        title = scene["title"]
        G.add_node(title)
        labels[title] = f'{scene["scene_number"]}. {title}'
        for conn in scene.get("connections", []):
            G.add_edge(title, conn)
            edge_colors.append('#d4af37')  # Gold color for edges

    # Use a more organic layout
    pos = nx.spring_layout(G, k=0.5, seed=42)  # k controls the spacing

    # Set node colors - current location is highlighted
    for node in G.nodes:
        if current_location and node.lower() == current_location.lower():
            node_colors.append("#ff5733")  # Bright orange-red for current location
        else:
            node_colors.append("#8b4513")  # Saddle brown for other locations

    # Create the figure with a parchment-like background
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#f5e8c9')
    ax.set_facecolor('#f5e8c9')  # Parchment color

    # Draw the graph with enhanced styling
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2500, 
                          edgecolors='black', linewidths=2, alpha=0.9)
    
    nx.draw_networkx_edges(G, pos, arrowstyle="-|>", arrowsize=20, 
                          edge_color=edge_colors, width=2, 
                          connectionstyle="arc3,rad=0.1")
    
    nx.draw_networkx_labels(G, pos, labels, font_size=10, 
                           font_family='serif', font_weight='bold',
                           font_color='white')

    # Add a decorative border to make it look like an old map
    border_width = 0.05
    ax.set_xlim(min(xx for xx, yy in pos.values()) - border_width, 
               max(xx for xx, yy in pos.values()) + border_width)
    ax.set_ylim(min(yy for xx, yy in pos.values()) - border_width, 
               max(yy for xx, yy in pos.values()) + border_width)
    
    # Add a title with fantasy font
    plt.title("Adventure Map", fontsize=20, fontweight='bold', 
             fontfamily='serif', color='#8b4513')

    # Add a decorative compass rose
    compass_pos = (min(xx for xx, yy in pos.values()) + 0.05, 
                  min(yy for xx, yy in pos.values()) + 0.05)
    compass_size = 0.05
    
    # Add legend with a parchment-like background
    legend_elements = [
        Patch(facecolor='#ff5733', edgecolor='black', label='Current Location'),
        Patch(facecolor='#8b4513', edgecolor='black', label='Other Locations')
    ]
    legend = plt.legend(handles=legend_elements, loc='lower right', 
                      frameon=True, facecolor='#f5e8c9', 
                      edgecolor='#8b4513', fontsize=10)
    legend.get_frame().set_linewidth(2)

    # Turn off axis
    plt.axis("off")
    plt.tight_layout()

    # Save the figure
    tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(tmpfile.name, format="png", dpi=300, bbox_inches="tight", 
              facecolor='#f5e8c9', edgecolor='#8b4513')
    plt.close()
    return tmpfile.name


def sample_npcs(n: int) -> dict:
    """Randomly sample N predefined NPCs and return them as NPCPlayer objects."""

    npc_pool = load_npc_pool()
    print(f"There are {len(npc_pool)} NPCs in the pool")
    sampled = random.sample(npc_pool, min(n, len(npc_pool)))
    print(f"Sampled {len(sampled)} NPCs")

    npc_players = [NPCPlayer.from_dict(npc) for npc in sampled]
    # Add random HP values to make it more gamelike
    for npc in npc_players:
        class_hp_base = {
            "Fighter": (25, 35),
            "Rogue": (15, 25),
            "Wizard": (10, 20),
            "Cleric": (18, 28),
            "Bard": (15, 22),
            "Paladin": (22, 32),
            "Warlock": (12, 22),
            "Ranger": (18, 28),
            "Artificer": (15, 25),
            "Healer": (18, 28),
            "Scholar": (10, 18)
        }
        
        base_range = class_hp_base.get(npc.char_class, (15, 25))
        npc.max_hp = random.randint(*base_range)
        npc.hp = npc.max_hp
    
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
        line = f"{p.name}: {p.hp}/{p.max_hp} HP"
        if hasattr(p, "stats") and isinstance(p.stats, dict):
            stat_str = ", ".join(f"{k}: {v}" for k, v in p.stats.items())
            line += f" | Stats: {stat_str}"
        lines.append(line)

    if npcs:
        for npc in npcs:
            if hasattr(npc, "hp"):
                line = f"{npc.name}: {npc.hp}/{npc.max_hp} HP"
                if hasattr(npc, "stats") and isinstance(npc.stats, dict):
                    stat_str = ", ".join(f"{k}: {v}" for k, v in npc.stats.items())
                    line += f" | Stats: {stat_str}"
                lines.append(line)

    return "\n".join(lines)


def render_sidebar(players):
    with st.sidebar:
        st.markdown("""
        <h1 style="text-align: center; font-family: 'Cinzel', serif; color: #fdf3d0; 
                   text-shadow: 0 0 5px #6c3e00; margin-bottom: 20px;">
            Adventure Tracker
        </h1>
        """, unsafe_allow_html=True)
        
        if "show_map" not in st.session_state:
            st.session_state["show_map"] = True
        st.sidebar.checkbox("🗺️ Show Adventure Map", key="show_map")
        
        st.markdown("""
        <h2 style="font-family: 'Cinzel', serif; color: #fdf3d0; border-bottom: 2px solid #fdf3d0; 
                  padding-bottom: 10px; margin-top: 30px;">
            🛡️ Party Members
        </h2>
        """, unsafe_allow_html=True)
        
        for player in players.values():
            # Calculate HP percentage
            hp_percent = int(player.hp / player.max_hp * 100)
            
            # Create a custom styled player card
            st.markdown(f"""
            <div style="background-color: rgba(40, 24, 8, 0.7); padding: 10px; border-radius: 5px; 
                       margin: 10px 0; border: 1px solid #fdf3d0;">
                <h3 style="font-family: 'Cinzel', serif; margin: 0; color: #fdf3d0; 
                          text-shadow: 0 0 3px #6c3e00;">{player.name}</h3>
                
                <div class="hp-bar-container">
                    <div class="hp-bar" style="width: {hp_percent}%;"></div>
                </div>
                <p style="text-align: center; margin: 5px 0; color: #fdf3d0;">
                    {player.hp}/{player.max_hp} HP
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Toggleable stat view with styled expander
            if hasattr(player, "stats"):
                with st.expander("📊 Character Stats", expanded=False):
                    # Create a styled stat block
                    stat_html = "<div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 5px;'>"
                    
                    for stat, value in player.stats.items():
                        # Calculate the modifier
                        modifier = (value - 10) // 2
                        mod_sign = "+" if modifier >= 0 else ""
                        
                        stat_html += f"""
                        <div style='background-color: rgba(60, 34, 18, 0.7); padding: 5px; 
                                   border-radius: 5px; text-align: center;'>
                            <div style='font-weight: bold; color: #fdf3d0;'>{stat}</div>
                            <div style='font-size: 1.2em; color: #fdf3d0;'>{value}</div>
                            <div style='color: {"#4dbd74" if modifier >= 0 else "#f86c6b"};'>
                                {mod_sign}{modifier}
                            </div>
                        </div>
                        """
                    
                    stat_html += "</div>"
                    st.markdown(stat_html, unsafe_allow_html=True)


def render_npcs(npcs: list):
    with st.sidebar:
        st.markdown("""
        <h2 style="font-family: 'Cinzel', serif; color: #fdf3d0; border-bottom: 2px solid #fdf3d0; 
                  padding-bottom: 10px; margin-top: 30px;">
            🧙 NPCs & Enemies
        </h2>
        """, unsafe_allow_html=True)

        if not npcs:
            st.info("No NPCs have joined your adventure yet.")
            return

        for npc in npcs:
            # Calculate HP percentage
            hp_percent = int(npc.hp / npc.max_hp * 100)
            
            # Different background colors based on combat role
            bg_colors = {
                "Support": "rgba(75, 119, 190, 0.7)",      # Blue for support
                "Healer": "rgba(82, 157, 82, 0.7)",        # Green for healers
                "Tank": "rgba(162, 118, 74, 0.7)",         # Brown for tanks
                "Damage": "rgba(190, 75, 75, 0.7)",        # Red for damage
                "Stealth": "rgba(75, 75, 75, 0.7)",        # Dark gray for stealth
                "Magic": "rgba(147, 75, 190, 0.7)"         # Purple for magic users
            }
            
            # Determine background color based on role
            role_keywords = ["Support", "Healer", "Tank", "Damage", "DPS", "Stealth", "Magic"]
            bg_color = "rgba(40, 24, 8, 0.7)"  # Default color
            
            for keyword in role_keywords:
                if keyword.lower() in npc.combat_role.lower():
                    bg_color = bg_colors.get(keyword, bg_color)
                    break
            
            st.markdown(f"""
            <div style="background-color: {bg_color}; padding: 10px; border-radius: 5px; 
                       margin: 10px 0; border: 1px solid #fdf3d0;">
                <h3 style="font-family: 'Cinzel', serif; margin: 0; color: #fdf3d0; 
                          text-shadow: 0 0 3px #6c3e00;">{npc.name}</h3>
                <p style="color: #fdf3d0; margin: 2px 0; font-style: italic;">
                    {npc.race} {npc.char_class}
                </p>
                
                <div class="hp-bar-container">
                    <div class="hp-bar" style="width: {hp_percent}%;"></div>
                </div>
                <p style="text-align: center; margin: 5px 0; color: #fdf3d0;">
                    {npc.hp}/{npc.max_hp} HP
                </p>
            </div>
            """, unsafe_allow_html=True)

            if isinstance(npc, NPCPlayer):
                with st.expander("Character Details", expanded=False):
                    if hasattr(npc, "stats") and isinstance(npc.stats, dict):
                        # Create a styled stat block
                        stat_html = "<div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 5px; margin-bottom: 10px;'>"
                        
                        for stat, value in npc.stats.items():
                            # Calculate the modifier
                            modifier = (value - 10) // 2
                            mod_sign = "+" if modifier >= 0 else ""
                            
                            stat_html += f"""
                            <div style='background-color: rgba(60, 34, 18, 0.7); padding: 5px; 
                                       border-radius: 5px; text-align: center;'>
                                <div style='font-weight: bold; color: #fdf3d0;'>{stat}</div>
                                <div style='font-size: 1.2em; color: #fdf3d0;'>{value}</div>
                                <div style='color: {"#4dbd74" if modifier >= 0 else "#f86c6b"};'>
                                    {mod_sign}{modifier}
                                </div>
                            </div>
                            """
                        
                        stat_html += "</div>"
                        st.markdown(stat_html, unsafe_allow_html=True)
                    
                    st.markdown(f"<div style='color: #fdf3d0;'><strong>Personality:</strong> {npc.personality}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='color: #fdf3d0;'><strong>Combat Role:</strong> {npc.combat_role}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='color: #fdf3d0;'><strong>Quirks:</strong> {npc.quirks}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='color: #fdf3d0;'><strong>Voice:</strong> {npc.voice}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='color: #fdf3d0;'><strong>Trait:</strong> {npc.trait}</div>", unsafe_allow_html=True)
            else:  # fallback
                with st.expander(f"{npc.name}"):
                    # Calculate HP bar without using 'bar' variable
                    hp_display = f"🩸 {'🟥' * int(npc.hp / npc.max_hp * 20)}{'⬛' * (20 - int(npc.hp / npc.max_hp * 20))} ({npc.hp}/{npc.max_hp})"
                    st.markdown(hp_display)


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
        st.info(f"💀 NPC '{name}' has fallen and was removed from the party.")


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
        effects = json.loads(content)
    except Exception as e:
        st.warning(f"❌ GPT could not parse HP effects: {e}")
        return

    if not effects:
        return

    # Apply each effect with styled notifications
    for effect in effects:
        player = players.get(effect["target"])

        # Check for NPCs if not a player
        if not player and "npcs" in st.session_state:
            for npc in st.session_state["npcs"]:
                if npc.name.lower() == effect["target"].lower():
                    player = npc
                    break

        if not player:
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
                    continue
        else:
            amount = int(raw_amount)

        # Apply effect with fantasy-styled notifications
        if kind == "heal":
            player.heal(amount)
            st.markdown(f"""
            <div style="background-color: rgba(82, 157, 82, 0.7); color: white; padding: 10px; 
                       border-radius: 5px; margin: 10px 0; border: 1px solid #4dbd74; 
                       font-family: 'Fondamento', cursive;">
                ✨ {player.name} is healed for {amount} points of vitality! ({player.hp}/{player.max_hp} HP)
            </div>
            """, unsafe_allow_html=True)
        elif kind == "damage":
            player.take_damage(amount)
            st.markdown(f"""
            <div style="background-color: rgba(190, 75, 75, 0.7); color: white; padding: 10px; 
                       border-radius: 5px; margin: 10px 0; border: 1px solid #f86c6b; 
                       font-family: 'Fondamento', cursive;">
                🗡️ {player.name} takes {amount} points of damage! ({player.hp}/{player.max_hp} HP)
            </div>
            """, unsafe_allow_html=True)
            # Remove if it's an NPC and HP hits zero
            if player.hp <= 0 and "npcs" in st.session_state and player in st.session_state["npcs"]:
                remove_npc(player.name)
                st.markdown(f"""
                <div style="background-color: rgba(0, 0, 0, 0.7); color: white; padding: 10px; 
                           border-radius: 5px; margin: 10px 0; border: 1px solid #6a0000; 
                           font-family: 'Fondamento', cursive; text-align: center;">
                    ☠️ {player.name} has fallen in battle!
                </div>
                """, unsafe_allow_html=True)


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
    """Set the current scene to a known scene by title. Use this whenever players move to a place.
    Make sure only to use the available scenes provided."""
    scene_data = st.session_state.get("scene_list")
    st.session_state["current_scene"] = scene_name

    if not scene_data:
        return "No scene data available"

    # Create a new enhanced scene map
    graph_path = save_scene_graph_image(scene_data, current_location=scene_name)
    with open(graph_path, "rb") as f:
        encoded_img = base64.b64encode(f.read()).decode()

    # Update the map
    st.session_state["scene_graph_img"] = encoded_img
    
    # Get description of the current scene for a more immersive notification
    scene_desc = ""
    for scene in scene_data:
        if scene["title"].lower() == scene_name.lower():
            scene_desc = scene.get("description", "")
            break
    
    return f"The party has moved to {scene_name}. {scene_desc}"


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
    return f"✅ {name} the {race} {class_} has joined the adventure with {hp}/{max_hp} HP."