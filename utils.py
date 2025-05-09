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
import io
import os
import requests
import math

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
    def __init__(self, name, max_hp, hp, race, char_class, personality, combat_role, quirks, voice, trait, alignment: str = "neutral",stats=None):
        super().__init__(name, max_hp, hp, stats)
        self.race = race
        self.char_class = char_class
        self.personality = personality
        self.combat_role = combat_role
        self.quirks = quirks
        self.voice = voice
        self.trait = trait
        self.alignment = alignment

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
            stats=d.get("stats"),
            alignment=d.get("alignment", "neutral"),
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
            "alignment": self.alignment,
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

def save_scene_graph_with_map_background(scene_data, map_image_path, current_location=None, custom_font=None):
    """Generate a scene graph image with fantasy map as background."""
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import networkx as nx
    import tempfile
    from matplotlib.patches import Patch
    import numpy as np

    # Create graph
    G = nx.DiGraph()
    labels = {}
    node_colors = []

    # Add all nodes and connections
    for scene in scene_data:
        title = scene["title"]
        G.add_node(title)
        labels[title] = f'{scene["scene_number"]}. {title}'
        for conn in scene.get("connections", []):
            G.add_edge(title, conn)

    # Load the map image first to get its dimensions
    map_img = mpimg.imread(map_image_path)
    img_height, img_width = map_img.shape[0], map_img.shape[1]
    aspect_ratio = img_width / img_height

    # Create a figure with the same aspect ratio as the map
    fig = plt.figure(figsize=(18, 18/aspect_ratio))  # 12 * 1.5 = 18 for 1.5x larger
    ax = fig.add_subplot(111)

    # Display the map
    ax.imshow(map_img, extent=[0, 1, 0, 1])

    # Use a different layout that keeps nodes more central
    # Start with spring layout
    pos = nx.spring_layout(G, seed=42)

    # Adjust positions to ensure they stay within map boundaries with padding
    padding = 0.15  # Padding from the edges (15%)
    for node in pos:
        x, y = pos[node]
        # Constrain positions to be within the map boundaries with padding
        pos[node] = (
            max(padding, min(1-padding, x)),
            max(padding, min(1-padding, y))
        )

    # Set node colors based on current location
    for node in G.nodes:
        if current_location and node.lower() == current_location.lower():
            node_colors.append("red")
        else:
            node_colors.append("lightblue")

    # Draw semi-transparent nodes with thick black outline for visibility
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1200, alpha=0.8,
                         edgecolors='black', linewidths=2, ax=ax)

    # Draw edges with arrows
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=15, edge_color="black",
                         width=2, alpha=0.8, ax=ax)

    # Draw text with better visibility against any background
    # Create a text outline effect
    for node, (x, y) in pos.items():
        text = labels[node]
        # Draw text outline in black
        offsets = [(1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)]
        for dx, dy in offsets:
            ax.text(x + dx*0.005, y + dy*0.005, text,
                    horizontalalignment='center', verticalalignment='center',
                    color='black', fontsize=10, fontweight='bold',
                    family=custom_font if custom_font else 'sans-serif',
                    zorder=5)

    # Then draw white text on top of the outline
    for node, (x, y) in pos.items():
        text = labels[node]
        ax.text(x, y, text,
                horizontalalignment='center', verticalalignment='center',
                color='white', fontsize=10, fontweight='bold',
                family=custom_font if custom_font else 'sans-serif',
                zorder=6)

    # Add legend
    legend_elements = [
        Patch(facecolor='red', edgecolor='black', label='Current Location'),
        Patch(facecolor='lightblue', edgecolor='black', label='Other Scenes')
    ]
    plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=2)

    # Remove axes to show only the map
    plt.axis('off')

    # Save the result
    tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.tight_layout(pad=0)
    plt.savefig(tmpfile.name, format="png", bbox_inches="tight", dpi=300)
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

        # convert message to dict if it's not already
        message_dict = {
            "role": message.role,
            "content": message.content
        }
        if hasattr(message, "tool_calls") and message.tool_calls:
            message_dict["tool_calls"] = message.tool_calls
        messages.append(message_dict)

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
    if isinstance(tool_call, dict):
        name = tool_call.get("function", {}).get("name")
        args = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
    else:
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
                line = f"{npc.name}: {npc.hp}/{npc.max_hp} HP  (alignment: {npc.alignment})"
                if hasattr(npc, "stats") and isinstance(npc.stats, dict):
                    stat_str = ", ".join(f"{k}: {v}" for k, v in npc.stats.items())
                    line += f" | Stats: {stat_str}"
                lines.append(line)

    # Add NPC summary
    if st.session_state.get("chat_log"):
        history = [m["content"] for m in st.session_state.chat_log if m["role"]!="tool"][-3:]
        lines.append("Recent events:")
        lines.extend(f"- {h}" for h in history)

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

        # Add this font selection code
        if "map_font" not in st.session_state:
            st.session_state["map_font"] = "serif"  # Default font

        # Font selection for node labels
        font_options = ["serif", "sans-serif", "monospace", "fantasy", "cursive"]
        selected_font = st.sidebar.selectbox(
            "Node Label Font",
            options=font_options,
            index=font_options.index(st.session_state["map_font"])
        )
        st.session_state["map_font"] = selected_font

        st.header("🩸 Party HP")
        for player in players.values():
            # Calculate HP percentage
            hp_percent = int(player.hp / player.max_hp * 100)

            # Create a custom styled player card
            st.markdown(
                f'<div style="background-color: rgba(40, 24, 8, 0.7); padding: 10px; border-radius: 5px; '
                'margin: 10px 0; border: 1px solid #fdf3d0;">'
                f'<h3 style="font-family: Cinzel, serif; margin: 0; color: #fdf3d0; '
                f'text-shadow: 0 0 3px #6c3e00;">{player.name}</h3>'
                f'<div class="hp-bar-container">'
                f'<div class="hp-bar" style="width: {hp_percent}%;"></div>'
                '</div>'
                f'<p style="text-align: center; margin: 5px 0; color: #fdf3d0;">{player.hp}/{player.max_hp} HP</p>'
                '</div>',
                unsafe_allow_html=True
            )

            # Toggleable stat view with styled expander
            if hasattr(player, "stats"):
                with st.expander("📊 Character Stats", expanded=False):
                    # Create a styled stat block
                    stat_html = "<div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 5px;'>"

                    for stat, value in player.stats.items():
                        # Calculate the modifier
                        modifier = (value - 10) // 2
                        mod_sign = "+" if modifier >= 0 else ""

                        stat_html += (
                            f"<div style='background-color: rgba(60, 34, 18, 0.7); padding: 5px; border-radius: 5px; text-align: center;'>"
                            f"<div style='font-weight: bold; color: #fdf3d0;'>{stat}</div>"
                            f"<div style='font-size: 1.2em; color: #fdf3d0;'>{value}</div>"
                            f"<div style='color: {'#4dbd74' if modifier >= 0 else '#f86c6b'};'> {mod_sign}{modifier}</div>"
                            f"</div>"
                        )

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
            bar = int(npc.hp / npc.max_hp * 20)
            # st.markdown(f"{npc.name} ({npc.char_class} - {npc.race})")
            st.caption(f"Alignment：{npc.alignment}")
            # st.markdown(f"🩸 {'🟥' * bar}{'⬛' * (20 - bar)} ({npc.hp}/{npc.max_hp})")
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

            st.markdown(
                f'<div style="background-color: {bg_color}; padding: 10px; border-radius: 5px; '
                'margin: 10px 0; border: 1px solid #fdf3d0;">'
                f'<h3 style="font-family: Cinzel, serif; margin: 0; color: #fdf3d0; '
                f'text-shadow: 0 0 3px #6c3e00;">{npc.name}</h3>'
                f'<p style="color: #fdf3d0; margin: 2px 0; font-style: italic;">{npc.race} {npc.char_class}</p>'
                f'<div class="hp-bar-container">'
                f'<div class="hp-bar" style="width: {hp_percent}%;"></div>'
                '</div>'
                f'<p style="text-align: center; margin: 5px 0; color: #fdf3d0;">{npc.hp}/{npc.max_hp} HP</p>'
                '</div>',
                unsafe_allow_html=True
            )

            if isinstance(npc, NPCPlayer):
                with st.expander("Character Details", expanded=False):
                    if hasattr(npc, "stats") and isinstance(npc.stats, dict):
                        # Create a styled stat block
                        stat_html = "<div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 5px; margin-bottom: 10px;'>"

                        for stat, value in npc.stats.items():
                            # Calculate the modifier
                            modifier = (value - 10) // 2
                            mod_sign = "+" if modifier >= 0 else ""

                            color = "#4dbd74" if modifier >= 0 else "#f86c6b"

                            stat_html += (
                                f"<div style='background-color: rgba(60, 34, 18, 0.7); padding: 5px; "
                                "border-radius: 5px; text-align: center;'>"
                                f"<div style='font-weight: bold; color: #fdf3d0;'>{stat}</div>"
                                f"<div style='font-size: 1.2em; color: #fdf3d0;'>{value}</div>"
                                f"<div style='color: {color};'>{mod_sign}{modifier}</div>"
                                "</div>"
                            )

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
    """Display the scene graph with perfect fitting for the map and nodes."""
    st.markdown(
        f"""
        <div style='
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 40vh; 
            height: 40vh;   
            background: white;
            border: 1px solid #ccc;
            border-radius: 8px;
            box-shadow: -2px -2px 10px rgba(0,0,0,0.1);
            z-index: 10000;
            padding: 12px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        '>
            <div style='
                font-weight: bold; 
                margin-bottom: 8px; 
                font-size: 16px;
                flex-shrink: 0;
            '>🗺️ Adventure Map</div>
            <div style='
                flex-grow: 1;
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: hidden;
                width: 100%;
                height: 100%;
            '>
                <img src='data:image/png;base64,{encoded_image}' 
                     style='
                        width: 100%;
                        height: 100%;
                        object-fit: cover;
                     ' 
                />
            </div>
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
        if isinstance(response_msg, dict):
            content = response_msg.get("content", "")
        else:
            content = response_msg.content if hasattr(response_msg, "content") else ""
        st.markdown(f"🔎 GPT returned:\n```json\n{content.strip()}\n```")
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

    # If we have a base map image, use it as background
    if "base_map_path" in st.session_state:
        graph_path = save_scene_graph_with_map_background(
            scene_data,
            st.session_state["base_map_path"],
            current_location=scene_name,
            custom_font=st.session_state.get("map_font", "serif")
        )
    else:
        # Otherwise regenerate the map
        if "map_description" in st.session_state:
            map_path = generate_map_image(client, st.session_state["map_description"], scene_data)
            if map_path:
                st.session_state["base_map_path"] = map_path
                graph_path = save_scene_graph_with_map_background(
                    scene_data,
                    map_path,
                    current_location=scene_name,
                    custom_font=st.session_state.get("map_font", "serif")
                )
            else:
                # Fallback to regular scene graph
                graph_path = save_scene_graph_image(scene_data, current_location=scene_name)
        else:
            # Fallback to regular scene graph
            graph_path = save_scene_graph_image(scene_data, current_location=scene_name)

    with open(graph_path, "rb") as f:
        encoded_img = base64.b64encode(f.read()).decode()

    st.session_state["scene_graph_img"] = encoded_img
    return "Move successful"


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
    alignment: str = ""
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
        trait=trait,
        alignment=alignment,
    )

    if "npcs" not in st.session_state:
        st.session_state["npcs"] = []

    st.session_state["npcs"].append(npc)
    return f"✅ NPC or enemy '{name}' has been added with {hp}/{max_hp} HP."


def generate_map_image(client, scene_data):
    """Generate a fantasy map image with truly correct positioning"""
    try:
        import networkx as nx

        # Create a graph to calculate node positions
        G = nx.DiGraph()

        # Add all nodes and connections
        for scene in scene_data:
            title = scene["title"]
            G.add_node(title)
            for conn in scene.get("connections", []):
                G.add_edge(title, conn)

        # Calculate node positions with a fixed seed
        raw_pos = nx.spring_layout(G, seed=42)

        # Normalize coordinates to [0,1] range
        normalized_pos = {}
        for node, (x, y) in raw_pos.items():
            normalized_pos[node] = ((x + 1) / 2, (y + 1) / 2)

        # Store for the overlay to use
        st.session_state["node_positions"] = normalized_pos

        # FINAL CORRECTED position mapping
        def coords_to_position(x, y):
            # Horizontal position (5 regions) - these are correct
            if x < 0.2:
                h_pos = "far left"
            elif x < 0.4:
                h_pos = "left"
            elif x < 0.6:
                h_pos = "center"
            elif x < 0.8:
                h_pos = "right"
            else:
                h_pos = "far right"

            # Vertical position (5 regions) - TRULY CORRECT VERSION NOW
            # Lower Y values (0.0-0.2) are at the TOP of the map
            # Higher Y values (0.8-1.0) are at the BOTTOM of the map
            if y < 0.2:
                v_pos = "lower"        # TOP of the map (low y value)
            elif y < 0.4:
                v_pos = "lower middle" # Between top and middle
            elif y < 0.6:
                v_pos = "middle"       # Middle of the map
            elif y < 0.8:
                v_pos = "upper middle" # Between middle and bottom
            else:
                v_pos = "upper"        # BOTTOM of the map (high y value)

            # Special case for true center
            if h_pos == "center" and v_pos == "middle":
                return "center"

            return f"{v_pos} {h_pos}"

        # Build scene descriptions with corrected positions
        scene_descriptions = []
        for scene in scene_data:
            title = scene["title"]
            location = scene.get("location", "")
            scene_num = scene.get("scene_number", 0)

            # Get normalized coordinates
            x, y = normalized_pos[title]

            # Convert to correct descriptive position
            position = coords_to_position(x, y)

            # Debug info
            print(f"Scene {scene_num}: {title} - positioned in the {position} area at ({x:.2f}, {y:.2f})")

            # Format the scene description
            scene_descriptions.append(
                f"Scene {scene_num}: {title} in {location} - located in the {position} area"
            )

        # Create the map generation prompt
        enhanced_prompt = (
            f"Create a single unified fantasy map showing these locations:\n\n"
            f"{'\n'.join(scene_descriptions)}\n\n"
            f"Map style: Top-down view of the entire adventure area on a single parchment sheet. "
            f"Hand-drawn with aged paper texture and ink illustrations. "
            f"Include fantasy elements like mountains, forests, paths, rivers, and landmarks appropriate to each location. "
            #f"This MUST be ONE FULL-FRAME continuous map showing ALL locations together in their specified positions. "
            #f"The map should extend all the way to the edges of the image without any borders or white space. "
            f"DO NOT include ANY text, labels, or writing. "
            f"Make it a square map with equal width and height."
        )

        # Print the exact prompt that will be sent to Imagen 3
        print("\n=== Prompt being sent to Imagen 3 ===")
        print(enhanced_prompt)
        print("=====================================\n")

        # Generate the image
        response = client.images.generate(
            model="google.imagen-3.0-generate",
            prompt=enhanced_prompt,
            n=1,
            size="1024x1024"
        )

        # Extract and save image data
        if hasattr(response, 'data') and len(response.data) > 0:
            if hasattr(response.data[0], "b64_json") and response.data[0].b64_json:
                image_data = base64.b64decode(response.data[0].b64_json)
            elif hasattr(response.data[0], "url") and response.data[0].url:
                image_url = response.data[0].url
                if image_url and image_url.startswith(('http://', 'https://')):
                    image_response = requests.get(image_url)
                    image_data = image_response.content
                else:
                    return None
            else:
                return None

            # Save the image
            img_path = "static/base_map.png"
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            with open(img_path, "wb") as f:
                f.write(image_data)

            return img_path

        return None
    except Exception as e:
        print(f"Error generating map: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def save_scene_graph_with_map_background(scene_data, map_image_path, current_location=None, custom_font=None):
    """Generate a scene graph image with fantasy map as background and larger nodes."""
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import networkx as nx
    import tempfile
    from matplotlib.patches import Patch
    import numpy as np

    # Create graph
    G = nx.DiGraph()
    labels = {}
    node_colors = []

    # Add all nodes and connections
    for scene in scene_data:
        title = scene["title"]
        G.add_node(title)
        labels[title] = f'{scene["scene_number"]}. {title}'
        for conn in scene.get("connections", []):
            G.add_edge(title, conn)

    # Load the map image first to get its dimensions
    map_img = mpimg.imread(map_image_path)
    img_height, img_width = map_img.shape[0], map_img.shape[1]
    aspect_ratio = img_width / img_height

    # Create a figure with the same aspect ratio as the map
    fig = plt.figure(figsize=(18, 18/aspect_ratio))  # 12 * 1.5 = 18 for 1.5x larger
    ax = fig.add_subplot(111)

    # Display the map with slightly reduced extent to allow for labels
    ax.imshow(map_img, extent=[0.05, 0.95, 0.05, 0.95])  # 5% padding on each side

    # Use positions from session state if available
    if "node_positions" in st.session_state:
        pos = st.session_state["node_positions"]
        print("Using stored node positions from NetworkX")
    else:
        # Fallback - recalculate exactly as in generate_map_image
        raw_pos = nx.spring_layout(G, seed=42)
        pos = {}
        for node, (x, y) in raw_pos.items():
            pos[node] = ((x + 1) / 2, (y + 1) / 2)
        print("Regenerated node positions with NetworkX")

    # Move node positions slightly inward if they're too close to the edges
    padding = 0.12  # 12% padding from edges
    for node in pos:
        x, y = pos[node]
        x = max(padding, min(1-padding, x))
        y = max(padding, min(1-padding, y))
        pos[node] = (x, y)

    # Set node colors based on current location
    for node in G.nodes:
        if current_location and node.lower() == current_location.lower():
            node_colors.append("red")
        else:
            node_colors.append("lightblue")

    # Draw nodes with thick black outline for visibility - LARGER
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2500, alpha=0.8,  # Increased from 1800 to 2500
                          edgecolors='black', linewidths=3, ax=ax)  # Thicker outlines

    # Draw edges with arrows - THICKER
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=20, edge_color="black",  # Larger arrows
                          width=3, alpha=0.8, ax=ax)  # Thicker lines

    # Draw text with full labels - LARGER
    for node, (x, y) in pos.items():
        text = labels[node]

        # Draw text with black background for better visibility
        ax.text(x, y, text,
                horizontalalignment='center', verticalalignment='center',
                color='white', fontsize=14,  # Increased from 12 to 14
                fontweight='bold',  # Make it bold for better visibility
                family=custom_font if custom_font else 'sans-serif',
                bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=3),  # More padding for larger text
                zorder=6)

    # Add legend - larger and more visible, moved to upper right
    legend_elements = [
        Patch(facecolor='red', edgecolor='black', label='Current Location'),
        Patch(facecolor='lightblue', edgecolor='black', label='Other Scenes')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12,  # Increased from 10 to 12, moved to upper right
              framealpha=0.8, edgecolor='black', bbox_to_anchor=(0.95, 0.95))  # Adjusted position

    # Remove axes
    plt.axis('off')

    # Save with proper tight layout to avoid white space
    tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(tmpfile.name, format="png", bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()

    return tmpfile.name

def save_scene_graph_image(scene_data, current_location=None, return_positions=False, custom_font=None):
    """Generate a scene graph image and optionally return node positions."""
    import matplotlib.pyplot as plt
    import networkx as nx
    import tempfile
    from matplotlib.patches import Patch

    G = nx.DiGraph()
    labels = {}
    node_colors = []

    # Add all nodes and connections
    for scene in scene_data:
        title = scene["title"]
        G.add_node(title)
        labels[title] = f'{scene["scene_number"]}. {title}'
        for conn in scene.get("connections", []):
            G.add_edge(title, conn)

    # Calculate the positions
    pos = nx.spring_layout(G, seed=42)

    # Set up the figure and plotting
    plt.figure(figsize=(6, 6))

    # Set node colors based on current location
    for node in G.nodes:
        if current_location and node.lower() == current_location.lower():
            node_colors.append("red")
        else:
            node_colors.append("lightblue")

    # Draw the network with custom font if specified
    font_props = {'font_size': 10}
    if custom_font:
        font_props['font_family'] = custom_font

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2200)
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=20, edge_color="gray")
    nx.draw_networkx_labels(G, pos, labels, **font_props)

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

    # Return both the image path and positions if requested
    if return_positions:
        return tmpfile.name, pos
    return tmpfile.name

# @tool
def set_npc_alignment(name: str, alignment: str):
    """
    Modify the alignment of an NPC.
    Set alignment to one of the three with information and chat so far
    alignment options: 'ally', 'enemy', 'neutral'.
    """
    if "npcs" not in st.session_state:
        return "No NPCs available in session."

    for npc in st.session_state["npcs"]:
        if npc.name.lower() == name.lower():
            npc.alignment = alignment
            return f"✅ NPC '{npc.name}' alignment set to '{alignment}'."
    return f"❌ NPC named '{name}' not found."

def load_combat_rules(path="combat_rules.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

_rules = load_combat_rules()
RACIAL_BONUS = _rules["racial_bonus"]
CLASS_BONUS = _rules["class_bonus"]
CLASS_ATTACK_MODE = _rules["class_attack_mode"]

# @tool
def calculate_attack(attacker_name: str,
                     defender_name: str,
                     mode: str = None,
                     weapon: dict = None) -> dict:
    """
    Calculate attack results between two entities.

    Args:
        attacker_name: Name of the attacking entity
        defender_name: Name of the defending entity
        mode: Attack mode ("melee", "ranged", or "spell") - if None, inferred from class
        weapon: Optional weapon data with keys like "damage" and "attack_bonus"

    Returns:
        Dictionary with attack results
    """
    # Helper to find entity
    def _get_entity(name: str):
        if "players" in st.session_state and name in st.session_state["players"]:
            return st.session_state["players"][name]
        for npc in st.session_state.get("npcs", []):
            if npc.name.lower() == name.lower():
                return npc
        raise ValueError(f"Entity '{name}' not found")

    # Get entities
    atk = _get_entity(attacker_name)
    defn = _get_entity(defender_name)

    # If mode not provided, get from class_attack_mode in JSON
    if mode is None:
        mode = CLASS_ATTACK_MODE.get(atk.char_class, "melee")

    # Roll d20 for attack
    roll = random.randint(1, 20)

    # Determine which stat to use based on attack mode
    stats = atk.stats or {}
    if mode == "melee":
        stat_key = "STR"
    elif mode == "ranged":
        stat_key = "DEX"
    else:  # spell
        stat_key = "INT"  # Could extend to use WIS/CHA for different caster classes

    # Calculate ability modifier: (stat-10)/2 rounded down
    base_mod = math.floor((stats.get(stat_key, 10) - 10) / 2)

    # Apply racial and class bonuses
    class_bonus = CLASS_BONUS.get(atk.char_class, {}).get(stat_key, 0)
    racial_bonus = RACIAL_BONUS.get(atk.race, {}).get(stat_key, 0)

    # Weapon bonus
    weapon_bonus = weapon.get("attack_bonus", 0) if weapon else 0

    # Total attack value
    total_attack = roll + base_mod + class_bonus + racial_bonus + weapon_bonus

    # Calculate AC (armor class) for defender
    def_stats = defn.stats or {}
    def_dex_mod = math.floor((def_stats.get("DEX", 10) - 10) / 2)
    ac = 10 + def_dex_mod  # Basic AC calculation

    # Determine hit/critical
    crit = (roll == 20)
    if roll <= 2:  # Critical miss on natural 1-2
        hit = False
    else:
        hit = crit or (total_attack >= ac)

    # Calculate damage
    damage = 0
    detail_parts = [f"Rolled {roll} + bonuses = {total_attack} vs AC {ac}"]

    if hit:
        # Default damage dice based on attack mode
        default_damage = {
            "melee": "1d6",
            "ranged": "1d4",
            "spell": "1d8"
        }

        # Get damage dice expression
        dmg_expr = weapon.get("damage", default_damage.get(mode, "1d4")) if weapon else default_damage.get(mode, "1d4")

        # Parse dice expression (e.g., "2d6")
        match = re.match(r"(\d+)d(\d+)", dmg_expr)
        if match:
            num_dice = int(match.group(1))
            sides = int(match.group(2))

            # Roll damage (doubled on crit)
            times = 2 if crit else 1
            dmg_roll = 0
            for _ in range(times):
                for _ in range(num_dice):
                    dmg_roll += random.randint(1, sides)

            # Apply damage bonus from stat
            dmg_bonus = base_mod + (weapon.get("damage_bonus", 0) if weapon else 0)
            damage = max(0, dmg_roll + dmg_bonus)

            # Build description
            detail_parts.append(f"{'Critical hit!' if crit else 'Hit!'} Damage: {dmg_roll}+{dmg_bonus} = {damage}")
        else:
            # Fallback if dice expression is invalid
            damage = random.randint(1, 4)
            detail_parts.append(f"Hit! Damage: {damage}")
    else:
        detail_parts.append("Missed!")

    # Return complete result
    return {
        "roll": roll,
        "total_attack": total_attack,
        "threshold": ac,
        "hit": hit,
        "crit": crit,
        "damage": damage,
        "detail": " ".join(detail_parts)
    }


# === Apply D&D themed styling ===
def set_dnd_theme():
    st.markdown("""
    <style>
        /* Main background and styles - using the custom parchment background */
        .stApp {
            background-image: url('https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/6e683a25-688c-4bce-9afc-d6644c82e45a/dogcr0-184d6391-3a1b-4491-b14b-257a31504ff0.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiJcL2ZcLzZlNjgzYTI1LTY4OGMtNGJjZS05YWZjLWQ2NjQ0YzgyZTQ1YVwvZG9nY3IwLTE4NGQ2MzkxLTNhMWItNDQ5MS1iMTRiLTI1N2EzMTUwNGZmMC5qcGcifV1dLCJhdWQiOlsidXJuOnNlcnZpY2U6ZmlsZS5kb3dubG9hZCJdfQ.ECo-N-4D32QxdJ05G9wvcbvOgxCS6oly_lBL-gjKBKk');
            background-size: 100% 100%;
            background-repeat: no-repeat;
            background-attachment: fixed;
            color: #3b2e1e;
        }

        /* Make all containers more transparent to let the parchment show through */
        div[data-testid="stVerticalBlock"] > div {
            background-color: rgba(252, 246, 231, 0.65);
            border-radius: 8px;
            padding: 10px;
            margin: 10px 0;
            border: 2px solid #9c7448;
            box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.2);
        }

        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Cinzel', serif;
            color: #6c3e00;
            text-shadow: 1px 1px 2px rgba(107, 83, 28, 0.3);
            border-bottom: 2px solid #9c7448;
            padding-bottom: 8px;
        }

        /* Main title */
        h1 {
            font-size: 3rem !important;
            text-align: center;
            margin-bottom: 30px !important;
            letter-spacing: 1px;
            background-image: linear-gradient(to right, rgba(156, 116, 72, 0), rgba(156, 116, 72, 0.5), rgba(156, 116, 72, 0));
            padding: 20px 0 !important;
        }

        /* Buttons */
        .stButton > button {
            background-color: #6c3e00;
            color: #fdf3d0;
            border: 2px solid #4e2c00;
            font-family: 'Cinzel', serif;
            font-weight: bold;
            border-radius: 5px;
            transition: all 0.3s;
        }

        .stButton > button:hover {
            background-color: #895000;
            border-color: #6c3e00;
            transform: translateY(-2px);
            box-shadow: 0 5px 10px rgba(0, 0, 0, 0.2);
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: rgba(58, 43, 24, 0.85);
            border-right: 3px solid #6c3e00;
        }

        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3 {
            color: #fdf3d0;
            border-bottom-color: #fdf3d0;
        }

        section[data-testid="stSidebar"] .stMarkdown {
            color: #fdf3d0;
        }

        /* Chat messages */
        .stChatMessage {
            background-color: rgba(252, 246, 231, 0.8) !important;
            border: 2px solid #9c7448 !important;
            border-radius: 8px !important;
            margin: 10px 0 !important;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
        }

        .stChatMessage.user [data-testid="chatAvatarIcon-user"] {
            background-color: #4569b3 !important;
        }

        .stChatMessage.assistant [data-testid="chatAvatarIcon-assistant"] {
            background-color: #8c4a00 !important;
        }

        /* File uploader */
        .stFileUploader {
            background-color: rgba(252, 246, 231, 0.7);
            padding: 15px;
            border-radius: 8px;
            border: 2px dashed #9c7448;
        }

        /* Number inputs, sliders, etc. */
        .stSlider {
            padding: 10px;
            background-color: rgba(252, 246, 231, 0.7);
            border-radius: 8px;
        }

        .stNumberInput {
            background-color: rgba(252, 246, 231, 0.7);
        }

        /* Text areas */
        .stTextArea textarea {
            background-color: rgba(252, 246, 231, 0.7);
            border: 1px solid #9c7448;
            font-family: 'Fondamento', cursive;
        }

        /* Select boxes */
        .stSelectbox {
            background-color: rgba(252, 246, 231, 0.7);
        }

        /* Expanders */
        .stExpander {
            border: 1px solid #9c7448;
            border-radius: 5px;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
        }

        /* HP bars styling */
        .hp-bar-container {
            width: 100%;
            height: 20px;
            background-color: #3a3a3a;
            border-radius: 10px;
            overflow: hidden;
            margin: 5px 0;
            border: 1px solid #000;
            box-shadow: inset 0 0 5px rgba(0, 0, 0, 0.5);
        }

        .hp-bar {
            height: 100%;
            background: linear-gradient(90deg, #6b0000 0%, #a90000 50%, #df1f1f 100%);
            border-radius: 10px;
            transition: width 0.5s ease-in-out;
        }

        /* Scene map styling */
        .scene-map {
            border: 5px solid #6c3e00;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
        }

        /* D&D-style decorative elements */
        .dnd-divider {
            text-align: center;
            margin: 20px 0;
            height: 20px;
            background-image: url('https://i.imgur.com/6ZwiYV4.png');
            background-repeat: repeat-x;
            background-size: contain;
        }

        /* Import fantasy fonts */
        @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Fondamento&display=swap');

        /* Custom styles for Character Cards */
        .character-card {
            border: 2px solid #9c7448;
            border-radius: 8px;
            padding: 10px;
            margin: 10px 0;
            background-color: rgba(252, 246, 231, 0.9);
            box-shadow: 3px 3px 8px rgba(0, 0, 0, 0.2);
        }

        .character-card h4 {
            margin-top: 0;
            border-bottom: 1px solid #9c7448;
            padding-bottom: 5px;
        }

        /* Make chat input more prominent */
        [data-testid="stChatInput"] {
            background-color: rgba(252, 246, 231, 0.9) !important;
            border: 2px solid #9c7448 !important;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            padding: 10px !important;
            font-family: 'Fondamento', cursive !important;
        }

        /* Ensure text is readable on the dark sidebar */
        .stCheckbox label p {
            color: #fdf3d0 !important; 
        }

        /* Custom NPC grid */
        .npc-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }

        /* Custom header with dragon decoration */
        .dnd-header {
            text-align: center;
            padding: 20px 10px 0;
            position: relative;
        }

        .dnd-header:before, .dnd-header:after {
            content: "";
            display: inline-block;
            width: 80px;
            height: 80px;
            background-image: url('https://i.imgur.com/JcXiklo.png');
            background-size: contain;
            background-repeat: no-repeat;
            position: absolute;
            top: 0;
        }

        .dnd-header:before {
            left: 10px;
            transform: scaleX(-1);
        }

        .dnd-header:after {
            right: 10px;
        }
    </style>

    <div class="dnd-header"></div>
    """, unsafe_allow_html=True)
