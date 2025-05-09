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

    # 设置图形风格为DnD主题
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['axes.facecolor'] = '#f0e6d2'
    plt.rcParams['figure.facecolor'] = '#f5efe0'
    plt.rcParams['font.family'] = 'serif'
    
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
            node_colors.append("#762f18")  # 当前场景为棕红色
        else:
            node_colors.append("#c9b18c")  # 其他场景为棕黄色

    plt.figure(figsize=(6, 6))
    
    # 绘制边缘
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=15, edge_color="#8b4513", width=1.5)
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2400, alpha=0.9, 
                          edgecolors="#5e2f0d", linewidths=2)