import streamlit as st
from openai import OpenAI
import openai
from pydantic import BaseModel, Field
from utils import *
import PyPDF2
from dotenv import load_dotenv
import os
import base64
import json
import re
import time
from streamlit_extras.stylable_container import stylable_container
from utils import save_scene_graph_image

# === 自定义主题和样式 ===
def set_dnd_theme():
    st.markdown("""
    <style>
        /* 整体应用样式 */
        .stApp {
            background-color: #f0e6d2;
            color: #2c2c2c;
        }
        
        /* 标题样式 */
        h1, h2, h3, h4 {
            font-family: 'Cinzel', serif !important;
            color: #762f18 !important;
            border-bottom: 2px solid #c9b18c;
            padding-bottom: 8px;
        }
        
        h1 {
            background-image: url('https://i.imgur.com/7JLE9RJ.png');
            background-repeat: no-repeat;
            background-position: left center;
            background-size: 50px;
            padding-left: 60px;
            margin-bottom: 30px !important;
        }
        
        /* 聊天容器 */
        .element-container:has(.stChatMessage) {
            background-color: #f5efe0;
            border-radius: 10px;
            border: 2px solid #c9b18c;
            padding: 10px;
            margin-bottom: 15px;
        }
        
        /* 用户气泡样式 */
        .stChatMessage [data-testid="StyledThumbUserMessage"] {
            background-color: #e0e8f5 !important;
            border: 1px solid #8ba3c7 !important;
            color: #2c3e50 !important;
        }
        
        /* AI气泡样式 */
        .stChatMessage [data-testid="StyledThumbAIMessage"] {
            background-color: #f7e8d5 !important;
            border: 1px solid #d0b894 !important;
            color: #4a3520 !important;
        }
        
        /* 按钮样式 */
        .stButton>button {
            background-color: #762f18;
            color: #f2e9d9;
            border: 2px solid #5e2f0d;
            border-radius: 6px;
            font-family: 'Cinzel', serif;
            font-weight: bold;
            padding: 8px 16px;
            transition: all 0.3s;
        }
        
        .stButton>button:hover {
            background-color: #924b33;
            border-color: #762f18;
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* 输入框样式 */
        .stTextInput>div>div>input, .stNumberInput>div>div>input {
            background-color: #f5efe0;
            border: 1px solid #c9b18c;
            border-radius: 6px;
            padding: 8px 12px;
            font-family: 'Spectral', serif;
        }
        
        /* 选择器样式 */
        .stSelectbox>div>div, .stMultiselect>div>div {
            background-color: #f5efe0;
            border: 1px solid #c9b18c;
        }
        
        /* 侧边栏样式 */
        [data-testid="stSidebar"] {
            background-color: #31281f;
            background-image: url('https://i.imgur.com/1uYP6n3.png');
            background-repeat: repeat;
            color: #f2e9d9;
            border-right: 3px solid #5e2f0d;
        }
        
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #f2e9d9 !important;
            border-bottom: 2px solid #924b33;
        }
        
        /* 进度条（HP条）自定义样式 */
        .hp-bar-container {
            width: 100%;
            height: 20px;
            background-color: #4e4e4e;
            border-radius: 10px;
            margin: 5px 0;
            overflow: hidden;
            position: relative;
        }
        
        .hp-bar {
            height: 100%;
            border-radius: 10px;
            background: linear-gradient(90deg, #a12b2b, #d93636);
            transition: width 0.5s ease-in-out;
        }
        
        /* 角色卡片样式 */
        .character-card {
            background-color: rgba(165, 134, 95, 0.2);
            border-radius: 8px;
            border: 1px solid #a5865f;
            padding: 10px;
            margin: 8px 0;
        }
        
        /* 聊天输入区域 */
        [data-testid="stChatInput"] {
            background-color: #f5efe0;
            border: 2px solid #c9b18c;
            border-radius: 10px;
            padding: 5px;
        }
        
        /* 文件上传器 */
        [data-testid="stFileUploader"] {
            background-color: #f5efe0;
            border: 2px dashed #c9b18c;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }
        
        /* Tab样式 */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #e0d6c2;
            border-radius: 10px 10px 0 0;
            padding: 5px 5px 0 5px;
            gap: 10px;
        }
        
        .stTabs [role="tab"] {
            background-color: #d3c0a3;
            border-radius: 5px 5px 0 0;
            border: 1px solid #a5865f;
            border-bottom: none;
            color: #5e2f0d;
            padding: 10px 20px;
            font-family: 'Cinzel', serif;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #f0e6d2;
            border-top: 3px solid #762f18;
            margin-top: -3px;
            font-weight: bold;
        }
        
        .stTabs [data-baseweb="tab-panel"] {
            background-color: #f0e6d2;
            border: 1px solid #a5865f;
            border-top: none;
            border-radius: 0 0 10px 10px;
            padding: 20px;
        }
        
        /* 装饰元素 */
        .decorative-divider {
            background-image: url('https://i.imgur.com/JgNiQkn.png');
            background-repeat: repeat-x;
            height: 20px;
            margin: 20px 0;
        }
        
        /* 自定义扩展器 */
        .stExpander {
            border: 1px solid #c9b18c;
            border-radius: 10px;
            overflow: hidden;
        }
        
        .stExpander [data-baseweb="accordion"] {
            background-color: #e5d9c3;
        }
        
        /* 滚动条样式 */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: #e5d9c3;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #a5865f;
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #8e7148;
        }
        
        /* 场景地图容器 */
        .scene-map-container {
            background-color: #f5efe0;
            border: 3px solid #a5865f;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        /* 字体导入 */
        @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Spectral:wght@400;600&display=swap');
    </style>
    """, unsafe_allow_html=True)

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

if "show_intro" not in st.session_state:
    st.session_state.show_intro = True

if "current_tab" not in st.session_state:
    st.session_state.current_tab = "adventure"

if "dice_history" not in st.session_state:
    st.session_state.dice_history = []

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

# 应用自定义主题
set_dnd_theme()

# 创建漂亮的标题效果
def fancy_header(title, icon="🧙"):
    st.markdown(f"""
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <div style="font-size: 2rem; margin-right: 0.5rem;">{icon}</div>
        <h1 style="margin: 0;">{title}</h1>
    </div>
    <div class="decorative-divider"></div>
    """, unsafe_allow_html=True)

# === 主应用界面 ===
fancy_header("Dragon's Scroll - D&D Adventure Manager", "🐉")

# 首次加载显示欢迎消息
if st.session_state.show_intro:
    with stylable_container(
        key="welcome_container",
        css_styles="""
            {
                background-color: #f7e8d5;
                border: 2px solid #c9b18c;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 25px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                position: relative;
                background-image: url('https://i.imgur.com/1uYP6n3.png');
                background-repeat: repeat;
                background-blend-mode: overlay;
            }
        """
    ):
        st.markdown("""
        ### 🧙‍♂️ Welcome, brave adventurer!

        This magical scroll will help you on your journey through realms unknown. 
        Here you can:
        
        - **Generate NPCs** for your campaign
        - **Upload adventure scripts** to create scene maps
        - **Chat with an AI Dungeon Master** that responds to your actions
        - **Roll dice** and track character status
        
        Begin your adventure by exploring the tabs below!
        """)
        if st.button("Dismiss", key="dismiss_intro"):
            st.session_state.show_intro = False
            st.experimental_rerun()

# 主标签页
tabs = st.tabs(["📖 Adventure", "🧙 NPCs", "🗺️ Scene Map", "📋 Character Sheets"])

with tabs[0]:  # 冒险标签
    st.session_state.current_tab = "adventure"
    
    # 聊天历史显示
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_log:
            role = msg["role"] if isinstance(msg, dict) else msg.role
            content = msg["content"] if isinstance(msg, dict) else msg.content

            if content is None or role == "tool":
                continue  # hide tool result content

            with st.chat_message(role):
                st.markdown(content)
    
    # 聊天输入区域样式增强
    with stylable_container(
        key="chat_input_container",
        css_styles="""
            {
                background-color: #f5efe0;
                border: 2px solid #c9b18c;
                border-radius: 10px;
                padding: 15px;
                margin-top: 20px;
            }
        """
    ):
        st.subheader("🎲 What happens next?")
        
        # 快速行动按钮
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("Attack", key="quick_attack"):
                prompt = "I attack the nearest enemy with my weapon."
                st.session_state.chat_log.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Run GPT with tool support
                new_messages, tool_outputs = run_full_turn(client, get_dm_agent(), st.session_state.chat_log)
                st.session_state.chat_log.extend(new_messages)
                
                # Apply effects like healing or damage
                apply_hp_effects(client, new_messages, get_dm_agent().players)
                st.experimental_rerun()
        
        with col2:
            if st.button("Look Around", key="quick_look"):
                prompt = "I look around and examine my surroundings carefully."
                st.session_state.chat_log.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Run GPT with tool support
                new_messages, tool_outputs = run_full_turn(client, get_dm_agent(), st.session_state.chat_log)
                st.session_state.chat_log.extend(new_messages)
                st.experimental_rerun()
        
        with col3:
            if st.button("Roll Initiative", key="quick_initiative"):
                prompt = "Let's roll for initiative."
                st.session_state.chat_log.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Run GPT with tool support
                new_messages, tool_outputs = run_full_turn(client, get_dm_agent(), st.session_state.chat_log)
                st.session_state.chat_log.extend(new_messages)
                st.experimental_rerun()
        
        with col4:
            if st.button("Cast Spell", key="quick_spell"):
                prompt = "I want to cast a spell."
                st.session_state.chat_log.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Run GPT with tool support
                new_messages, tool_outputs = run_full_turn(client, get_dm_agent(), st.session_state.chat_log)
                st.session_state.chat_log.extend(new_messages)
                st.experimental_rerun()
        
        # 主聊天输入
        if prompt := st.chat_input("Enter your action..."):
            st.session_state.chat_log.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Run GPT with tool support
            new_messages, tool_outputs = run_full_turn(client, get_dm_agent(), st.session_state.chat_log)
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
            apply_hp_effects(client, new_messages, get_dm_agent().players)
            st.experimental_rerun()
    
    # 显示骰子历史
    if st.session_state.dice_history:
        with st.expander("📜 Dice Roll History", expanded=True):
            for i, roll in enumerate(reversed(st.session_state.dice_history[-5:])):
                st.markdown(f"**Roll {len(st.session_state.dice_history)-i}:** {roll['text']}")
    
with tabs[1]:  # NPC标签
    st.session_state.current_tab = "npcs"
    
    st.markdown("""
    ### 🧙 NPC Generator
    
    Create unique non-player characters for your adventure. These characters will be available 
    to your AI Dungeon Master during the adventure.
    """)
    
    with stylable_container(
        key="npc_generator",
        css_styles="""
            {
                background-color: #f5efe0;
                border: 2px solid #c9b18c;
                border-radius: 10px;
                padding: 15px;
                margin-top: 10px;
                margin-bottom: 20px;
            }
        """
    ):
        col1, col2 = st.columns([3, 1])
        with col1:
            npc_count = st.number_input("How many NPCs do you want to generate?", min_value=1, max_value=10, value=3)
        with col2:
            generate_button = st.button("🧙 Generate NPCs", use_container_width=True)
        
        if generate_button:
            with st.spinner("The mystical forces are creating characters..."):
                result = sample_npcs(npc_count)
                st.session_state["npcs"] = result["npcs"]
                time.sleep(0.5)  # 添加短暂延迟以增强效果感
            
            st.success("✨ NPCs have been summoned successfully!")
    
    # NPC Cards Display with improved visuals
    if "npcs" in st.session_state:
        st.markdown("### 📚 Generated NPCs")
        
        # 网格显示NPC
        npc_cols = st.columns(2)  # 每行2个NPC
        
        for i, npc in enumerate(st.session_state["npcs"]):
            col_idx = i % 2
            
            with npc_cols[col_idx]:
                with stylable_container(
                    key=f"npc_card_{i}",
                    css_styles="""
                        {
                            background-color: #f7e8d5;
                            border: 2px solid #c9b18c;
                            border-radius: 10px;
                            padding: 15px;
                            margin-bottom: 15px;
                            position: relative;
                            overflow: hidden;
                        }
                        
                        :hover {
                            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                            transform: translateY(-2px);
                            transition: all 0.3s ease;
                        }
                    """
                ):
                    # NPC Header
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h3 style="margin: 0; color: #762f18;">{npc.name}</h3>
                        <div style="font-style: italic; color: #8b4513;">{npc.race} {npc.char_class}</div>
                    </div>
                    <div style="height: 2px; background: linear-gradient(90deg, #c9b18c, transparent); margin: 8px 0;"></div>
                    """, unsafe_allow_html=True)
                    
                    # HP条
                    hp_percent = int((npc.hp / npc.max_hp) * 100)
                    st.markdown(f"""
                    <div style="margin: 10px 0;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.9em; margin-bottom: 5px;">
                            <span>HP</span>
                            <span>{npc.hp}/{npc.max_hp}</span>
                        </div>
                        <div class="hp-bar-container">
                            <div class="hp-bar" style="width: {hp_percent}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # NPC详情
                    with st.expander("Character Details"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Personality:** {npc.personality}")
                            st.markdown(f"**Combat Role:** {npc.combat_role}")
                            st.markdown(f"**Voice:** {npc.voice}")
                        with col2:
                            st.markdown(f"**Quirks:** {npc.quirks}")
                            st.markdown(f"**Unique Trait:** {npc.trait}")
                        
                        # 显示属性点
                        if hasattr(npc, "stats") and isinstance(npc.stats, dict):
                            st.markdown("##### Ability Scores")
                            stat_cols = st.columns(6)
                            for i, (stat, value) in enumerate(npc.stats.items()):
                                with stat_cols[i]:
                                    modifier = (value - 10) // 2
                                    mod_str = f"+{modifier}" if modifier >= 0 else f"{modifier}"
                                    st.markdown(f"""
                                    <div style="text-align: center; background-color: rgba(255,255,255,0.5); 
                                                border-radius: 5px; padding: 5px; border: 1px solid #c9b18c;">
                                        <div style="font-weight: bold;">{stat}</div>
                                        <div style="font-size: 1.2em;">{value}</div>
                                        <div style="font-size: 0.8em; color: {'green' if modifier >= 0 else 'red'};">
                                            {mod_str}
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)

with tabs[2]:  # 场景地图标签
    st.session_state.current_tab = "scene_map"
    
    st.markdown("""
    ### 🗺️ Adventure Script & Scene Map
    
    Upload your adventure script to automatically generate an interactive scene map.
    The map will be used by the AI Dungeon Master to guide your adventure.
    """)
    
    # 脚本上传区域
    with stylable_container(
        key="script_upload",
        css_styles="""
            {
                background-color: #f5efe0;
                border: 2px solid #c9b18c;
                border-radius: 10px;
                padding: 20px;
                margin-top: 10px;
                margin-bottom: 20px;
            }
        """
    ):
        uploaded_file = st.file_uploader("Upload your campaign script (PDF or TXT)", type=["pdf", "txt"])
        
        if uploaded_file:
            with st.spinner("📜 Analyzing the ancient script..."):
                if uploaded_file.type == "application/pdf":
                    try:
                        pdf_reader = PyPDF2.PdfReader(uploaded_file)
                        plot = ""
                        for page in pdf_reader.pages:
                            plot += page.extract_text() or ""
                    except Exception as e:
                        st.error(f"❌ Could not read PDF: {e}")
                        plot = ""
                elif uploaded_file.type == "text/plain":
                    plot = uploaded_file.read().decode("utf-8")
                else:
                    st.warning("Unsupported file type")
                    plot = ""
                st.session_state["script_text"] = plot
                
                # 处理场景图
                messages = [{"role": "user", "content": plot}]
                scene_response, _ = run_full_turn(client, scene_graph_agent, messages)
                scene_data_raw = scene_response[-1].content if hasattr(scene_response[-1], "content") else scene_response[-1]["content"]

                # Strip ```json ... ``` if GPT wrapped it
                scene_data_raw = re.sub(r"^```(?:json)?|```$", "", scene_data_raw.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()

                try:
                    scene_data = json.loads(scene_data_raw)
                    
                    st.session_state["scene_list"] = scene_data
                    st.session_state["current_scene"] = scene_data[0]["title"] if scene_data else "Unknown"
                    graph_path = save_scene_graph_image(scene_data, current_location=st.session_state["current_scene"])
                    with open(graph_path, "rb") as f:
                        encoded_img = base64.b64encode(f.read()).decode()

                    # Store encoded image for display
                    st.session_state["scene_graph_img"] = encoded_img
                    
                    st.success("✅ Scene map successfully extracted!")
                except json.JSONDecodeError as e:
                    st.error(f"❌ Could not parse scene data: {e}")
    
    # 场景图和脚本预览
    col1, col2 = st.columns(2)
    
    with col1:
        if "scene_graph_img" in st.session_state:
            st.markdown("#### 🗺️ Scene Map")
            with stylable_container(
                key="scene_map_display",
                css_styles="""
                    {
                        background-color: #f7e8d5;
                        border: 3px solid #a5865f;
                        border-radius: 10px;
                        padding: 15px;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                    }
                """
            ):
                st.image(f"data:image/png;base64,{st.session_state['scene_graph_img']}", use_column_width=True)
                
                if "scene_list" in st.session_state:
                    # 场景选择器
                    scene_titles = [scene["title"] for scene in st.session_state["scene_list"]]
                    selected_scene = st.selectbox("Current Location", scene_titles, 
                                                 index=scene_titles.index(st.session_state["current_scene"]) if st.session_state["current_scene"] in scene_titles else 0)
                    
                    if selected_scene != st.session_state["current_scene"]:
                        st.session_state["current_scene"] = selected_scene
                        # 更新场景图
                        graph_path = save_scene_graph_image(st.session_state["scene_list"], current_location=selected_scene)
                        with open(graph_path, "rb") as f:
                            st.session_state["scene_graph_img"] = base64.b64encode(f.read()).decode()
                        st.experimental_rerun()
    
    with col2:
        if "script_text" in st.session_state:
            st.markdown("#### 📜 Script Preview")
            with stylable_container(
                key="script_preview",
                css_styles="""
                    {
                        background-color: #f7e8d5;
                        border: 1px solid #c9b18c;
                        border-radius: 10px;
                        padding: 15px;
                        height: 400px;
                        overflow-y: auto;
                    }
                """
            ):
                st.text_area("", st.session_state["script_text"], height=350)
        else:
            st.info("Please upload a PDF or TXT file to begin setting up your world.")
    
    # 场景详情
    if "scene_list" in st.session_state and "current_scene" in st.session_state:
        current_scene_data = None
        for scene in st.session_state["scene_list"]:
            if scene["title"] == st.session_state["current_scene"]:
                current_scene_data = scene
                break
        
        if current_scene_data:
            st.markdown("#### 📌 Current Scene Details")
            with stylable_container(
                key="scene_details",
                css_styles="""
                    {
                        background-color: #f5efe0;
                        border: 2px solid #c9b18c;
                        border-radius: 10px;
                        padding: 15px;
                        margin-top: 15px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }
                """
            ):
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h4 style="margin: 0;">{current_scene_data["title"]}</h4>
                        <div style="font-style: italic; color: #762f18; margin-top: 5px;">
                            {current_scene_data["location"]}
                        </div>
                    </div>
                    <div style="background-color: #762f18; color: white; padding: 5px 10px; 
                                border-radius: 5px; font-weight: bold;">
                        Scene #{current_scene_data["scene_number"]}
                    </div>
                </div>
                <div style="height: 2px; background: linear-gradient(90deg, #c9b18c, transparent); margin: 10px 0;"></div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"**Description:** {current_scene_data['description']}")
                
                if current_scene_data.get("connections"):
                    st.markdown("**Connected Scenes:**")
                    for conn in current_scene_data["connections"]:
                        # 创建一个按钮可以直接移动到连接的场景
                        if st.button(f"→ {conn}", key=f"move_to_{conn}"):
                            st.session_state["current_scene"] = conn
                            graph_path = save_scene_graph_image(st.session_state["scene_list"], current_location=conn)
                            with open(graph_path, "rb") as f:
                                st.session_state["scene_graph_img"] = base64.b64encode(f.read()).decode()
                            
                            # 添加场景移动到聊天记录
                            message = f"The party moves to {conn}."
                            st.session_state.chat_log.append({"role": "user", "content": message})
                            
                            # 让DM响应场景变化
                            new_messages, _ = run_full_turn(client, get_dm_agent(), st.session_state.chat_log)
                            st.session_state.chat_log.extend(new_messages)
                            
                            st.experimental_rerun()