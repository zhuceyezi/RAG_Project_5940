This is a D&D bot working in progress for a final project in INFO 5940 Group 8

To run this project, you need a .env file.
It looks like this:

OPENAI_API_KEY=<your_api_key>

OPENAI_BASE_URL=<cornell_proxy_base_url

Current features:
1. Add/remove NPCs automatically
2. Calculate damage based on rules
3. generate a map(very fansy and feels very D&D) and track current location
4. show stats like hp, attack, defend on sidebar
5. Also show NPCs, their alignment, and their hp bars.
6. Handles any weird interaction
7. Maintains actual hp, updates in real time when take damage/heal.
