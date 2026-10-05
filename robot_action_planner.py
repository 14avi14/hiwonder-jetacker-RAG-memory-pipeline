"""robot_action_planner.py - uses memory data to answer query

The script uses the vector DB containing prior data to answer a query.
First, the query is fed to a LLM, which outputs semantic hints to use to
search the vector DB. Once the vector DB is searched, the most similar
semantically stored "memories" are retrieved, along with the timestamp
and spatial data. The data and the captions are fed to an LLM to be
condensed into a reasonable summary. Finally, this summary is fed to
the LLM yet another time for a final, JSON formatted output.

The LLM model used is the gpt-oss-120b, while the API is Groq.
"""
import os
import json
import pprint
import time

from dotenv import load_dotenv
import requests

load_dotenv()

USE_LLM = True
URL = "https://api.groq.com/openai/v1/chat/completions"
API_KEY_REF = os.getenv("GROQ_API_KEY")
headers = {
    "Authorization": f"Bearer {API_KEY_REF}",
    "Content-Type": "application/json"
}
MODEL = "openai/gpt-oss-120b"

SEMANTIC_SEARCH_SYSTEM_PROMPT = """You are a search query generator for
a vision-only RAG pipeline. The database contains image summaries
featuring object names, logos, brands, colors, and spatial size cues.

Task:
Convert the user's request into 1 short, highly specific search prompt
optimized to query this visual database.

Output format:
Return ONLY valid JSON with no markdown wrapping or extra text, strictly
in this format:
{"prompts": ["<search_query_string>"]}"""
CONTEXT_SYSTEM_PROMPT = """You are a memory navigation context synthesizer.
Analyze the user's data and output a strictly 1-sentence context snippet
to answer the query. 

Guidelines:
- Include exact coordinates, headings, and timestamps whenever available
  and relevant.
- Be extremely concise; omit all unnecessary filler words.
- Focus purely on actionable navigation data."""
FINAL_POSITION_PROMPT = """You are the final decision-maker in a robot
navigation RAG pipeline. Analyze the user's context to determine the
final target position and reasoning.

Rules:
- If the user explicitly asks to be led or navigated somewhere, include
  the `goal_pose` as `[x, y, heading_degrees]`. Heading must be between -180 and 180.
- If the user does NOT explicitly ask to be led/navigated somewhere, set
  `goal_pose` to `null` and put the requested answer inside `reasoning`.

Output format:
Return ONLY valid JSON matching this exact schema, with no markdown or
extra text:
{
    "goal_pose": [0.0, 0.0, 0] or null,
    "reasoning": "<string explanation>"
}"""

FINAL_OUTPUT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "ouput",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "goal_pose": {
                    "type": ["array", "null"],
                    "items": {"type": "number"}
                },
                "reasoning": {"type": "string"}
            },
            "required": ["goal_pose", "reasoning"],
            "additionalProperties": False
        }
    }
}

def get_llm_context(query, retrieved_memories):
    if not USE_LLM:
        return "There was a chair and desk at (1.0, -1.0, 0.0)"

    user_prompt = f"Query: {query} Data: {retrieved_memories}"

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": CONTEXT_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "temperature": 0.2,
        "max_tokens": 900
    }

    response = requests.post(url=URL, headers=headers, json=payload)
    print("Raw response:", response.json())
    text = response.json()["choices"][0]["message"]["content"]
    return text


def get_semantic_prompts(query):
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": SEMANTIC_SEARCH_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": query
            }
        ]
    }

    response = requests.post(url=URL, headers=headers, json=payload)
    print("Raw response:", response.json())
    text = response.json()["choices"][0]["message"]["content"]
    json_data = json.loads(text)
    return json_data

def obtain_context(query, vec_db, current_context=None):
    func_start = time.time()
    
    start = time.time()
    semantic_search_prompts = get_semantic_prompts(query)["prompts"]
    print("-" * 25 + " SEMANTIC SEARCH PROMPTS " + "-" * 25)
    print(f"Took {round(time.time() - start)} seconds\n")
    pprint.pprint(semantic_search_prompts)

    start = time.time()
    semantic_similar_docs = []
    for prompt in semantic_search_prompts:
        docs = vec_db.text_query(prompt, top_k=2)
        for doc in docs:
            if doc not in semantic_similar_docs:
                semantic_similar_docs.append(doc)

    print("-" * 25 + " SEMANTIC SIMILAR DOCS " + "-" * 25)
    print(f"Took {round(time.time() - start)} seconds\n")

    start = time.time()
    semantic_context = get_llm_context(query, semantic_similar_docs)
    print("-" * 25 + " LLM semantic memory summary " + "-" * 25)
    print(f"Took {round(time.time() - start)} seconds\n")
    print(semantic_context)

    # timestamp_search =
    # timestamp_similar_docs = vec_db.euclidian_similarity_query(timestamp_search, "timestamp")
    # timestamp_context = get_llm_context(query, timestamp_similar_docs)
    # print("----- LLM timestamp-based memory summary -----")
    # print(timestamp_context)
    
    # USE LLM to summarize into cohesive context
    # text = get_llm_context(query, semantic_context + " " + timestamp_context)
    text = semantic_context
    print("-" * 25 + " FINAL CONTEXT " + "-" * 25)
    print(text)
    print()
    print(f"Total function took {round(time.time() - func_start)} seconds\n")
    return text

def get_final_position(query, context):
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": FINAL_POSITION_PROMPT
            },
            {
                "role": "user",
                "content": f"Context: {context} | Query: {query}"
            }
        ],
        "response_format": FINAL_OUTPUT_SCHEMA
    }

    response = requests.post(url=URL, headers=headers, json=payload)
    print("Raw response:", response.json())
    final_json = json.loads(response.json()["choices"][0]["message"]["content"])
    return final_json
