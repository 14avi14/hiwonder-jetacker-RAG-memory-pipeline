"""memory_creation.py - populates RAG memory DB

This takes in images, spatial coordinates, and timestamp data and
stores those in a RAG-based vector DB. Groups of images are summarized
via a VLM textually. These summarizations are embedded and stored for
use during semantic similarity searches. The associated timestamp and
spatial, data are fed to the vector DB to be stored as well(for
details, check vector_store.py)

For the VLM, currently qwen 3.8 27b is being used and the API is Groq.

Below, there is also the ability to feed videos. To do so,
frames are still fed as individual images, but they will be turned into
a short clip and sent to a video model if that is chosen.
"""
import os
import random
import base64
import pprint
import time

import requests
from dotenv import load_dotenv
import cv2

load_dotenv()


USE_LLM = True
SEND_VIDEO_FORM = False
#URL = "https://openrouter.ai/api/v1/chat/completions"
#API_KEY_REF = os.getenv("OPENROUTER_KEY")
#headers = {
#    "Authorization": f"Bearer {API_KEY_REF}",
#    "Content-Type": "application/json"
#}

#MODEL = "inclusionai/ling-3.0-flash-vl:free"

URL = "https://api.groq.com/openai/v1/chat/completions"
API_KEY_REF = os.getenv("GROQ_API_KEY")
headers = {
    "Authorization": f"Bearer {API_KEY_REF}",
    "Content-Type": "application/json"
}
MODEL = "qwen/qwen3.8-27b"

CAPTIONING_PROMPT = """You are making descriptions for a scene to be stored
in a RAG based search system database for navigaton. Use the images to output
ONE caption of all landmarks, objects, logos, words,
actions, etc. in the images(1-2 SENTENCES ONLY).
Use identifying adjectives(e.g. color, size, distance from camera etc.)
that are useful.
"""

MOCK_CAPTIONS = [
    """There is a black chair next to a dark desk with
    multiple books and a laptop on top, one meter or so away. A student is sitting
    with a pencil in hand.""",
    """There is a dog sitting on a bed, with dark brown fur. Behind, a person
    is laying down with a book on their face. There is a nice window view.""",
    """There is a dining table in the center of a small room. Behind the dining
    table, there is a stove and an oven, with a black microwave on top"""
]

TEMP_VID_FILENAME = "output.mp4"

def frames_to_video(frames):
    frame_width = len(frames[0])
    frame_height = len(frames)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps = 10 # ARBITRARY, should be the same as topic publishing frequency
    out = cv2.VideoWriter(TEMP_VID_FILENAME, fourcc, fps, (frame_width, frame_height))

    for frame in frames:
        out.write(frame)

    out.release()

def encode_to_base64(obj):
    return base64.b64encode(obj).decode("utf-8")

def enode_from_path_to_base64(path):
    with open(path, "rb") as f:
        return encode_to_base64(f.read())

def get_caption(video):
    if not USE_LLM:
        # return default caption
        return random.choice(MOCK_CAPTIONS)

    # Will either process one vidoe, or multiple images, depends on API
    # availability
    if SEND_VIDEO_FORM:
        video_input = [{
            "type": "video_url",
            "video_url": {
                "url": video
            }
        }]
    else:
        video_input = [{
            "type": "image_url",
            "image_url": {
                "url": frame_url
            }
        } for frame_url in video]
    
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": CAPTIONING_PROMPT
                    },
                    *video_input # Will either be multiple images or one video
                ]
            }
        ],
        "temperature": 0.0,
        "max_tokens": 900
    }

    print("[memory_creation.py - get_caption(vid)]")
    response = requests.post(url=URL, headers=headers, json=payload)
    print(f"API RESPONSE:")
    pprint.pprint(response.json())
    
    response_json = response.json()
    if "error" in response_json:
        print("-" * 25 + " API Response Error " + "-" * 25)
        text = None
    else:
        text = response_json["choices"][0]["message"]["content"]
        print("RESPONSE TEXT:", text)
    return text

def memorize(frames, car_states, timestamps, vec_db, max_frames_per):
    captions = []
    for i in range(0, len(frames), max_frames_per):
        video_frames = frames[i:i+max_frames_per]
        if SEND_VIDEO_FORM:
            frames_to_video(video_frames)
            base64_video = encode_to_base64(TEMP_VID_FILENAME)
            full_video_url = f"data:video/mp4;base64,{base64_video}"
        else:
            byte_frames = [cv2.imencode(".jpg", frame)[1].tobytes() for frame in video_frames]
            b64_frames = [encode_to_base64(img) for img in byte_frames]
            full_video_url = [f"data:image/jpeg;base64,{frame}" for frame in b64_frames]
    
        caption = get_caption(full_video_url)
        try:
            while caption is None:
                print("Trying again after 35 seconds...")
                time.sleep(35)
                caption = get_caption(full_video_url)
                #request_again = input("Try again(y/n)?: ")
                #if request_again.lower() == "y":
                #    caption = get_caption(full_video_url)
                #else:
                #    print("Discontinuing...(Memory that has already been processed will be saved)")
                #    return
        except KeyboardInterrupt:
            print(f"----- DISCONTINUING MEMORY PROCESSING -----")
            return

        captions.append(caption)
        if os.path.isfile(TEMP_VID_FILENAME):
            os.remove(TEMP_VID_FILENAME) # Not sure if this will help with latency at all

        # Insert into memory
        docs = [{
            "text": caption, "location": car_states[i]["position_xy"],
            "heading_deg": car_states[i]["heading_deg"], "timestamp": timestamps[i]
            } for i in range(len(captions))]
        vec_db.insert(docs)

