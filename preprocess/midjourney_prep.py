import re
import os
import json
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass
import sys
sys.path.append('..')
from utils import *

# midjourney-specific functions

@dataclass
class UserRequest:
    prompt: str
    generated_url: str
    timestamp: str
    author_id: list
    username: list

def get_prompt(message):
    """Extracts the prompt from the message content, which is located between double stars."""
    content = message["content"]
    # Replace newlines with spaces; makes the regex below work.
    content = content.replace("\n", " ")
    # Find the text enclosed by two consecutive stars.
    BETWEEN_STARS = "\\*\\*(.*?)\\*\\*"
    match = re.search(BETWEEN_STARS, content)
    if match:
        return match.group()[2:-2]  # Exclude the stars.

def get_message_type(message):
    # Detect the message type based on the UI components shown to the user.
    # See https://discord.com/developers/docs/interactions/message-components#what-is-a-component
    COMPONENTS_FOR_INITIAL_AND_VARIATION = set(
        ['U1', 'U2', 'U3', 'U4', '⟳', 'V1', 'V2', 'V3', 'V4'])
    COMPONENTS_FOR_UPSCALE = set(
        ['Make Variations', 'Upscale to Max', 'Light Upscale Redo'])
    
    """Figures out the message type based on the UI components displayed."""
    for components in message["components"]:
        for component in components["components"]:
            if component["label"] in COMPONENTS_FOR_INITIAL_AND_VARIATION:
            # For (very few) messages that are supposedly initial or variation requests, the content indicates
            # that they are actually upscale requests. We will just put these aside.
                if "Upscaled" in message["content"]:
                    return "INCONCLUSIVE"
                return "INITIAL_OR_VARIATION"
            elif component["label"] in COMPONENTS_FOR_UPSCALE:
                return "UPSCALE"
    return "TEXT_MESSAGE"

def get_timestamp(message):
    return message['timestamp']

def get_author_id(message):
    if len(message['mentions']) >= 1:
        ids = []
        usernames = []
        for mention in message['mentions']:
            if mention['username'] == "MidJourney Bot" and mention['id'] == '936929561302675456':
                continue
            else:
                ids.append(mention['id'])
                usernames.append(mention['username'])
        return ids, usernames
    else:
        return None, None

def get_generated_image_url(message):
    """Extracts the URL of the generated image from the message."""
    attachments = message["attachments"]
    if len(attachments) == 1:
        return attachments[0]["url"]

def get_midjourney(raw_data_root, save_root):

    filepaths = []
    for dirname, _, filenames in os.walk(raw_data_root):
        for filename in filenames:
            filepaths.append(os.path.join(dirname, filename))

    messages_by_type = defaultdict(list)
    for filepath in filepaths:
        with open(filepath, "r") as f:
            content = json.load(f)
            for single_message_list in content["messages"]:
                assert len(single_message_list) == 1
                message = single_message_list[0]
                message_type = get_message_type(message)
                messages_by_type[message_type].append(message)


    user_requests = []
    for m in messages_by_type["INITIAL_OR_VARIATION"]:
        prompt = get_prompt(m)
        generated_url = get_generated_image_url(m)
        timestamp = get_timestamp(m)
        author_id, username = get_author_id(m)
        # In *very* rare cases, messages are malformed and these fields cannot be extracted.
        if prompt and generated_url:
            user_requests.append(UserRequest(prompt.lower().replace('::', ' :: ').replace(':: -', '::- '), generated_url, timestamp, author_id, username))
            
    num_messages = len(messages_by_type["INITIAL_OR_VARIATION"])

    data_file = pd.DataFrame({"prompt": [r.prompt for r in user_requests if len(r.author_id) == 1],
                        "url": [r.generated_url for r in user_requests if len(r.author_id) == 1],
                        "timestamp": [r.timestamp for r in user_requests if len(r.author_id) == 1],
                        "author_id": [r.author_id[0] for r in user_requests if len(r.author_id) == 1],
                        "username": [r.username[0] for r in user_requests if len(r.username) == 1]
                        })

    data_file.to_csv(os.path.join(save_root, 'midjourney_raw.csv'), index=False)
    return os.path.join(save_root, 'midjourney_raw.csv')