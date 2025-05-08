import random
import string
import json
#from clemgame import get_logger

import logging

logger = logging.getLogger(__name__)

def cleanupanswer(prompt_answer: str) -> str:
    """Clean up the answer from the LLM DM."""
    #if "```json" in prompt_answer:
    prompt_answer = prompt_answer.replace("```json", "").replace("```", "")
    '''
    try:
        prompt_answer = json.loads(prompt_answer)
        return prompt_answer
    except Exception as error:
        logger.error(f"Error in cleanupanswer: {error}")
        return None
    '''
    return prompt_answer

def generate_reference_number(length=6):
    characters = string.ascii_uppercase + string.digits  # Uppercase letters and digits
    random_string = ''.join(random.choices(characters, k=length))
    return random_string