import copy
from typing import Dict, Any, List, Tuple
import json

#import clemgame
#from clemcore import get_logger
from dialogue_systems.modprogdsys.players import ModProgLLMSpeaker
from utils import cleanupanswer

import logging

logger = logging.getLogger(__name__)


class BookingAggregator:
    def __init__(self, model_name, model_spec, book_aggr_prompt):
        self.model_name = model_name
        self.model_spec = model_spec
        self.base_prompt = book_aggr_prompt
        self.player = None
        self._setup()

    def _setup(self) -> None:
        # If the model_name is of type "LLM", then we need to instantiate the player
        # TODO: Implement for other model types ("Custom Model", "RASA", etc.)
        self.player = ModProgLLMSpeaker(self.model_spec, "booking_aggregator", "", {})
        self.player.history.append({"role": "user", "content": self.base_prompt})

    def run(self, utterance: Dict, turn_idx: int) -> str:
        message = json.dumps(utterance) if isinstance(utterance, Dict) else utterance

        self.player.history[-1]["content"] += message

        prompt, raw_answer, answer = self.player(self.player.history, turn_idx, None, None)
        logger.info(f"Booking aggregator raw response:\n{answer}")
        return prompt, raw_answer, cleanupanswer(answer)
    
    def get_history(self):
        return self.player.history
    
    def clear_history(self):
        self.player.history = [{"role": "user", "content": self.base_prompt}]    
