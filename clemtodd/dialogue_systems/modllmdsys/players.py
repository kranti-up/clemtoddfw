import random
from typing import List
import json


#from clemgame import get_logger
from clemcore.clemgame import Player

import logging

logger = logging.getLogger(__name__)

class ModLLMSpeaker(Player):
    def __init__(self, model_name: str, player: str, task: str, slots: dict):
        # always initialise the Player class with the model_name argument
        # if the player is a program and you don't want to make API calls to
        # LLMS, use model_name="programmatic"
        super().__init__(model_name)

        self.player: str = player
        self.task = task
        self.slots = slots

        # a list to keep the dialogue history
        self.history: List = []

        self.human_input = None
        if model_name == "gradio":
            self.mode = "gradio"
            #self._launch_gradio()
        else:
            self.mode = "programmatic"

    # implement this method as you prefer, with these same arguments
    def _custom_response(self, messages, turn_idx, respformat) -> str:
        """Return a mock message with the suitable letter and format."""
        if self.mode  == "programmatic":
            slotsdict = dict.fromkeys(self.slots, '')
            if self.player == 'A':
                return self.task if turn_idx == 1 else "DONE"# + json.dumps({"slots": slotsdict})
            else:
                return json.dumps({"status": "booking-confirmed", "details": slotsdict})