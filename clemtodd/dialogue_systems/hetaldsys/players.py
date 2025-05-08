import random
from typing import List
import json

from clemcore.clemgame import Player


class HetalSpeaker(Player):
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

        self.cursystem = None

    # implement this method as you prefer, with these same arguments
    def _custom_response(self, messages, turn_idx) -> str:
        return "Mock response"