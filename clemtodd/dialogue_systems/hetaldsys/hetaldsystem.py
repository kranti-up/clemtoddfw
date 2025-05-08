from typing import Dict
from dialogue_systems.basedsystem import DialogueSystem
from dialogue_systems.hetaldsys.interact import Interact
from dialogue_systems.hetaldsys.players import HetalSpeaker

class HETALDialogueSystem(DialogueSystem):
    """A neural network-based dialogue system implementation."""

    def __init__(self, model_name, model_spec, db_path, **kwargs):
        super().__init__(**kwargs)

        domain_player = HetalSpeaker(model_spec, "domain_detection", "", {})
        state_tracker = HetalSpeaker(model_spec, "state_tracker", "", {})
        response_generator = HetalSpeaker(model_spec, "response_generator", "", {})
        player_dict = {"domain_detection": domain_player, "state_tracker": state_tracker, "response_generator": response_generator}

        self.interact = Interact(model_name, player_dict, db_path)

    def process_user_input(self, user_input: str, current_turn: int) -> str:
        return self.interact.run(user_input, current_turn)
    
    def get_booking_data(self) -> Dict:
        """Returns generated slots."""
        return self.interact.getgenslots()