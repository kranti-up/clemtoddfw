from typing import Dict

from dialogue_systems.basedsystem import DialogueSystem
from dialogue_systems.xuetaldsys.engine import EngineDSystem
from dialogue_systems.xuetaldsys.players import XuetalSpeaker

class XUETALDialogueSystem(DialogueSystem):
    """A neural network-based dialogue system implementation."""
    def __init__(self, model_name, model_spec, db_path,  **kwargs):
        """Rule-based systems may not need extra parameters."""
        super().__init__(**kwargs)
        self.model_name = model_name
        self.db_path = db_path

        llm_player = XuetalSpeaker(model_spec, "tod_llm", "", {})
        player_dict = {"llm_player": llm_player}


        self.engine = EngineDSystem(model_name, player_dict, db_path, "func")


    def process_user_input(self, user_input: str, current_turn: int) -> str:
        return self.engine.run(user_input, current_turn)
    
    def get_booking_data(self) -> Dict:
        """Returns generated slots."""
        return self.engine.get_booking_data()