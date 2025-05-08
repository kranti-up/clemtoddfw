from typing import Dict, List
from dialogue_systems.basedsystem import DialogueSystem
from dialogue_systems.monolithicsys.monollm import MonoLLM
from dialogue_systems.monolithicsys.players import MonoLLMSpeaker

class MONODialogueSystem(DialogueSystem):
    """A neural network-based dialogue system implementation."""

    def __init__(self, model_name, model_spec, prompts_dict, resp_json_schema, **kwargs):
        super().__init__(**kwargs)

        self.monollm_player = MonoLLMSpeaker(model_spec, "monolithic_llm", "", {})
        player_dict = {"monollm_player": self.monollm_player}

        self.monollm = MonoLLM(model_name, prompts_dict, player_dict, resp_json_schema)

    def process_user_input(self, user_input: str, current_turn: int) -> str:
        return self.monollm.run(user_input, current_turn)
    
    def get_booking_data(self) -> Dict:
        """Returns generated slots."""
        return self.monollm.get_booking_data()


    def get_player_prompt(self) -> List:
        return self.monollm_player.history
