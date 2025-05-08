from typing import List, Dict
from dialogue_systems.basedsystem import DialogueSystem
from dialogue_systems.modprogdsys.modprogllm import ModProgLLM
from dialogue_systems.modprogdsys.players import ModProgLLMSpeaker

class MODULARPROGDialogueSystem(DialogueSystem):
    """A neural network-based dialogue system implementation."""

    def __init__(self, model_name, model_spec, prompts_dict, resp_json_schema, liberal_processing, booking_mandatory_keys, **kwargs):
        super().__init__(**kwargs)

        self.modllm_player = ModProgLLMSpeaker(model_spec, "modular_prog", "", {})
        player_dict = {}#{"monollm_player": monollm_player}

        self.modprogllm = ModProgLLM(model_name, model_spec, prompts_dict, player_dict, resp_json_schema, liberal_processing, booking_mandatory_keys)

    def process_user_input(self, user_input: str, current_turn: int) -> str:
        return self.modprogllm.run(user_input, current_turn)
    
    def get_booking_data(self) -> Dict:
        """Returns generated slots."""
        return self.modprogllm.get_booking_data()

    def get_player_prompt(self) -> List:
        return self.modllm_player.history
