from typing import Dict, List
import json

import logging

from dialogue_systems.basedsystem import DialogueSystem
from dialogue_systems.cetaldsys.players import CetalSpeaker
from dialogue_systems.cetaldsys.e2e_utils import E2E_InstrucTOD
from dialogue_systems.cetaldsys.config import CONFIG
from dialogue_systems.cetaldsys.modelargs import ModelArguments
from dialogue_systems.cetaldsys.dataargs import DataArguments
from dialogue_systems.cetaldsys.mwozdata import MWOZ_Dataset

logger = logging.getLogger(__name__)

class CETALDialogueSystem(DialogueSystem):
    """A neural network-based dialogue system implementation."""
    def __init__(self, model_name, model_spec, db_path: str, dialogue_domains: List, **kwargs):
        """Rule-based systems may not need extra parameters."""
        super().__init__(**kwargs)
        self.model_name = model_name
        self.db_path = db_path
        self.dialogue_domains = dialogue_domains

        self.model_args = ModelArguments()
        self.data_args = DataArguments()

        self.model_args.model_name_or_path_agent = self.model_name
        self.data_args.mwoz_path = self.db_path#"games/todsystem/dialogue_systems/data/multiwoz/" #+ self.db_path
        self.data_args.dataset_name = "multiwoz"
        self.data_args.ontology_path = self.data_args.mwoz_path + "/ontology.json"
        self.data_args.single_domain_only = False
        self.data_args.dialog_history_limit_dst = 0
        self.data_args.dialog_history_limit_e2e = -1
        self.data_args.dialog_history_limit_rg = -1
        self.data_args.with_slot_domain_diff = False
        self.data_args.with_all_slots = False
        self.data_args.with_slot_description = False        

        mwoz = MWOZ_Dataset(CONFIG, self.data_args)
        dataset = mwoz.dataset

        llm_player = CetalSpeaker(model_spec, "tod_llm", "", {})
        llm_agent = CetalSpeaker(model_spec, "agent_llm", "", {})
        player_dict = {"llm_player": llm_player, "llm_agent": llm_agent}

        self.e2einstructod = E2E_InstrucTOD(CONFIG, self.model_args, self.data_args,
                                            dataset, player_dict, self.dialogue_domains)
        self.dialogue_context = ""
        self.preds = []
        logger.info("CETALDialogueSystem initialized.")


    def process_user_input(self, utterance: str, current_turn: int) -> str:
        logger.info(f"Processing user input: Utterance = {utterance}")
        self.dialogue_context += "USER: " + utterance
        logger.info(f"Dialogue Context = {self.dialogue_context}")
        dsyslogs, response, response = self.e2einstructod.process_user_input(self.dialogue_context, utterance, current_turn)

        logger.info(f"DSystem Response: {response}")
        self.preds.append(response)
        self.dialogue_context += "\nSYSTEM: " + response
        #result = {"status": "follow-up", "details": response}
        #dialogue_context = response#self.dialogue_context.split("\n")[-1].replace("SYSTEM: ", "")
        return dsyslogs, response, response
    
    def get_booking_data(self) -> Dict:
        """Returns generated slots."""
        return self.e2einstructod.get_booking_data(self.preds)
    

    def inference(self, input_idx):
        return self.e2einstructod.inference(input_idx)

if __name__ == "__main__":
    cdsys = CETALDialogueSystem(model_id="gpt-4o-2024-08-06", dialogue_id="PMUL4648.json", dialogue_domains=["restaurant", "attraction", "taxi"])

    user_messages = ["You are traveling to Cambridge and looking forward to try local restaurants",
      "You are looking for a particular attraction. Its name is called nusha",
      "Make sure you get and address",
      "You are also looking for a place to dine. The restaurant should be in the expensive price range and should serve food",
      "The restaurant should be in the centre",
      "Make sure you get address"]
    

    #for user_message in user_messages:
    #    _, _, result = cdsys.process_user_input(user_message)
    #    print(result["details"])

    cdsys.get_booking_data()
  
    