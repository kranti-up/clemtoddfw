import random
import copy
from typing import List, Dict, Tuple
from string import ascii_lowercase as letters
import importlib

import numpy as np
import json

import clemcore.clemgame.metrics as ms
from clemcore.clemgame import GameMaster, GameBenchmark, GameScorer, GameSpec
from clemcore.backends import Model
#from clemcore import get_logger

from instancegenerator import GAME_NAME
from players import LLMSpeaker
from dialogue_systems.factory import get_dialogue_system
from computemetrics import ComputeMetrics
from gamevalidator import GameValidator
from dbquerybuilder import DBQueryBuilder
from utils import processgtslots
from processbooking import ProcessBooking


# use the framework logger to log events relevant at runtime;
# this is independent from the game records / transcript
import logging

logger = logging.getLogger(__name__)


global DSYSTEM_ERROR_MESSAGE


class DMSystemMaster(GameMaster):
    def __init__(
        self,
        gamename: str,
        game_path: str,
        experiment: Dict,
        player_backends: List[str],
        other_modules: classmethod,
    ):
        super().__init__(gamename, game_path, experiment, player_backends)
        # save experiment and player attributes that will be necessary later
        self.topic = experiment["name"]
        self.model_a = player_backends[0]
        self.model_b = player_backends[1]
        self.human_input = ["gradio", "slurk"]  # Add the human input backends here

        # initialise attributes that will be used for the evaluation scores
        self.aborted: bool = False
        self.lose: bool = False
        self.success: bool = False
        self.complete_turns: int = 0

        if other_modules:
            self.other_modules = other_modules

    def setup(self, data: Dict, game_id: int) -> None:
        """Setup the episode (mandatory)."""

        #logging.disable(logging.CRITICAL) 

        self._setcommonsetup(data, game_id)
        self._setgamespecificsetup(data, game_id)

        # add initial prompts to each player's messages
        self.initiate(self.prompt_player_a, self.prompt_player_b)

        # always log the details of the players in this format (see logdoc)
        self.log_players(
            {
                "GM": "Game master for DMSystem",
                "Player 1": f"Player A: {self.model_a}",
                "Player 2": f"Player B: {self.model_b}",
            }
        )

        # log any additional keys that will be relevant for evaluation
        self.log_key("n_turns", self.n_turns)

    def _setcommonsetup(self, data: Dict, game_id: int) -> None:
        self.game_id = game_id
        self.instancedata = data
        self.dfilename = data["filename"]
        self.n_turns = data["n_turns"]
        self.goal = data["message"]
        self.domains = list(data["domains"].keys())
        self.tsystem = data["tsystem"]
        self.tasktype = data["tasktype"]
        self.statusmsg = data["statusmsg"]
        self.db_path = data["db_path"]
        self.domaindbkeys = data["domaindbkeys"]
        self.booking_mandatory_keys = data["booking_mandatory_keys"]
        self.json_schema = data["json_schema"]

        self.gamedata = None
        self.slots_gen = None
        self.slots_gen_loss = None
        self.misses = None
        self.slots_gt_raw = data["domains"]
        self.slots_gt = processgtslots(self.slots_gt_raw)
        self.booking_slots = data["book_keys"]
        self.gen_dialogue = []

        logger.info(f"User Goal: {self.goal}")
        logger.info(f"Ground Truth Slots: {self.slots_gt}")
        self.gamevalidator = GameValidator(self.tsystem, self.slots_gt)

        # initialise game variables
        self.current_turn: int = 0

        # initialise common metrics
        self.request_counts = [0] * (self.n_turns + 1)
        self.parsed_request_counts = [0] * (self.n_turns + 1)
        self.violated_request_counts = [0] * (self.n_turns + 1)

    def _process_gt_slots(self, slots: Dict) -> Dict:
        pass

    def _save_prompts(self, tsystem, promptsdict):
        self.prompt_player_a = promptsdict["prompt_a"]
        self.turn_prompt_player_a = promptsdict["turn_prompt_a"]

        if "prompt_b" in promptsdict:
            self.prompt_player_b = promptsdict["prompt_a"]
        else:
            self.prompt_player_b = None

    def _setgamespecificsetup(self, data: Dict, game_id: int) -> None:
        # instantiate both players

        self._save_prompts(data["tsystem"], data["prompts"])

        self.player_a = LLMSpeaker(self.model_a, "A", self.goal, self.slots_gt)
        self.player_b = LLMSpeaker(self.model_b, "B", None, None)

        # instantiate the DB query builder
        self.dbquery = DBQueryBuilder(
            self.domains,
            data["mwozschema"],
            self.db_path,
            self.statusmsg,
        )

        dialogue_params = {
            "model_name": self.model_b.model_spec["model_name"],
            "model_spec": self.model_b,
            "db_path": self.db_path,
            "dialogue_domains": self.domains,
            "prompts_dict": data["prompts"],
            "resp_json_schema": self.json_schema,
            "liberal_processing": data["liberal_processing"],
            "booking_mandatory_keys": self.booking_mandatory_keys,
        }
        self.dsystem = get_dialogue_system(data["tsystem"], **dialogue_params)

        self.bookingsystem = None
        if data["tsystem"] in ["monolithic_llm", "modular_llm", "modular_prog"]:
            gamedata = {
                "dbquery": self.dbquery,
                "domaindbkeys": self.domaindbkeys,
                "booking_mandatory_keys": self.booking_mandatory_keys,
                "booking_slots": self.booking_slots,
                "statusmsg": self.statusmsg,
            }
            self.bookingsystem = ProcessBooking(gamedata)

    def play(self) -> None:
        """Play the game until the end (mandatory)."""
        # play the game
        while self.proceed():
            self.current_turn += 1
            # always call log_next_turn when a new turn starts
            self.log_next_turn()
            self.turn()

        if self.complete_turns == self.n_turns:
            if not self.success:
                self.lose = True
                # TODO: Extracting the generated slots from the dialogue system
                if self.tsystem not in [
                    "monolithic_llm",
                    "modular_llm",
                    "modular_prog",
                ]:
                    self.slots_gen_loss = self.dsystem.get_booking_data()

        self.gamedata = {
            "filename": self.dfilename,
            "goal": self.goal,
            "domains": self.domains,
            "tsystem": self.tsystem,
            "tasktype": self.tasktype,
            "dialogue_type": self.instancedata["dialogue_type"],
            "slots_gt": self.slots_gt,
            "slots_gen": self.slots_gen,
            "slots_gt_raw": self.slots_gt_raw,
            "slots_gen_loss": self.slots_gen_loss,
            "n_turns": self.n_turns,
            "play_turns": self.current_turn,
            "instancedomains": self.instancedata["domains"],
            "corpususer": self.instancedata["corpususer"],
            "gendialogue": self.gen_dialogue,
        }

        # if self.complete_turns == self.n_turns:
        # log a message informing that the game was successfuly played
        if not self.aborted:
            gt_slots = "Ground truth slots:\n" + json.dumps(self.slots_gt)
            gen_slots = "Generated slots:\n" + json.dumps(self.slots_gen)

            action = {
                "type": "info",
                "content": f"{gt_slots}\n{gen_slots}",
            }
            self.log_event(from_="GM", to="GM", action=action)

        if self.success:
            action = {
                "type": "info",
                "content": "The game is successful; all the generated data matched.",
            }
        elif self.lose:
            if self.misses:
                action = {
                    "type": "info",
                    "content": "The game is lost because of a mismatch in the generated data."
                    + "\n"
                    + json.dumps(self.misses),
                }
            elif self.complete_turns == self.n_turns:
                action = {
                    "type": "info",
                    "content": f"The game is lost as the maximum number of turns ({self.n_turns}) was reached",
                }
            else:
                action = {
                    "type": "info",
                    "content": f"lost game",
                }
        elif self.aborted:
            action = {
                "type": "info",
                "content": "The game has been aborted due to an invalid input.",
            }
        else:
            action = {
                "type": "info",
                "content": f"The game was lost after ({self.n_turns}) turns.",
            }
        self.log_event(from_="GM", to="GM", action=action)

        # Log the prompt of the player to understand the flow
        #action = {"type": "player prompt", "content": self.dsystem.get_player_prompt()}
        #self.log_event(from_="GM", to="GM", action=action)


        # log a final message saying that the game did came to an end
        action = {"type": "info", "content": "end game"}
        self.log_event(from_="GM", to="GM", action=action)
        # log all temporary game variables that are needed for evaluation
        self.log_eval_assets()

    def initiate(self, prompt_player_a: str, prompt_player_b: str) -> None:
        """Initialise the dialogue history (firstlast specific)."""
        # always call log_next_turn what a turn starts
        self.log_next_turn()

        # append the initial message of each player to their history
        # the value user means the message is from an interlocutor of the model
        if self.model_a.model_spec["model_name"] in self.human_input:
            user_welcome = self.statusmsg["welcome"].replace("$domain", self.domain)
            self.player_a.history.append({"role": "user", "content": user_welcome})
        else:
            self.player_a.history.append({"role": "user", "content": prompt_player_a})

        self.player_b.history.append({"role": "user", "content": ""})

        # also log the messages as events for the transcriptions
        action = {"type": "send message", "content": prompt_player_a}
        self.log_event(from_="GM", to="Player 1", action=action)
        
        if self.tsystem in ["monolithic_llm", "modular_llm"]:
            action = {'type': 'send message', 'content': prompt_player_b}
            self.log_event(from_='GM', to='Player 2', action=action)

    def proceed(self) -> None:
        """Check if the game loop should continue (dmsystem specific)."""
        return (
            self.current_turn < self.n_turns
            and not self.aborted
            and not self.lose
            and not self.success
        )

    def turn(self) -> None:
        """Perform a game turn, utterances by A and B (firstlast specific)."""
        global DSYSTEM_ERROR_MESSAGE
        # TODO: Where to add violated requests?
        # get player A's reply and add it to its history
        status, response, _ = self._triggerplayer("a")

        if status is None:
            # stop game
            return None

        if self.success or self.lose or self.aborted:
            return None

        # add A's reply to B's history
        logger.info(f"Appended Player A answer to PlayerB\n{response}")
        self._append_utterance(response, "b", "user")
        self.gen_dialogue.append({"user": response})
        # also add the reply to the transcript
        action = {
            "type": "send message",
            "content": self.player_b.history[-1]["content"],
        }
        self.log_event(from_="GM", to="Player 2", action=action)

        # now do the same for player B
        status, response, details = self._triggerplayer("b")
        if status is None:
            # stop game
            return None

        if self.tsystem not in ["monolithic_llm", "modular_llm", "modular_prog"]:
            if self.lose or self.aborted:
                return None

            # add B's reply to A's history
            logger.info(f"Appended Player B answer to PlayerA\n{details}")
            self._append_utterance(details, "a", "user")
            self.gen_dialogue[-1].update({"system": details})
            # also add the reply to the transcript
            action = {
                "type": "send message",
                "content": self.player_a.history[-1]["content"],
            }
            self.log_event(from_="GM", to="Player 1", action=action)
        else:
            while True:
                if response == "follow-up":
                    self._handleplayerb_response(details)
                    break

                elif response == "db-query":
                    # handle db query
                    status, response, details = self._handle_dbquery(details)
                    if status is None:
                        # stop game
                        self.aborted = True
                        # log the abortion event
                        # action = {"type": "invalid format", "content": "game aborted due to an issue in dbquery input"}
                        action = {
                            "type": "invalid format",
                            "content": DSYSTEM_ERROR_MESSAGE,
                        }
                        self.log_event(from_="GM", to="GM", action=action)
                        # increase the counter of requests that violate form rules
                        self.violated_request_counts[self.current_turn] += 1
                        return None

                elif response == "validate-booking":
                    # handle booking
                    status, response, details = self._handle_booking(details)
                    if status is None:
                        # stop game
                        self.aborted = True
                        # log the abortion event
                        # action = {"type": "invalid format", "content": "game aborted due to an issue in booking system input"}
                        action = {
                            "type": "invalid format",
                            "content": DSYSTEM_ERROR_MESSAGE,
                        }
                        self.log_event(from_="GM", to="GM", action=action)
                        # increase the counter of requests that violate form rules
                        self.violated_request_counts[self.current_turn] += 1
                        return None

        self.complete_turns += 1

    def _get_utterance(self, player: str) -> str:
        """Get utterance from a player and log it (firstlast specific)."""
        assert player in ("a", "b")
        if player == "a":
            if (
                self.current_turn == 1
                and self.model_a.model_spec["model_name"] in self.human_input
            ):
                usergoal = self.statusmsg["usergoal"].replace("$goal", self.goal)
                usergoal += (
                    "\n"
                    + "Additional information:\nbookday: Friday, people: 4, time: 14:15,\nfood: chinese, price: cheap, name: charlie chan\n"
                )
                self.player_a.history[-1]["content"] += "\n\n" + usergoal
            # make an API call (or get a programmatic response) from player a
            prompt, raw_answer, answer = self.player_a(
                self.player_a.history, self.current_turn, None, None
            )
            # add API call to the records
            action = {"type": "get message", "content": answer}
            self.log_event(
                from_="Player 1",
                to="GM",
                action=action,
                call=(copy.deepcopy(prompt), raw_answer),
            )
            # add reply to its own memory
            self._append_utterance(answer, "a", "assistant")

        else:
            # make an API call (or get a programmatic response) from player b
            promptlogs, raw_answer, answer_utter = self.dsystem.process_user_input(
                self.player_b.history[-1]["content"], self.current_turn
            )

            for turn in promptlogs:
                if isinstance(turn["content"], str):
                    action = {
                        "type": "info",
                        "content": turn["content"],
                    }
                    self.log_event(from_="Player 2", to="Player 2", action=action)
                else:
                    prompt = turn["content"]["prompt"]
                    raw_answer = turn["content"]["raw_answer"]
                    # TODO: This answer is added to Player A, check if this is correct or should it be answer_utter?
                    # May be in the last turn, both are equal -> But check the transcripts again
                    answer = turn["content"]["answer"]
                    action = {"type": "get message", "content": answer}
                    if self.tsystem == "monolithic_llm":
                        logto = "GM"
                    else:
                        logto = "Player 2"
                    self.log_event(
                        from_="Player 2",
                        to=logto,
                        action=action,
                        call=(copy.deepcopy(prompt), raw_answer),
                    )
            answer = answer_utter
            # add reply to its own memory
            self._append_utterance(answer_utter, "b", "assistant")

            # add API call to the records
            if self.tsystem != "monolithic_llm":
                action = {"type": "get message", "content": answer_utter}
                self.log_event(
                    from_="Player 2",
                    to="GM",
                    action=action,
                    # call=(copy.deepcopy(prompt), raw_answer),
                )

        # increase the number of API requests
        # TODO: For Modular DM, there are multiple calls to the sub-modules which are not counted here
        self.request_counts[self.current_turn] += 1
        return answer




    def _append_utterance(self, utterance: str, player: str, role: str) -> None:
        """Add an utterance to the history of a player (firstlast specific)."""
        assert player in ("a", "b")
        logger.info(f"Player {player}: {type(utterance)}, {utterance} current turn = {self.current_turn}")

        if isinstance(utterance, dict):
            b_response = json.dumps(utterance)
        else:
            b_response = utterance

        if player == "a":
            if role == "assistant":
                self.player_a.history.append({"role": role, "content": b_response})
            else:
                content = self.statusmsg["dmresponse"].replace("$response", "\n" + b_response) if self.model_a.model_spec["model_name"] in self.human_input else self.turn_prompt_player_a + "\n\n" + b_response
                self.player_a.history.append({"role": role, "content": content})
        else:
            if role == "assistant":
                self.player_b.history.append({"role": role, "content": b_response})
            else:
                if len(self.player_b.history) == 1:
                    if self.tsystem not in ["modular_prog"]:
                        self.player_b.history[-1]["content"] = b_response
                    else:
                        self.player_b.history[-1]["content"] = f"USER REQUEST: {b_response}" if self.tsystem in ["modular_prog"] else self.player_b.history[-1]["content"] + "\n\n" + b_response
                        self.player_b.history[-1]["content"] = self.player_b.history[-1]["content"].strip()
                else:
                    turn_prompt = ""
                    if role == "user":
                        turn_prompt = "USER REQUEST:" if self.tsystem in ["modular_prog"] else ""
                    elif role == "db-query":
                        turn_prompt = "DATABASE RETRIEVAL RESULTS:"
                    elif role == "validate-booking":
                        turn_prompt = "BOOKING VALIDATION STATUS:"

                    self.player_b.history.append({"role": "user", "content": (turn_prompt + "\n\n" + b_response).strip()})
                logger.info(f"Player B: {self.player_b.history[-1]}")

    def _isvalidturn(self, player: str, answer: str) -> bool:
        """Check if answer is valid and correct (firstlast specific)."""
        # parse answer
        global DSYSTEM_ERROR_MESSAGE
        response, details = self.parse(player, answer)
        logger.info(
            f"_isvalidturn(): player: {player}, response: {response}, details = {details}"
        )

        if response is None or (
            player == "b"
            and self.tsystem in ["monolithic_llm", "modular_prog", "modular_llm"]
            and details is None
        ):
            self.aborted = True
            # log the abortion event
            # action = {"type": "invalid format", "content": "game aborted due to an issue in parsing the response"}
            action = {"type": "invalid format", "content": DSYSTEM_ERROR_MESSAGE}
            self.log_event(from_="GM", to="GM", action=action)
            # increase the counter of requests that violate form rules
            self.violated_request_counts[self.current_turn] += 1
            return False, response, details

        # increase the counter of requests that conform to form rules
        self.parsed_request_counts[self.current_turn] += 1
        # log the event that the string was valid (no strange characters)
        action = {"type": "metadata", "content": "response confirms to rules"}
        self.log_event(from_="GM", to="GM", action=action)

        return True, response, details

    def _process_player_response(self, player: str, response: str) -> None:
        if player == "a":
            #if "done" in response.lower():
            if response.lower().strip() in ["done", "done,", "done."]:
                response = response.lower().strip()
                # if response in ["done", "done,", "done."]:
                # Being liberal in parsing the response
                action = {"type": "parse", "content": f"received done"}
                self.log_event(from_="GM", to="GM", action=action)

                # Compare the slots of the ground truth and the generated slots
                # This comparison is done in handle_booking() method
                if self.tsystem not in [
                    "monolithic_llm",
                    "modular_llm",
                    "modular_prog",
                ]:
                    self.slots_gen = self.dsystem.get_booking_data()
                logger.info(f"Generated slots in Player2: {self.slots_gen}")

                status, missed_values = self.gamevalidator.run(self.slots_gen)
                if status:
                    self.success = True
                else:
                    self.lose = True
                    self.misses = missed_values
                """
                else:
                    self.aborted = True
                    # log the abortion event
                    action = {"type": "invalid format", "content": "abort"}
                    self.log_event(from_="GM", to="GM", action=action)
                    # increase the counter of requests that violate form rules
                    self.violated_request_counts[self.current_turn] += 1
                """
        else:
            pass

    def _triggerplayer(self, player: str) -> Tuple[bool, str, str]:
        answer = self._get_utterance(player)
        # print(f"3. Player B answer\n{answer_b}")
        logger.info(f"Player-{player} answer\n{answer} {type(answer)}")
        is_valid_turn, response, details = self._isvalidturn(player, answer)

        logger.info(
            f"Player-{player} is_valid_turn: {is_valid_turn}, response: {response}, details: {details}"
        )
        if not is_valid_turn:
            # stop game
            return None, response, details

        self._process_player_response(player, response)
        return True, response, details

    def _handleplayerb_response(self, details) -> None:
        # add B's reply to A's history
        if self.current_turn < self.n_turns:
            logger.info(f"Appended Player B answer to PlayerA\n{details}")
            self._append_utterance(details, "a", "user")
            self.gen_dialogue[-1].update({"system": details})
            # also add the reply to the transcript
            action = {
                "type": "send message",
                "content": self.player_a.history[-1]["content"],
            }
            self.log_event(from_="GM", to="Player 1", action=action)

    def _handle_dbquery(self, details) -> bool:
        global DSYSTEM_ERROR_MESSAGE
        num_db_queries = 0
        while num_db_queries < self.n_turns and details is not None:
            if "domain" not in details:
                logger.error("Domain not found in the details")
                DSYSTEM_ERROR_MESSAGE = "Domain is not found in the details."
                message = f"Cannot trigger DB query as 'domain' key is not found in the response. Please stick to the names/keys mentioned in the schema."
            else:
                logger.info(
                    f"continuing with db query slots = {details} {type(details)}"
                )
                dbresult = self.dbquery.run(details)
                message = (
                    dbresult["status_response"] + json.dumps(dbresult["data"])
                    if dbresult["status"] == "success"
                    else f'{dbresult["error"]}'
                )

            self._append_utterance(message, "b", "db-query")
            action = {
                "type": "send message",
                "content": self.player_b.history[-1]["content"],
            }
            self.log_event(from_="GM", to="Player 2", action=action)
            status, response, details = self._triggerplayer("b")
            logger.info(f"status is {status}, response: {response} details: {details}")

            if status is None:
                DSYSTEM_ERROR_MESSAGE = (
                    "Failure in DB query."
                )
                return None, response, details

            if response in ["follow-up", "validate-booking"]:
                return status, response, details

            num_db_queries += 1

        DSYSTEM_ERROR_MESSAGE = (
            "DBQuery attempts exceeded the maximum number of attempts."
        )
        return None, None, None

    def _handle_booking(self, details) -> bool:
        if not isinstance(details, dict):
            return None, None, details

        global DSYSTEM_ERROR_MESSAGE
        num_booking_attempts = 0

        while num_booking_attempts < self.n_turns and details is not None:
            message, booking_status = self.bookingsystem.run(details)

            if booking_status:
                self.log_event(
                    from_="GM",
                    to="GM",
                    action={"type": "parse", "content": "booking completed"},
                )
                self.slots_gen = self.bookingsystem.get_booking_data()


            self._append_utterance(message, "b", "validate-booking")
            self.log_event(
                from_="GM",
                to="Player 2",
                action={
                    "type": "send message",
                    "content": self.player_b.history[-1]["content"],
                },
            )

            if not message:
                logger.error("Message is empty")
                return None, None, details

            status, response, details = self._triggerplayer("b")
            if status is None:
                DSYSTEM_ERROR_MESSAGE = (
                    "Failure in booking."
                )
                return None, response, details

            if response in ["follow-up", "db-query"]:
                return status, response, details

            num_booking_attempts += 1

        DSYSTEM_ERROR_MESSAGE = (
            "Booking attempts exceeded the maximum number of attempts."
        )
        return None, None, None

    @staticmethod
    def parse(player: str, utterance: str) -> bool:
        """Check if utterance is valid and return first/last tokens (firstlast specific)."""
        # TODO: Any other logic required?
        global DSYSTEM_ERROR_MESSAGE
        if player == "a":
            if not utterance:
                DSYSTEM_ERROR_MESSAGE = "Empty response from the user simulator"
                return None, None
            return utterance, None
        else:
            try:
                if isinstance(utterance, str):
                    result = json.loads(utterance)

                elif isinstance(utterance, dict):
                    result = utterance

                else:
                    DSYSTEM_ERROR_MESSAGE = "Invalid format of the response"
                    return None, None

                if "status" not in result or "details" not in result:
                    DSYSTEM_ERROR_MESSAGE = (
                        "Status or details key not found in the response"
                    )
                    return None, None

                if result["status"] not in [
                    "follow-up",
                    "db-query",
                    "validate-booking",
                ]:
                    DSYSTEM_ERROR_MESSAGE = "Invalid status found in the response"
                    return None, None

                if result["status"] in ["db-query", "validate-booking"]:
                    if not isinstance(result["details"], dict):
                        DSYSTEM_ERROR_MESSAGE = "Details should be a dictionary for status in: db-query and validate-booking"
                        return None, None

                    # if "domain" not in result["details"]:
                    #    DSYSTEM_ERROR_MESSAGE = "Domain key not found in the details"
                    #    return None, None

                return result["status"], result["details"]

            except Exception as e:
                logger.error(f"Error in parsing the response: {e}")
                DSYSTEM_ERROR_MESSAGE = "Invalid response format"
                return None, None

        return result, None

    def log_eval_assets(self) -> None:
        """Aux to log variables needed for scoring (firstlast specific)"""
        self.log_key("Played turns", self.current_turn)
        self.log_key("Complete turns", self.complete_turns)
        self.log_key(ms.METRIC_ABORTED, self.aborted)
        self.log_key(ms.METRIC_LOSE, self.lose)
        self.log_key(ms.METRIC_REQUEST_COUNT, self.request_counts)
        self.log_key(ms.METRIC_REQUEST_COUNT_PARSED, self.parsed_request_counts)
        self.log_key(ms.METRIC_REQUEST_COUNT_VIOLATED, self.violated_request_counts)
        self.log_key("Evaluation", self.gamedata)


class DMSystemScorer(GameScorer):
    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)
        self.cm = ComputeMetrics()

    def compute_scores(self, episode_interactions: Dict) -> None:
        played_turns = episode_interactions["Played turns"]
        complete_turns = episode_interactions["Complete turns"]
        # turn 0 was only the initial prompts, so we disregard it here
        reqs = episode_interactions[ms.METRIC_REQUEST_COUNT][1:]
        p_reqs = episode_interactions[ms.METRIC_REQUEST_COUNT_PARSED][1:]
        v_reqs = episode_interactions[ms.METRIC_REQUEST_COUNT_VIOLATED][1:]
        n_turns = len(reqs)

        for turn in range(0, played_turns):
            self.log_turn_score(turn, ms.METRIC_REQUEST_COUNT, reqs[turn])
            self.log_turn_score(turn, ms.METRIC_REQUEST_COUNT_PARSED, p_reqs[turn])
            self.log_turn_score(turn, ms.METRIC_REQUEST_COUNT_VIOLATED, v_reqs[turn])

        aborted = int(episode_interactions[ms.METRIC_ABORTED])
        lose = int(episode_interactions[ms.METRIC_LOSE]) if not aborted else 0
        success = 1 - lose if not aborted else 0
        bench_score = complete_turns / n_turns if not aborted else np.nan

        self.log_episode_score(ms.METRIC_ABORTED, aborted)
        self.log_episode_score(ms.METRIC_LOSE, lose)
        self.log_episode_score(ms.METRIC_SUCCESS, success)
        self.log_episode_score(ms.METRIC_REQUEST_COUNT, sum(reqs))
        self.log_episode_score(ms.METRIC_REQUEST_COUNT_PARSED, sum(p_reqs))
        self.log_episode_score(ms.METRIC_REQUEST_COUNT_VIOLATED, sum(v_reqs))
        self.log_episode_score(ms.METRIC_REQUEST_SUCCESS, sum(p_reqs) / sum(reqs))
        self.log_episode_score(ms.BENCH_SCORE, bench_score)

        self.log_episode_score("Played turns", played_turns)
        self.log_episode_score("Complete turns", complete_turns)
        self.log_episode_score("Max turns", n_turns)

        # correct_slots, total_slots, accuracy = self.cm.run(episode_interactions["Evaluation"])
        # self.log_episode_score("Correct Slots", correct_slots)
        # self.log_episode_score("Total Slots", total_slots)
        # self.log_episode_score("Slot Accuracy", accuracy)

        # results = episode_interactions["Evaluation"]
        # logger.error(f"Eval Results: {results}")
        # with open("/home/admin/Desktop/codebase/cocobots/clembenchfork_dm_code/clembench/reval/results.json", "w") as f:
        #    json.dump(results, f)
        """
        accuracy, partialacc, missdata = self.cm.run(results)

        self.log_episode_score("accuracy", accuracy)
        self.log_episode_score("partial_accuracy", partialacc)
        self.log_episode_score("missdata", missdata)
        """


class DMSystemBenchmark(GameBenchmark):
    """Integrate the game into the benchmark run."""

    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    # defines whether the game is single player or not
    def is_single_player(self):
        return False

    # add a description of your game
    def get_description(self):
        return (
            "A simple game in which a human collaborate with a bot to complete a task."
        )

    # copy this, replacing the name of the game master in the return statement
    def create_game_master(
        self, experiment: Dict, player_models: List[Model]
    ) -> GameMaster:
        return DMSystemMaster(self.game_name, self.game_path, experiment, player_models, None)

    def create_game_scorer(self, experiment_config, game_instance) -> GameScorer:
        return DMSystemScorer(self.game_name, experiment_config, game_instance)
