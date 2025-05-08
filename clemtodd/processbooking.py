from typing import Dict
#from clemgame import get_logger
from utils import generate_reference_number


import logging

logger = logging.getLogger(__name__)


class ProcessBooking:
    def __init__(self, gamedata: Dict):
        self.gamedata = gamedata
        self.slots_gen = None
        self.statusmsg = gamedata["statusmsg"]
        self.domaindbkeys = gamedata["domaindbkeys"]
        self.booking_slots = gamedata["booking_slots"]
        self.booking_mandatory_keys = gamedata["booking_mandatory_keys"]
        self.dbquery = gamedata["dbquery"]

    def _validate_gen_book_slots(self, base_slots, gen_slots):
        message = ""
        missing_slots = [slot for slot in base_slots if slot not in gen_slots]

        if missing_slots:
            logger.info(f"Missing slots: {missing_slots}")
            missing_slots_info = self.statusmsg["missing_slots"].replace(
                "$slots", ", ".join(missing_slots)
            )
            message = (
                "Missed some slots: "
                + missing_slots_info
                + ". Please stick to the names/keys mentioned in the schema."
            )

        return message, missing_slots

    def _prepare_matching_db_slots_from_booking(
        self, gen_domain, booking_details, booking_slots_gen, db_match_keys
    ):
        db_match_slots = list(set(booking_slots_gen).intersection(db_match_keys))
        logger.info(
            f"Preparing DB Query for the DB Match db_match_slots = {db_match_slots} domain_db_keys: {db_match_keys}"
        )

        message = ""
        if not db_match_slots:
            logger.error("No DB Match slots found in the booking information")
            invalid_value_info = self.statusmsg["missing_other_slots_booking"].replace(
                "$slots", ", ".join(db_match_keys)
            )
            message = invalid_value_info
            return message, None, None

        else:
            missing_values = []
            db_query_data = {"domain": gen_domain}
            for slot in db_match_slots:
                slot_allowed_values = self.dbquery.get_valid_values(gen_domain, slot)
                if slot_allowed_values:
                    if booking_slots_gen[slot] not in slot_allowed_values:
                        missing_values.append(slot)
                    else:
                        # Using details instead of booking_slots_gen as details contains correct operators and DB query requires them
                        # db_query_data.update({slot: booking_slots_gen[slot]})
                        db_query_data.update({slot: booking_details[slot]})
                else:
                    # Using details instead of booking_slots_gen as details contains correct operators and DB query requires them
                    # db_query_data.update({slot: booking_slots_gen[slot]})
                    db_query_data.update({slot: booking_details[slot]})
            logger.info(f"preparing data for db query: message: {message}, missing_values: {missing_values}, db_query_data: {db_query_data}")
            return message, missing_values, db_query_data
        
    def _savegenerateddata(self, gendomain, genslots, db_match_keys, only_book_keys):
        if self.slots_gen is None:
            self.slots_gen = {}

        if gendomain not in self.slots_gen:
            self.slots_gen[gendomain] = {}

        for key, value in genslots.items():
            if key in db_match_keys:
                if "info" not in self.slots_gen[gendomain]:
                    self.slots_gen[gendomain]["info"] = {}
                self.slots_gen[gendomain]["info"].update({key: value})
            elif key in only_book_keys:
                if "book" not in self.slots_gen[gendomain]:
                    self.slots_gen[gendomain]["book"] = {}
                self.slots_gen[gendomain]["book"].update({key: value})
            else:
                if "reqt" not in self.slots_gen[gendomain]:
                    self.slots_gen[gendomain]["reqt"] = {}
                if key == "domain":
                    continue
                self.slots_gen[gendomain]["reqt"].update({key: value})

    def get_booking_data(self):
        return self.slots_gen        

    def _process_db_booking_query(
        self, db_query_data, gen_domain, booking_slots_gt, details
    ):
        booking_status = False
        dbquery_result = self.dbquery.run(db_query_data)
        logger.info(f"DB Query Result: {dbquery_result}")
        if dbquery_result["status"] == "success":
            if len(dbquery_result["data"]) == 1:
                # Fetch the booking reference number
                bookrefnum = generate_reference_number()
                refnumber = self.statusmsg["booking_reference"].replace(
                    "$refnum", bookrefnum
                )
                message = f"{self.statusmsg['success']} {self.statusmsg['validatebooking']}\n{refnumber}"
                self._savegenerateddata(
                    gen_domain, details, self.domaindbkeys[gen_domain], list(booking_slots_gt.keys())
                )
                booking_status = True
            else:
                message = self.statusmsg["multiple_entries"] + " " + self.statusmsg["booking_failure"]

        else:
            message = self.statusmsg["novaluematch"] + " " + self.statusmsg["booking_failure"]
            #missing_values = list(db_query_data.keys())
            #invalid_value_info = self.statusmsg["invalid_value"].replace(
            #    "$slot", ", ".join(missing_values)
            #)
            #message = invalid_value_info

        return message, booking_status

    def _prepare_gen_slots(self, details):
        booking_slots_gen = {}
        for k, v in details.items():
            if k == "domain":
                continue
            if isinstance(v, dict):
                v_dictvalues = list(v.values())
                booking_slots_gen[k.lower()] = str(v_dictvalues[1]).lower()

            else:
                booking_slots_gen.update({k.lower(): str(v).lower()})

        # booking_slots_gen = {k.lower(): str(v).lower() for k, v in details.items() if k != "domain" and v and not isinstance(v, (list, dict, tuple))}
        return booking_slots_gen

    def run(self, details: Dict) -> Dict:
        logger.info(f"Processing booking details {details}")

        booking_status = False
        if not isinstance(details, dict):
            return None, booking_status

        logger.info(f"received booking details: {details}, gt_slots: {self.booking_slots}")

        if "domain" not in details:
            logger.error("Domain not found in the details")
            DSYSTEM_ERROR_MESSAGE = "Domain key not found in the details"
            message = f"Cannot trigger DB query as 'domain' key is not found in the response. Please stick to the names/keys mentioned in the schema."
            return message, booking_status

        gen_domain = details["domain"].lower()
        """
        possible_domains = self.json_schema["schema"]["properties"]["details"]["oneOf"][2]["properties"]["domain"]["enum"]
        if gen_domain not in possible_domains or gen_domain not in self.booking_slots:
            logger.error(f"Domain {gen_domain} not found in the possible domains")
            #return None, None, details
            message = f"Extracted domain: {gen_domain} not found in the available domains"

        else:
        """
        #This scenario should not happen as the tasks are chosen in such a way that the booking information is always present
        if gen_domain not in self.booking_slots:
            logger.error(f"Domain {gen_domain} not found in the possible domains for booking")
            message = f"Booking is currently not supported for this domain '{gen_domain}', however you can gather the details for the same."
            return message, booking_status

        booking_slots_gt = self.booking_slots[gen_domain]
        booking_slots_gen = self._prepare_gen_slots(details)

        logger.info(
            f"booking_slots_gt: {booking_slots_gt}, booking_slots_gen: {booking_slots_gen}"
        )

        message = ""
        if gen_domain in self.booking_mandatory_keys:
            message, missing_slots = self._validate_gen_book_slots(
                self.booking_mandatory_keys[gen_domain], booking_slots_gen
            )

        if not missing_slots:
            message, missing_slots = self._validate_gen_book_slots(
                booking_slots_gt, booking_slots_gen
            )
            if not missing_slots:
                # Check if the values are valid
                logger.info(f"Checking if the values are valid")
                missing_values = []

                # Check if the values are present in the DB
                (
                    message,
                    missing_values,
                    db_query_data,
                ) = self._prepare_matching_db_slots_from_booking(
                    gen_domain, details, booking_slots_gen, self.domaindbkeys[gen_domain]
                )

                if missing_values:
                    logger.info(
                        f"Mismatches in configuration allowed for the slot values: {missing_values}"
                    )
                    invalid_value_info = self.statusmsg["invalid_value"].replace(
                        "$slot", ", ".join(missing_values)
                    )
                    message = invalid_value_info

                elif db_query_data:
                    # Make a DB Query and check if the value is present in the DB
                    logger.info(f"Making DB Query with the data {db_query_data}")
                    message, booking_status = self._process_db_booking_query(
                        db_query_data, gen_domain, booking_slots_gt, details
                    )

        return message, booking_status
