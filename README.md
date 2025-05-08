# clemtoddfw
clemtodd is developed based on [clembench](https://github.com/clp-research/clemcore) framework to evaluate dialogue systems under similar constraints.
* In-order to interact with the game master, the integrated dialogue system should respond in the following format
   - {'status': 'ds-status', 'details': 'ds-details'}
      * Possible values for 'ds-status': followup, querydb and validatebooking
      * Possible values for 'ds-details': A string indicating the follow-up message or a key-value pair indicating the DB/Booking requests