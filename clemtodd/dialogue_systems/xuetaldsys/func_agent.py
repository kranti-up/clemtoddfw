import json
import re
import sqlite3
from functools import partial

import openai
#from langchain import SQLDatabase
from langchain_community.utilities import SQLDatabase
from langchain.schema import LLMResult
import tenacity

from dialogue_systems.xuetaldsys.booking import make_booking_db, make_booking_taxi
from dialogue_systems.xuetaldsys.utils import DB_PATH, tenacity_retry_log

import logging

logger = logging.getLogger(__name__)

GREEN_COLOR = '\u001b[1;32m'
MAG_COLOR = '\u001b[1;35m'
CYAN_COLOR = '\u001b[1;36m'
RESET_COLOR = '\u001b[0m'


system_prompt = '''You are an intelligent AI Assistant to help the user complete complex tasks. The task may contain several sub-tasks, and you first determines which sub-tasks are involved in the user's utterance, and then completes the user's request according to the instructions of the corresponding sub-tasks.

# Task Overall:

You specializes in travel guidance in Cambridge, and are able to find the venue according to the user's constraints and make reservations or book a train or taxi.

There are several sub-tasks, and each sub-task contains three parts: Task Description, Task Tools, and Task Logic.
- **Task Description** provides an overview of the task, including the constraints that will be used in searching for venues.
- **Task Functions** give the external functions that would be used to complete the task, such as querying a database or making a reservation.
- **Task Logic** introduces the general flow to complete the task, including how to respond to the user in various scenarios.

# Sub-task #1: Restaurant

## Task Description

The AI Assistant helps the user find a restaurant and/or make a reservation.
The user provides the constraints of the restaurant for searching, and then provides the reservation constraints.

The search constraints include:
1. area: the location of the restaurant.
2. price: the price range of the restaurant.
3. food: the food type or cuisine of the restaurant.
4. name: sometimes the user may directly say the name of restaurant.

The reservation constraints include:
1. people: the number of people.
2. day: the day when the people go in a week.
3. time: the time of the reservation.
The AI Assistant can only make a reservation if the restaurant name and people, day, time constraints are all clear.

## Task Functions

- query_restaurants: Use an SQL statement to query the restaurants in the database to find proper information.
- book_restaurant: Book a restaurant with certain requirements.

## Task Logic

- The user would provide some constraints to the AI Assistant to search for a restaurant.
- The AI Assistant can use the Restaurant Query tool to query restaurants that meet the constraints, and then recommend the restaurant names to the user for choosing.
- The user would also directly specify the name of the restaurant, and the AI assistant will query the database and tell the user the information of the restaurant.
- The AI Assistant can use the Restaurant Reservation tool to book a restaurant. Reservations can only be made if the restaurant name and all the reservation constraints (people, day, time) are specified.

# Sub-task #2: Hotel

## Task Description

The AI Assistant helps the user find a restaurant and/or make a reservation.
The user provides the constraints of the restaurant for searching, and then provides the reservation constraints.

The search constraints include:
1. area: the location of the hotel.
2. price: the price range of the hotel.
3. type: the type of the hotel.
4. parking: whether the hotel has free parking.
5. internet: whether the hotel has free internet/wifi.
6. stars: the star rating of the hotel.
7. name: sometimes the user may directly say the name of hotel.

The reservation constraints include:
1. people: the number of people.
2. day: the day when the people go.
3. stay: the number of days to stay.

The AI Assistant can only make a reservation if the restaurant name and people, day, time constraints are all clear.

## Task Functions

- query_hotels: Use an SQL statement to query the hotels in the database to find proper information.
- book_hotel: Book a hotel with certain requirements

## Task Logic

- The user would provide some constraints to the AI Assistant to search for a hotel.
- The AI Assistant can use the Hotel Query tool to query hotels that meet the constraints, and then recommend the hotel names to the user for choosing.
- If there are too many hotels, the AI Assistant could ask the user to provide more constraints.
- The user would also directly specify the name of the hotel, and the AI assistant will query the database and tell the user the information of the hotel.
- The AI Assistant can use the Hotel Reservation tool to book a hotel. Reservations can only be made if the hotel name and all the reservation constraints (people, day, stay) are specified.

# Remember

- Don't make assumptions about what values to plug into functions. Ask for clarification if any parameter is missing or ambiguous.
- You must not call the same function with the same parameters again and again.
- When finishing the user's goals, saying goodby to the user and finish the dialogue.
'''

dsys_logs = []


def prepare_query_db_functions(domain, db_path=DB_PATH):

    global dsys_logs

    def query_db(sql, table=None, db_path=DB_PATH):
        # if 'SELECT *' in sql:
        #     return "It's not allowed to use `SELECT *` to query all columns at the same time. You must query only the columns that are needed."
        if table is None:
            return 'Please provide the table name to query.', domain, None
        if table and table not in sql:
            return  f'Please query the {table} table in the database.', domain, None
        logger.info(f"Inside query_db: sql: {sql}, table: {table}, DB_at: {db_path}")

        #logger.info(f"Inside query_db: dsy_logs: {dsys_logs}")
        dsys_role = 'assistant' if dsys_logs[-1]['role'] == 'user' else 'user'
        dsys_logs.append({'role': dsys_role, 'content': f"DB Query: {sql}"})

        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.execute(sql)
        except Exception as e:
            return str(e), domain, None
        records = cursor.fetchall()

        logger.info(f"DB records: {records}")

        if len(records) == 0:
            #dsys_logs.append({'role': 'assistant', 'content': f"DB Query Result: No results found."})
            return 'No results found.', domain, None
        
        # Make result string
        max_items = 5
        max_chars = 500

        if domain == "restaurant":
            column_names = [desc[0] for desc in cursor.description if desc[0] not in ["id", "postcode", "signature", "introduction", "phone"]]
            column_values_ids = [0, 4, 6, 7, 9]
        else:
            column_names = [desc[0] for desc in cursor.description]
            column_values_ids = []

        

        result = []
        n_chars = 0
        line = '| ' + ' | '.join(column_names) + ' |'
        n_chars += len(line) + 1
        result.append(line)
        line = '| ' + ' | '.join(['---'] * len(column_names)) + ' |'
        n_chars += len(line) + 1
        result.append(line)
        for i, record in enumerate(records, start=1):
            line = '| ' + ' | '.join(str(v) for index, v in enumerate(record) if index not in column_values_ids) + ' |'
            n_chars += len(line) + 1
            if n_chars <= max_chars and i <= max_items:
                result.append(line)
            else:
                n_left = len(records) - i + 1
                result.append(f'\n{n_left} more records ...')
                break
        result = '\n'.join(result)

        pattern = r"(\w+)\s*(?:=|>=|<=|>|<)\s*'([^']*)'"

        # Find all matches
        details = dict(re.findall(pattern, sql))

        logger.info(f"Returning from query_db: result: {result} domain: {domain}, table: {table}, DB_at: {db_path}")
        #dsys_logs.append({'role': 'assistant', 'content': f"DB Query Result: {result}"})
        return result, domain, details


    def get_table_info(domain, db_path):
        logger.info(f"Getting table info for {domain} from db_path: {db_path} DB_PATH = {DB_PATH}")
        db = SQLDatabase.from_uri(
            database_uri=f'sqlite:///{DB_PATH}',
            include_tables=[domain],
            sample_rows_in_table_info=2,
        )
        table_info = db.get_table_info()
        return table_info
    
    def make_schema(domain, name, table_info):
        func_desc_temp = '''Use an SQL statement to query the {domain} table to get required information.

        Table Schema:
        {table_info}'''

        param_desc_temp = f'The SQL statement to query the {domain} table.'
        
        schema = {
            "type": "function",
            "function": {             
                'name': name,
                'description': func_desc_temp.format(table_info=table_info, domain=domain),
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'sql': {
                            'type': 'string',
                            'description': param_desc_temp.format(domain=domain),
                        }
                    },
                    'required': ['sql'],
                }
            }
        }
        return schema

    assert domain in ['restaurant', 'hotel', 'attraction', 'train']
    name = f'query_{domain}s'  # query_restaurants, query_hotels, query_attractions, query_trains
    function = partial(query_db, table=domain)
    table_info = get_table_info(domain, db_path)
    schema = make_schema(domain, name, table_info)

    return {'name': name, 'function': function, 'schema': schema}


def prepare_book_functions(domain):
    global dsys_logs    
    if domain == 'restaurant':

        def book_restaurant(name, people, day, time):
            info = {'name': name, 'people': str(people), 'day': day, 'time': time}
            flag, msg = make_booking_db('restaurant', info)
            logger.info(f"Booking status: flag = {flag}, msg = {msg}")

            dsys_role = 'assistant' if dsys_logs[-1]['role'] == 'user' else 'user'
            dsys_logs.append({'role': dsys_role, 'content': f"Booking Status: {msg}"})

            if flag:
                return msg, domain, info
            return msg, domain, None

        name = 'book_restaurant'
        schema = {
            "type": "function",
            "function": {             
                'name': name,
                'description': 'Book a restaurant with certain requirements.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'name': {
                            'type': 'string',
                            'description': 'the name of the restaurant to book',
                        },
                        'people': {
                            'type': 'integer',
                            'description': 'the number of people',
                        },
                        'day': {
                            'type': 'string',
                            "enum": ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'],
                            'description': 'the day when the people go to the restaurant',
                        },
                        'time': {
                            'type': 'string',
                            'description': 'the time of the reservation',
                        },
                    },
                    'required': ['name', 'people', 'day', 'time'],
                }
            }
        }
        return {'name': name, 'function': book_restaurant, 'schema': schema}

    elif domain == 'hotel':

        def book_hotel(name, people, day, stay):
            info = {'name': name, 'people': str(people), 'day': day, 'stay': str(stay)}
            flag, msg = make_booking_db('hotel', info)
            logger.info(f"Booking status: flag = {flag}, msg = {msg}")

            dsys_role = 'assistant' if dsys_logs[-1]['role'] == 'user' else 'user'
            dsys_logs.append({'role': dsys_role, 'content': f"Booking Status: {msg}"})

            if flag:
                return msg, domain, info
            return msg, domain, None

        name = 'book_hotel'
        schema = {
            "type": "function",
            "function": {             
                'name': name,
                'description': 'Book a hotel with certain requirements.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'name': {
                            'type': 'string',
                            'description': 'the name of the hotel to book',
                        },
                        'people': {
                            'type': 'integer',
                            'description': 'the number of people',
                        },
                        'day': {
                            'type': 'string',
                            "enum": ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'],
                            'description': 'the day when the reservation starts',
                        },
                        'stay': {
                            'type': 'integer',
                            'description': 'the number of days of the reservation',
                        },
                    },
                    'required': ['name', 'people', 'day', 'stay'],
                }
            }
        }
        return {'name': name, 'function': book_hotel, 'schema': schema}

    elif domain == 'train':

        def buy_train_tickets(train_id, tickets):
            info = {'train id': train_id, 'tickets': str(tickets)}
            flag, msg = make_booking_db('train', info)
            logger.info(f"Booking status: flag = {flag}, msg = {msg}")

            dsys_role = 'assistant' if dsys_logs[-1]['role'] == 'user' else 'user'
            dsys_logs.append({'role': dsys_role, 'content': f"Booking Status: {msg}"})
            if flag:
                return msg, domain, info
            return msg, domain, None

        name = 'buy_train_tickets'
        schema = {
            "type": "function",
            "function": {             
                'name': name,
                'description': 'Buy train tickets.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'train_id': {
                            'type': 'string',
                            'description': 'the unique id of the train',
                        },
                        'tickets': {
                            'type': 'integer',
                            'description': 'the number of tickets to buy',
                        },
                    },
                    'required': ['train_id', 'tickets'],
                }
            }
        }
        return {'name': name, 'function': buy_train_tickets, 'schema': schema}

    elif domain == 'taxi':

        def book_taxi(departure, destination, leaveat=None, arriveby=None):
            info = {'departure': departure, 'destination': destination}
            if leaveat:
                info['leaveat'] = leaveat
            if arriveby:
                info['arriveby'] = arriveby
            flag, msg = make_booking_taxi(info) #Dont add -dbase.db for taxi
            logger.info(f"Booking status: flag = {flag}, msg = {msg}")

            dsys_role = 'assistant' if dsys_logs[-1]['role'] == 'user' else 'user'
            dsys_logs.append({'role': dsys_role, 'content': f"Booking Status: {msg}"})
            if flag:
                return msg, domain, info
            return msg, domain, None

        name = 'book_taxi'
        schema = {
            "type": "function",
            "function": {            
                'name': name,
                'description': 'Book a taxi with certain requirements.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'departure': {
                            'type': 'string',
                            'description': 'the departure of the taxi',
                        },
                        'destination': {
                            'type': 'string',
                            'description': 'the destination of the taxi',
                        },
                        'leaveat': {
                            'type': 'string',
                            'description': 'the leave time of the taxi',
                        },
                        'arriveby': {
                            'type': 'string',
                            'description': 'the arrive time of the taxi',
                        },
                    },
                    'required': ['departure', 'destination'],
                }
            }
        }
        return {'name': name, 'function': book_taxi, 'schema': schema}
    
    else:
        raise ValueError(f'{domain = }')


class FuncAgent:

    def __init__(self, model, player_llm, db_path):
        self.model = model #model is nothing model_id in other systems
        self.player_llm = player_llm
        self.db_path = db_path
        self.turn_idx = 0
        self.messages = [{"role": "system", "content": system_prompt}]
        self.prompt_updt_func_name = None
        self.func_map = {}
        self.schema_map = {}
        self.schemas = []
        self.factories = [
            partial(prepare_query_db_functions, domain='restaurant'),
            partial(prepare_book_functions, domain='restaurant'),
            partial(prepare_query_db_functions, domain='hotel'),
            partial(prepare_book_functions, domain='hotel'),
            partial(prepare_query_db_functions, domain='attraction'),
            partial(prepare_query_db_functions, domain='train'),
            partial(prepare_book_functions, domain='train'),
            partial(prepare_book_functions, domain='taxi'),
        ]
        for factory in self.factories:
            result = factory()
            self.func_map[result['name']] = result['function']
            self.schema_map[result['name']] = result['schema']
            self.schemas.append(result['schema'])

        self.genslots = {}
        logger.info(f"FuncAgent __init__ model={model}\nself.schemas={self.schemas}")

    @tenacity.retry(wait=tenacity.wait_exponential(min=2, max=60),
                    stop=tenacity.stop_after_attempt(8),
                    before_sleep=tenacity_retry_log,
                    retry=tenacity.retry_if_exception_type(openai.OpenAIError))

    def chat(self, messages, current_turn, callbacks: list[LLMResult] =[]):

        prompt, raw_answer, answer = self.player_llm(messages, current_turn, self.schemas, None)

        dsys_role = 'assistant' if dsys_logs[-1]['role'] == 'user' else 'user'        
        dsys_logs.append({'role': dsys_role, 'content': {'prompt': prompt, 'raw_answer': raw_answer['message_full'],
                                                                    'answer': f"Agent Follow-up: {answer}"}})        
        logger.info(f"FuncAgent chat: raw_answer: {raw_answer}, answer: {answer}")

        '''
        llm_output = {'model_name': completion.model, 'token_usage': completion.usage.total_tokens}
        reponse = LLMResult(generations=[], llm_output=llm_output)
        for callback in callbacks:
            if hasattr(callable, 'on_llm_end'):
                callback.on_llm_end(reponse)
        '''
        #Since clemfw returns answer as a string, we need to convert it to a dict
        try:
            answer = json.loads(answer)
            if isinstance(answer, dict):
                answer = [answer]
        except Exception as error:
            logger.error(f"Error in converting answer to dict: {error}")
            answer = answer.strip()

        return raw_answer['tool_calls'], answer


    def ishfmodel(self):
        #This response formatting is not helping, hence disabled this
        #return False
        return True if any(model in self.model for model in ["Qwen", "Llama"]) else False        

    def __call__(self, user_utter, current_turn, callbacks=[]):
        self.turn_idx += 1

        global dsys_logs
        dsys_logs = [{'role': 'user', 'content': f"User Input: {user_utter}"}]
        # No need to add tool content here as this is user response but not a function call response
        self.messages.append({'role': 'user', 'content': user_utter})


        num_turns = 0
        while True:
            if num_turns > 3:
                dsys_role = 'assistant' if dsys_logs[-1]['role'] == 'user' else 'user'
                dsys_logs.append({'role': dsys_role, 'content': f"Agent Follow-up: {self.messages[-1]['content']}"})

                return self.messages[-1]['content'], dsys_logs

            raw_answer, msg = self.chat(self.messages, current_turn, callbacks=callbacks)
            logger.info(f"FuncAgent chat response: {msg}, {raw_answer} {type(msg)}")
            
            # Assistant Response
            #if msg.content is not None:
            #if not isinstance(msg, openai.types.chat.chat_completion_message.FunctionCall) and not isinstance(msg, openai.types.chat.chat_completion_message_tool_call.ChatCompletionMessageToolCall):
            #if not isinstance(msg, openai.types.chat.chat_completion_message_tool_call.ChatCompletionMessageToolCall):

            if isinstance(msg, str):
                utter = msg.strip()#msg.content.strip()

                dsys_role = 'assistant' if dsys_logs[-1]['role'] == 'user' else 'user'
                dsys_logs.append({'role': dsys_role, 'content': f"Agent Follow-up: {msg}"})

                for handler in callbacks:
                    if hasattr(handler, 'on_turn_end'):
                        utter = handler.on_turn_end(utter, self.turn_idx)

                logger.info(f"FuncAgent returning utter: {utter}")
                #TODO: Check if this needs further modifications for HF models - Tool schema style response
                self.messages.append({'role': 'assistant', 'content': utter})

                dsys_response = {"status": "follow-up", "details": utter}
                return dsys_response, dsys_logs
            
            if not self.ishfmodel():
                self.messages.append(raw_answer)
            # Function calling
            for msg_toolcall in msg:
                logger.info(f"FuncAgent: calling parse_function_call")

                succeed, check_msg, name, args = self.parse_function_call(msg_toolcall)


                if self.ishfmodel():
                    #prompt_updt_func_name - function name and args that are to be used in the prompt list
                    tool_content = [{"type": "function", "function": {"name": name, "arguments": args}}]
                    self.messages.append({'role': "assistant",  "tool_calls": tool_content})
                #else:
                    #self.messages.append(raw_answer)

                dsys_role = 'assistant' if dsys_logs[-1]['role'] == 'user' else 'user'            
                dsys_logs.append({'role': dsys_role, 'content': f"Function-call: name: {name}, args: {args}"}) 

                #self.messages.append({'role': "assistant", 'content': f"Function-call: name: {name}, args: {args}"})           
                #This was workking until replaced with below tool calls code - may not be necessary here
                #self.messages.append({'role': "assistant", 'tool_calls': [msg]})

                logger.info(f"succeed: {succeed}, check_msg: {check_msg}, name: {name}, args: {args}")
                if succeed:
                    #print()
                    #print('Function: ' + MAG_COLOR + f'{name}' + RESET_COLOR)
                    #print('Arguments: ' + GREEN_COLOR + f'{args}' + RESET_COLOR)
                    func = self.func_map[name]
                    logger.info(f"Calling the function: {func}, {args}")

                    result, domain, details = func(**args)
                    logger.info(f"Function result: {result} domain: {domain} details = {details}")
                    if details:
                        if domain not in self.genslots:
                            self.genslots[domain] = {}
                        self.genslots[domain].update(details)
                        logger.info(f"Updated genslots: {self.genslots}")

                else:
                    #print()
                    #print('Function parsing error:')
                    #print(f'function_call: {msg.function_call}')
                    #print(check_msg)
                    result = check_msg
                    logger.info(f"After Function parsing error result: {result}")            

                #print('Result: ' + CYAN_COLOR + f'{result}' + RESET_COLOR)
                logger.info(f"Result = {result}")

                dsys_role = 'assistant' if dsys_logs[-1]['role'] == 'user' else 'user'            
                dsys_logs.append({'role': dsys_role, 'content': f"Function-Call Result: {result}"})
                '''
                if self.messages[-1]["role"] == "function":
                #if self.messages[-1]["role"] == "tool":
                    self.messages[-1]['name'] = name
                    self.messages[-1]["content"] = result
                else:
                    self.messages.append({'role': 'function', 'name': name, 'content': result})
                    #self.messages.append({"role": "tool", "tool_call_id": msg.id, "content":result})
                '''
                #msgrole = "assistant" if self.messages[-1]["role"] == "user" else "user"

                '''
                #This was working until replaced with below tool calls code
                if isinstance(msg, dict):
                    self.messages.append({'role': "tool",  'content': result})
                else:
                    self.messages.append({'role': "tool",  "tool_call_id": msg.id, 'content': result})
                '''
                if self.ishfmodel():
                    tool_content = {"name": name, "content": result}
                    self.messages.append({'role': 'tool', 'content': tool_content})
                else:
                    self.messages.append({'role': 'tool', "tool_call_id": msg_toolcall['id'], 'content': str(result)})                
            num_turns += 1

    def parse_function_call(self, function_call):
        logger.info(f"Inside parse_function_call: {function_call}")#, {function_call.name}, {function_call.arguments}")

        if function_call is None:
            return False, 'No "function_call" provided.', None, None

        if isinstance(function_call, openai.types.chat.chat_completion_message.FunctionCall):

            if not hasattr(function_call, 'name'):
                return False, 'No function name provided.', None, None

            if not hasattr(function_call, 'arguments'):
                return False, 'No function arguments provided.', None, None
            
            name, args = function_call.name, function_call.arguments
        
        elif isinstance(function_call, openai.types.chat.chat_completion_message_tool_call.ChatCompletionMessageToolCall):
            if not hasattr(function_call, 'function'):
                return False, 'No "function_call" provided.', None, None

            if not hasattr(function_call.function, 'name'):
                return False, 'No function name provided.', None, None

            if not hasattr(function_call.function, 'arguments'):
                return False, 'No function arguments provided.', None, None


            name = function_call.function.name
            args = function_call.function.arguments
            #args = json.loads(args)

        elif isinstance(function_call, dict):
            if 'name' not in function_call:
                return False, 'No function name provided.', None, None

            if 'parameters' not in function_call and 'arguments' not in function_call:
                return False, 'No function arguments provided.', None, None
            

            name = function_call['name']
            if 'parameters' in function_call:
                args = function_call['parameters']
            else:
                args = function_call['arguments']

            if not isinstance(args, str):
                args = json.dumps(args)

        logger.info(f"name: {name}, args: {args}")

        if name not in self.func_map:
            return False, f'Function {name} does not exist and cannot be called.', None, None

        # Parase: (\\'): "{\n  \"sql\": \"SELECT area, address FROM hotel WHERE name = 'rosa\\'s bed and breakfast'\"\n}"
        args = re.sub(r'''("sql": ".*WHERE.*name = '.*)\\('.*'")''', r'\1\2', args)
        try:
            args = json.loads(args)
        except json.JSONDecodeError as e:
            return False, f'Failed to parse the function arguments: {args}.', None, None


        if 'sql' in args:
            args['sql'] = re.sub(r"name = '(.*'.*)'", r'name = "\1"', args['sql'])

        schema = self.schema_map[name]
        if error_args := [arg for arg in args if arg not in schema['function']['parameters']['properties']]:
            error_args_str = ', '.join(f'"{x}"' for x in error_args)
            right_args_str = ', '.join(f'"{x}"' for x in schema['function']['parameters']['properties'])
            return False, f'Parameters {error_args_str} are not valid. Please provide valid parameters {right_args_str}.', None, None
        elif missing_args := [arg for arg in schema['function']['parameters']['required'] if arg not in args]:
            args_str = ', '.join(f'"{x}"' for x in missing_args)
            return False,  f'The required parameters {args_str} are missing.', None, None
        
        return True, 'succeed', name, args

    

    def get_booking_data(self):
        logger.info(f"Returing generated data: {self.genslots}")
        return self.genslots
