import uuid
import random
import json
from dbretriever import DBRetriever
from preparenewmultiwozdb import setup_sqlite_db
random.seed(121)

existing_train_ids = set()
existing_hotel_names = set()
existing_restaurant_names = set()
# Ontology as before
ontology = {
    # Hotel
    "hotel-pricerange": ["cheap", "moderate", "expensive"],#, "do n't care"],
    "hotel-type": ["hotel", "guesthouse"],
    "hotel-parking": ["yes", "no", "free"],# "do n't care"],
    "hotel-stars": ["1", "2", "3", "4", "5"],
    "hotel-internet": ["yes", "no", "free"],#, "do n't care"],
    "hotel-name": ["the lensfield hotel",
        "finches bed and breakfast",
        "worth house",
        "wandlebury coutn",
        "allenbell",
        "rosa's bed and breakfast",
        "home from home",
        "avalon",
        "alpha-milton guest house",
        "alexander bed and breakfast",
        "cityroomz",
        "limehouse",
        "archway house",
        "warkworth house",
        "lovell lodge",
        "aylesbray lodge guest house",
        "carolina bed and breakfast",
        "huntingdon marriott hotel",
        "hobsons house",
        "hamilton lodge",
        "whale",
        "alp",
        "cambridge belfry",
        "bridge guest house",
        "gonville",
        "cambridge",
        "acorn  house"],
    "hotel-bookstay": ["1", "2", "3", "4", "5", "6", "7", "8"],
    "hotel-bookday": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
    "hotel-bookpeople": ["1", "2", "3", "4", "5", "6", "7", "8"],
    "hotel-area": ["east", "west", "north", "south", "centre"],

    # Restaurant
    "restaurant-food": ["turkish", "indian", "chinese", "seafood", "italian", "british", "australian", "asian oriental", "thai"],
    "restaurant-pricerange": ["cheap", "moderate", "expensive"],# "do n't care"],
    "restaurant-area": ["east", "west", "north", "south", "centre"],# "do n't care"],
    "restaurant-name": ["meze bar restaurant",
        "indian",
        "pizza hut city centre",
        "the good luck chinese food takeaway",
        "caffe uno",
        "the gardenia",
        "the oak bistro",
        "sala thong",
        "thanh binh",
        "riverside brasserie",
        "cambri",
        "pizza express",
        "yippee noodle bar",
        "curry prince",
        "midsummer house restaurant",
        "cote",
        "restaurant alimentum",
        "nandos city centre",
        "chiquito restaurant bar",
        "maharajah tandoori restaurant",
        "yu garden",
        "bangkok city",
        "copper kettle",
        "backstreet bistro",
        "the golden curry",
        "don pasquale pizzeria",
        "sesame restaurant and bar",
        "charlie",
        "the cow pizza kitchen and bar",
        "india house",
        "loch fyne",
        "eraina",
        "royal spice",
        "prezzo",
        "curry king",
        "the nirala",
        "curry garden",
        "zizzi cambridge",
        "da vinci pizzeria",
        "jinling noodle bar",
        "la raza"],
    "restaurant-bookday": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
    "restaurant-booktime": ["12:00", "18:00", "19:30", "20:00", "21:00"],
    "restaurant-bookpeople": ["1", "2", "3", "4", "5", "6", "7", "8"],

    # Train
    "train-leaveat": ["21:15",
        "12:45",
        "19:45",
        "14:00",
        "15:15",
        "10:00",
        "09:29",
        "10:15",
        "09:15",
        "19:15",
        "11:15",
        "17:45",
        "14:30",
        "20:00",
        "09:30",
        "08:15",
        "20:15",
        "10:45",
        "14:45",
        "17:00",
        "21:00",
        "17:30",
        "11:30",
        "13:45",
        "12:15",
        "08:45",
        "11:45",
        "09:00",
        "18:45",
        "05:15",
        "18:00",
        "16:00",
        "11:00",
        "05:00",
        "08:00",
        "18:30",
        "21:45",
        "16:30",
        "14:15",
        "16:15",
        "10:32",
        "12:30",
        "13:00",
        "15:32",
        "13:30",
        "02:00",
        "08:30",
        "15:00",
        "10:30",
        "15:45",
        "09:45",
        "21:30",
        "05:59",
        "06:00",
        "18:15"],
    "train-departure": ["birmingham new street",
          "bishops stortford",
          "broxbourne",
          "cambridge",
          "ely",
          "kings lynn",
          "leicester",
          "london kings cross",
          "london liverpool street",
          "norwich",
          "peterborough",
          "stansted airport",
          "stevenage"],
    "train-destination": ["birmingham new street",
          "bishops stortford",
          "broxbourne",
          "cambridge",
          "ely",
          "kings lynn",
          "leicester",
          "london kings cross",
          "london liverpool street",
          "norwich",
          "peterborough",
          "stansted airport",
          "stevenage"],
    "train-day": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
    "train-arriveby": ["19:45",
        "20:45",
        "11:30",
        "14:45",
        "08:15",
        "13:45",
        "08:00",
        "08:45",
        "11:00",
        "11:45",
        "08:30",
        "10:45",
        "05:00",
        "20:00",
        "15:15",
        "11:15",
        "19:00",
        "20:15",
        "16:45",
        "12:45",
        "10:15",
        "21:00",
        "15:00",
        "17:15",
        "14:30",
        "20:30",
        "18:15",
        "12:15",
        "18:45",
        "12:00",
        "10:00",
        "21:45",
        "16:15",
        "14:00",
        "18:30",
        "17:45",
        "13:30",
        "19:30",
        "12:30",
        "09:15",
        "09:00",
        "16:00",
        "23:00",
        "10:30",
        "09:30",
        "16:30",
        "18:00",
        "09:45",
        "15:45",
        "19:15",
        "18:23",
        "13:00",
        "17:30",
        "15:30",
        "21:15",
        "13:15",
        "21:30",
        "17:00",
        "8:00",
        "14:15"],
    "train-bookpeople": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
}

domain_slots = {
    "hotel": {
        "constraints": ["hotel-name", "hotel-area", "hotel-pricerange",
                         "hotel-type", "hotel-parking", "hotel-stars", "hotel-internet"],
        "booking": ["hotel-bookpeople", "hotel-bookday", "hotel-bookstay"],
        "attributes": ["hotel-address", "hotel-phone", "hotel-postcode"]
    },
    "restaurant": {
        "constraints": ["restaurant-name", "restaurant-area", "restaurant-food", "restaurant-pricerange"],
        "booking": ["restaurant-bookpeople", "restaurant-bookday", "restaurant-booktime"],
        "attributes": ["restaurant-address", "restaurant-phone", "restaurant-postcode"]
    },
    "train": {
        "constraints": ["train-departure", "train-destination", "train-day", "train-leaveat",
                         "train-arriveby"],
        "booking": ["train-bookpeople"],
        "attributes": ["train-price", "train-duration"]
    }
}

def convert_names(slot_name):
    if slot_name == "hotel-area":
        return "info", "area"
    elif slot_name == "hotel-pricerange":
        return "info", "pricerange"
    elif slot_name == "hotel-type":
        return "info", "type"
    elif slot_name == "hotel-parking":
        return "info", "parking"
    elif slot_name == "hotel-stars":
        return "info", "stars"
    elif slot_name == "hotel-internet":
        return "info", "internet"
    elif slot_name == "hotel-name":
        return "info", "name"
    elif slot_name == "hotel-bookstay":
        return "book", "stay"
    elif slot_name == "hotel-bookday":
        return "book", "day"
    elif slot_name == "hotel-bookpeople":
        return "book", "people"
    elif slot_name == "hotel-address":
        return "req", "address"
    elif slot_name == "hotel-phone":
        return "req", "phone"
    elif slot_name == "hotel-postcode":
        return "req", "postcode"
    
    elif slot_name == "restaurant-area":
        return "info", "area"
    elif slot_name == "restaurant-food":
        return "info", "food"
    elif slot_name == "restaurant-pricerange":
        return "info", "pricerange"
    elif slot_name == "restaurant-name":
        return "info", "name"
    elif slot_name == "restaurant-bookday":
        return "book", "day"
    elif slot_name == "restaurant-booktime":
        return "book", "time"
    elif slot_name == "restaurant-bookpeople":
        return "book", "people"
    elif slot_name == "restaurant-address":
        return "req", "address"
    elif slot_name == "restaurant-phone":
        return "req", "phone"
    elif slot_name == "restaurant-postcode":
        return "req", "postcode"
    
    elif slot_name == "train-departure":
        return "info", "departure"
    elif slot_name == "train-destination":
        return "info", "destination"
    elif slot_name == "train-day":
        return "info", "day"
    elif slot_name == "train-leaveat":
        return "info", "leaveat"
    elif slot_name == "train-arriveby":
        return "info", "arriveby"
    elif slot_name == "train-bookpeople":
        return "book", "people"
    elif slot_name == "train-price":
        return "req", "price"
    elif slot_name == "train-duration":
        return "req", "duration"

    
    

def generate_goal_for_domains(selected_domains, num_counter):
    goal = {}

    for domain in selected_domains:
        if domain not in goal:
            goal[domain] = {}
        slots = domain_slots[domain]
        selected_constraints = random.sample(slots["constraints"], random.randint(1, len(slots["constraints"])))
        for slot in selected_constraints:
            slot_type, slot_name = convert_names(slot)
            if slot_type not in goal[domain]:
                goal[domain][slot_type] = {}
            goal[domain][slot_type][slot_name] = random.choice(ontology[slot])
        if domain == "train":
            slot_type, slot_name = convert_names("train-departure")
            if slot_type not in goal[domain]:
                goal[domain][slot_type] = {}

            tdeparture = random.choice(ontology["train-departure"])
            goal[domain][slot_type][slot_name] = tdeparture

            dest = random.choice([d for d in ontology["train-destination"] if d != tdeparture])
            slot_type, slot_name = convert_names("train-destination")
            goal[domain][slot_type][slot_name] = dest

            slot_type, slot_name = convert_names("train-day")
            goal[domain][slot_type][slot_name] = random.choice(ontology["train-day"])

            time_slot = random.choice(["train-leaveat", "train-arriveby"])
            slot_type, slot_name = convert_names(time_slot)
            goal[domain][slot_type][slot_name] = random.choice(ontology[time_slot])
        #else:
        for slot in slots["booking"]:
            slot_type, slot_name = convert_names(slot)
            if slot_type not in goal[domain]:
                goal[domain][slot_type] = {}
            goal[domain][slot_type][slot_name] = random.choice(ontology[slot])
        goal[domain]["fail_info"] = {}
        goal[domain]["fail_book"] = {}

    goal_id = "SNG" if len(selected_domains) == 1 else "MUL"
    goal_id += f"0{num_counter+1}.json"  



    return goal, goal_id

def goal_to_text(goal, domain_set):
    sentences = []


    for domain in domain_set:
        info_slots = goal[domain]["info"]
        book_slots = goal[domain]["book"]

        if "req" in goal[domain]:
            attr_slots = goal[domain]["req"]
        else:
            attr_slots = None


        # HOTEL
        #print(domain_set)
        #input()
        #if any(slot.startswith("hotel-") for slot in goal):
        if "hotel" == domain:
            name_phrase = f"called {info_slots.get('name')}" if "name" in info_slots else ""
            type_phrase = info_slots.get("type", "hotel")
            location_phrase = f"in the {info_slots['area']}" if "area" in info_slots else ""
            price_val = info_slots.get("pricerange")
            price_phrase = f"a {price_val}" if price_val and price_val != "do n't care" else "a"

            parking_phrase = ""
            if "parking" in info_slots and info_slots["parking"] != "do n't care" and info_slots["parking"].lower() in ["yes", "free"]:
                parking_phrase = f" with {'free' if info_slots['parking'] == 'free' else ''} parking"

            stars_phrase = ""
            if "stars" in info_slots:
                stars_phrase = f" with {info_slots['stars']} stars"

            internet_phrase = ""
            if "internet" in info_slots and info_slots["internet"].lower() in ["yes", "free"]:
                internet_phrase = " with internet"

            sentence = f"I am looking for {price_phrase} {type_phrase} {name_phrase} {location_phrase}{parking_phrase}{stars_phrase}{internet_phrase}.".strip()
            sentence = sentence.replace("  ", " ")
            if all(k in book_slots for k in ["people", "stay", "day"]):
                sentence += f" I want to book it for {book_slots['people']} people for {book_slots['stay']} nights starting from {book_slots['day']}."
            sentences.append(sentence)

            if attr_slots:
                extras = []
                for slot in attr_slots:
                    if slot == "phone":
                        extras.append("phone number")

                    if slot == "address":
                        extras.append("address")

                    if slot == "postcode":
                        extras.append("postcode")
                if extras:
                    sentence = f"I would also like to know the hotel's {', '.join(extras)}."
                    sentence = sentence.replace("  ", " ").strip()
                    sentences.append(sentence)     

        # RESTAURANT
        #if any(slot.startswith("restaurant-") for slot in goal):
        if "restaurant" == domain:    
            food = info_slots.get("food")
            price = info_slots.get("pricerange")
            area = info_slots.get("area")
            name_phrase = f"called {info_slots.get('name')}" if "name" in info_slots else ""            
            food_phrase = f"{food}" if food else ""
            price_phrase = f"{price} priced" if price and price != "do n't care" else ""
            area_phrase = f"in the {area}" if area else ""
            phrases = " ".join(filter(None, [price_phrase, food_phrase, "restaurant", name_phrase, area_phrase]))
            sentence = f"I am looking for {phrases.strip()}.".replace("  ", " ")
            if all(k in book_slots for k in ["people", "day", "time"]):
                sentence += f" I want to book a table for {book_slots['people']} people on {book_slots['day']} at {book_slots['time']}."
            sentences.append(sentence)

            if attr_slots:
                extras = []
                for slot in attr_slots:
                    if slot == "phone":
                        extras.append("phone number")

                    if slot == "address":
                        extras.append("address")

                    if slot == "postcode":
                        extras.append("postcode")
                if extras:
                    sentence = f"I would also like to know the restaurant's {', '.join(extras)}."
                    sentence = sentence.replace("  ", " ").strip()
                    sentences.append(sentence)              

        # TRAIN
        #if any(slot.startswith("train-") for slot in goal):
        if "train" == domain:     
            departure = info_slots.get("departure")
            destination = info_slots.get("destination")
            train_day = info_slots.get("day")
            if departure and destination and train_day:
                sentence = f"I want to travel by train from {departure} to {destination} on {train_day}."
            elif departure and destination:
                sentence = f"I want to travel by train from {departure} to {destination}."
            elif departure:
                sentence = f"I want to travel by train from {departure}."
            else:
                sentence = "I want to travel by train."
            if "leaveat" in goal:
                sentence += f" I want to leave at {info_slots['leaveat']}."
            elif "arriveby" in goal:
                sentence += f" I want to arrive by {info_slots['arriveby']}."

            if all(k in book_slots for k in ["people"]):
                sentence += f" I want to book tickets for {book_slots['people']} people."
            sentences.append(sentence)

            if attr_slots:
                extras = []
                for slot in attr_slots:

                    if slot == "price":
                        extras.append("price")

                    if slot == "duration":
                        extras.append("duration")
                if extras:
                    sentence = f"I would also like to know the train's {', '.join(extras)}."
                    sentence = sentence.replace("  ", " ").strip()
                    sentences.append(sentence)                



    return " ".join(sentences)


def validate_goal(goal, domain_set):
    errors = []
    for domain in domain_set:
        #if any(k.startswith("hotel-") for k in goal):
        if "hotel" == domain:
            for slot in domain_slots["hotel"]["booking"]:
                slot_name = slot.split('-')[-1].replace("book", "")
                if slot_name not in goal[domain]["book"]:
                    errors.append(f"Missing hotel slot: {slot_name}")
            info_slots = goal["hotel"]["info"]
            if "stars" in info_slots and "pricerange" in info_slots:
                if info_slots["pricerange"] == "expensive" and info_slots["stars"] not in ["4", "5"]:
                    errors.append("Inconsistent hotel stars and price range")
                elif info_slots["pricerange"] == "cheap" and info_slots["stars"] in ["3", "4", "5"]:
                    errors.append("Inconsistent hotel stars and price range")
                elif info_slots["pricerange"] == "moderate" and info_slots["stars"] in ["4", "5"]:
                    errors.append("Inconsistent hotel stars and price range")
                

                
        #if any(k.startswith("restaurant-") for k in goal):
        if "restaurant" == domain:
            for slot in domain_slots["restaurant"]["booking"]:
                slot_name = slot.split('-')[-1].replace("book", "")            
                if slot_name not in goal[domain]["book"]:
                    errors.append(f"Missing restaurant slot: {slot_name}")
        #if any(k.startswith("train-") for k in goal):
        if "train" == domain:
            if not all(k in goal[domain]["info"] for k in ["departure", "destination", "day"]):
                errors.append("Missing train slots: departure, destination, or day")
            if not ("leaveat" in goal[domain]["info"] or "arriveby" in goal[domain]["info"]):
                errors.append("Missing train time slot")
    return errors

def fill_values_for_db_entry(domain, info_slots):
    db_info = info_slots.copy()
    if domain == "hotel":
        if "name" not in db_info:
            while True:
                hotel_name = random.choice(ontology["hotel-name"])
                if hotel_name not in existing_hotel_names:
                    existing_hotel_names.add(hotel_name)
                    db_info["name"] = hotel_name
                    break
        if "area" not in db_info:
            db_info["area"] = random.choice(ontology["hotel-area"])
        if "pricerange" not in db_info:
            if "stars" in db_info:
                if db_info["stars"] in ["4", "5"]:
                    db_info["pricerange"] = "expensive"
                elif db_info["stars"] in ["1", "2"]:
                    db_info["pricerange"] = "cheap"
                elif db_info["stars"] == "3":
                    db_info["pricerange"] = random.choice(["cheap", "moderate"])
            else:
                db_info["pricerange"] = random.choice(ontology["hotel-pricerange"])
        if "type" not in db_info:
            db_info["type"] = random.choice(ontology["hotel-type"])
        if "parking" not in db_info:
            db_info["parking"] = random.choice(ontology["hotel-parking"])
        if "stars" not in db_info:
            if "pricerange" in db_info:
                if db_info["pricerange"] == "expensive":
                    db_info["stars"] = random.choice(["4", "5"])
                elif db_info["pricerange"] == "cheap":
                    db_info["stars"] = random.choice(["1", "2"])
                else:
                    db_info["stars"] = random.choice(["1", "2", "3"])
            else:
                db_info["stars"] = random.choice(ontology["hotel-stars"])
        if "internet" not in db_info:
            db_info["internet"] = random.choice(ontology["hotel-internet"])
        if "address" not in db_info:
            db_info["address"] = "birmingham new street"
        if "phone" not in db_info:
            db_info["phone"] = "01223363682"
        if "postcode" not in db_info:
            db_info["postcode"] = "zh2311"
        db_info["takesbookings"] = "yes"
        db_info["single"] = random.choice(["yes", "no"])
        if db_info["single"] == "yes":
            db_info["double"] = "no"
        else:
            db_info["double"] = random.choice(["yes", "no"])
        db_info["family"] = random.choice(["yes", "no"])

    elif domain == "restaurant":
        if "name" not in db_info:
            while True:
                restaurant_name = random.choice(ontology["restaurant-name"])
                if restaurant_name not in existing_restaurant_names:
                    existing_restaurant_names.add(restaurant_name)
                    db_info["name"] = restaurant_name
                    break
        if "area" not in db_info:
            db_info["area"] = random.choice(ontology["restaurant-area"])
        if "food" not in db_info:
            db_info["food"] = random.choice(ontology["restaurant-food"])
        if "pricerange" not in db_info:
            db_info["pricerange"] = random.choice(ontology["restaurant-pricerange"])
        if "address" not in db_info:
            db_info["address"] = "birmingham new street"
        if "phone" not in db_info:
            db_info["phone"] = "01223363682"
        if "postcode" not in db_info:
            db_info["postcode"] = "zh2311"
    elif domain == "train":
        if "departure" not in db_info:
            db_info["departure"] = random.choice(ontology["train-departure"])
        if "destination" not in db_info:
            db_info["destination"] = random.choice(ontology["train-destination"])
        if "day" not in db_info:
            db_info["day"] = random.choice(ontology["train-day"])
        if "leaveat" not in db_info:
            db_info["leaveat"] = random.choice(ontology["train-leaveat"])
        if "arriveby" not in db_info:
            db_info["arriveby"] = random.choice(ontology["train-arriveby"])
        if "price" not in db_info:
            db_info["price"] = "15.5 pounds"
        if "duration" not in db_info:
            db_info["duration"] = "90 minutes"
        if "trainid" not in db_info:
            while True:
                uid = uuid.uuid4().int % 10000
                train_id = f"TR{uid:04d}"
                if train_id not in existing_train_ids:
                    existing_train_ids.add(train_id)
                    db_info["trainid"] = train_id
                    break
    return db_info



def do_db_validation(goal, domain_set):

    dbr = DBRetriever(domain_set, "/home/admin/Desktop/codebase/cocobots/todsystems/clembench/clemtod/resources/data/en/multiwoz/")

    for domain in domain_set:
        if "info" not in goal[domain]:
            print(f"Info not in goal, {goal}")
            input()
        
        info_slots = goal[domain]["info"]

        db_slots = {}
        for slot in info_slots:
            if info_slots[slot] != "do n't care":
                if info_slots[slot] == "free":
                    db_slots[slot] = "yes"
                else:
                    db_slots[slot] = info_slots[slot]
        #Prepare DB Query
        query = f"SELECT * FROM {domain} WHERE " + " AND ".join([f"{slot} = ?" for slot in db_slots])
        values = [db_slots[slot] for slot in db_slots]
        if not values:
            print(f"Values are empty for {domain} with query: {query} and goal: {goal}")
            #input()
            return False
        db_result = dbr.run(domain, query, values)
        if db_result:
            dbr.reset(domain)
            print(f"Exists in DB - we dont want to reuse that")
            #print(f"Goal: {goal}")
            #input()
            return False
        dbr.reset(domain)


        if "req" not in goal[domain]:
            goal[domain]["req"] = {}
        for attr in domain_slots[domain]["attributes"]:
            if random.random() < 0.5:
                # Fake values
                try:
                    if attr.endswith("phone"):
                        goal[domain]["req"]["phone"] = "01223363682"
                    elif attr.endswith("postcode"):
                        goal[domain]["req"]["postcode"] = "zh2311"
                    elif attr.endswith("address"):
                        goal[domain]["req"]["address"] = "birmingham new street"
                    elif attr == "train-price":
                        goal[domain]["req"]["price"] = "15.5 pounds"
                    elif attr == "train-duration":
                        goal[domain]["req"]["duration"] = "90 minutes"
                except Exception as error:
                    print(f"Error while adding attributes: {error}, {db_result}")
                    input()
        conn = setup_sqlite_db(domain, f"{domain}-dbase.db")
        cursor = conn.cursor()

        # Fill in the values for the database entry
        db_slots_all = fill_values_for_db_entry(domain, db_slots)
        columns = ", ".join(db_slots_all.keys())         # e.g., "area, stars"
        placeholders = ", ".join(["?"] * len(db_slots_all))  # e.g., "?, ?"
        values = tuple(db_slots_all.values())         
        # Insert the values into the table
        cursor.execute(
            f"INSERT INTO {domain} ({columns}) VALUES ({placeholders})",
            values
        )
        conn.commit()
        cursor.close()
        conn.close()

    return True

def generate_and_save(num_samples=120, filename="synthetic_goals.jsonl"):
    domains = list(domain_slots.keys())
    per_domain = num_samples // len(domains)
    samples_per_type = []

    # Single-domain goals
    for domain in domains:
        samples_per_type.append(([domain], per_domain // 2))


    # Two-domain combinations (balanced and rotated)
    two_domain_combinations = [("hotel", "restaurant"), ("restaurant", "train"),
                                ("hotel", "train")]
    for combo in two_domain_combinations:
        samples_per_type.append((list(combo), per_domain // 2))

    count = 0
    skip_goals = 0
    with open(filename, "w") as f:
        for domain_set, n in samples_per_type:
            print(f"Generating {n} samples for domains: {domain_set}")
            for index in range(n):
                while True:
                    goal, goal_id = generate_goal_for_domains(domain_set, index)
                    errors = validate_goal(goal, domain_set)
                    if errors:
                        skip_goals += 1
                        continue
                    db_check = do_db_validation(goal, domain_set)
                    if not db_check:
                        skip_goals += 1
                        continue
                    text = goal_to_text(goal, domain_set)
                    entry = {
                        "domains": goal,
                        "rawmessage": text,
                        "message": text,
                        #"domains": domain_set,
                        "filename": goal_id,
                        "data_split": "test",
                        "tasktype": "info_book_all",
                        "dialogue_type": "single" if "SNG" in goal_id else "multi",
                        "corpususer": [{"utterance": "synthetic", "dialog_act": {}}]

                    }
                    f.write(json.dumps(entry) + "\n")
                    count += 1
                    break
    print(f"Skipped {skip_goals} invalid goals.")
    print(f"✅ Saved {count} goals to {filename}")

# Run
if __name__ == "__main__":
    generate_and_save(num_samples=120)
