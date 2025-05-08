import uuid
import random
import json
from dbretriever import DBRetriever
from preparenewmultiwozdb import setup_sqlite_db
random.seed(121)

existing_train_ids = set()
existing_hotel_names = set()
existing_restaurant_names = set()


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

ontology = {
    # Hotel
    "hotel-pricerange": ["nonsense", "undefined", "blockbuster"],
    "hotel-type": ["dungeon", "wormhole"],
    "hotel-parking": ["on the river", "next to the table"],
    "hotel-stars": ["-10", "1/2", "3^8", "100000", "-0.00072"],
    "hotel-internet": ["45g", "edge"],
    "hotel-name": ["you name it", 'do you know', 'pen and pencil', "speaker", "windy", "shoe polish",
                   "the worst", "corona", "pandemic", "depression", "goddamn", "nevermind", "landslide", "hackernews",
                   "hot furnace", "toy", "brainstorm", "floods", "earthquake", "tsunami", "hurricane",
                   "bullish", "black sea", "night warrior", "lord of the rings", "winter is coming",
                   "fall dance", "british sheldon", "whale watch", "fly high", "skyfall", "roaring lions",
                   "martian king", "airpods", "bits and bytes", "short stays", "dailymail", "day dreams",
                   "frustrating kid", "whatever", "lol", "brb", "idk", "neurips", "sigdial", "corl"],
    "hotel-bookstay": ["0", "-2", "3.75", "40000", "5long", "6stay", "7afternoons", "8seasons"],
    "hotel-bookday": ["yesterday", "lastyear", "daybefore", "someday", "rainyday", "clearday", "goodday"],
    "hotel-bookpeople": ["1.5", "-2", "0", "488976", "-5000", "6L", "7B", "8GB"],
    "hotel-area": ["sky", "earth's core", "nowhere", "middle of ocean", "volcano"],

    # Restaurant
    "restaurant-food": ["oxygen", "rotten", "leftover", "nofood", "chargers", "pixels", "gibli", "clipart", "yoga"],
    "restaurant-pricerange": ["nonsense", "undefined", "blockbuster"],
    "restaurant-area": ["sky", "earth's core", "nowhere", "middle of ocean", "volcano"],
    "restaurant-name": ["laptop", "water", "undercurrent", "gpu", "cheap", "book", "cupboard", "lengthy", "connect", "flying",
                        "twillight", "john doe", "suits", "queen's backyard", "last summer", "spring nights", "fall colors",
                        "sam's dairy", "quantum sleeper", "dark", "saveme", "sos", "forgiveme", "standout", "chinup",
                        "truth or dare", "florist", "sweettooth", "blood rush", "heavy metal", "suitcase", "suit yourself",
                        "try me", "overleaf", "latex", "248hours", "jacket tribe", "plain jeans", "timeup", "aoe timeline", "moon's maid"],
    "restaurant-bookday": ["yesterday", "lastyear", "daybefore", "someday", "rainyday", "clearday", "goodday"],
    "restaurant-booktime": ["99:00", "00:00", "1000:1000", "1e4:30", "07:3x", "??:00", "00:??", "??:??", "2 8:7 6"],
    "restaurant-bookpeople": ["1.5", "-2", "0", "488976", "-5000", "6L", "7B", "8GB"],

    # Train
    "train-leaveat": ["99:00", "00:00", "1000:1000", "1e4:30", "07:3x"],
    "train-departure": ["moon", "earth's core", "nowhere", "middle of ocean", "volcano", "himalayas", "alps", "antarctica",
                        "pyramids", "mount everest", "sun", "interstellar", "smokepond", "sandstorm", "snakebite", "penguin's home",
                        "bluemoon", "saturn", "milkyway", "hubble", "hundredth floor", "myhome", "ourhome", 
                        "intersection", "crossover", "crossfire", "misfire", "codenames", "wordle", "sudoku", "dixit", "mindmap",
                        "silicon valley", "amazon forest", "pacific ocean", "north pole", "iceberg", "mountain top", "birdsnest",
                        "treehole", "power play", "traceback", "callstack", "parachute"],
    "train-destination": ["hogwarts", "deadsea", "blackhole", "magneticfield", "netflix", "sugarcane", "moneyland", "dreams", 
                          "heaven", "hell", "jungle", "bottomless pit", "cloud nine", "south pole", "sun's land", "sunnyside",
                          "keyhole", "doortrap", "gravity", "last over", "yourhome", "nobody's home", "pavement", 'fisshynet',
                          "stacktrace", "crashpad", "dungeons", "end of the tunnel", "lost in time", "centre of space", "musical notes",
                          "spotify", "cherry on top", "cakewalk", "moondance", "stealthy cave", "hidden treasure", "treasure island"],
    "train-day": ["yesterday", "lastyear", "daybefore", "someday", "rainyday", "clearday", "goodday"],
    "train-arriveby": ["99:00", "??:00", "00:??", "??:??", "2 8:7 6", "00:00", "1000:1000", "1e4:30", "07:3x"],
    "train-bookpeople": ["1.5", "-2", "0", "488976", "-5000", "6L", "7B", "8GB"],
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
            price_phrase = f"a {price_val} price range" if price_val and price_val != "do n't care" else "a"

            parking_phrase = ""
            if "parking" in info_slots and info_slots["parking"] != "do n't care":
                parking_phrase = f" with {info_slots['parking']} parking"

            stars_phrase = ""
            if "stars" in info_slots:
                stars_phrase = f" with {info_slots['stars']} stars"

            internet_phrase = ""
            if "internet" in info_slots:
                internet_phrase = f" with {info_slots['internet']} internet"

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
            db_info["pricerange"] = random.choice(ontology["hotel-pricerange"])
        if "type" not in db_info:
            db_info["type"] = random.choice(ontology["hotel-type"])
        if "parking" not in db_info:
            db_info["parking"] = random.choice(ontology["hotel-parking"])
        if "stars" not in db_info:
            db_info["stars"] = random.choice(ontology["hotel-stars"])
        if "internet" not in db_info:
            db_info["internet"] = random.choice(ontology["hotel-internet"])
        if "address" not in db_info:
            db_info["address"] = "corner of mars, opposite the sun"
        if "phone" not in db_info:
            db_info["phone"] = "a-b-c-d-e-1"
        if "postcode" not in db_info:
            db_info["postcode"] = "202504DW"
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
            db_info["address"] = "corner of mars, opposite the sun"
        if "phone" not in db_info:
            db_info["phone"] = "a-b-c-d-e-1"
        if "postcode" not in db_info:
            db_info["postcode"] = "202504DW"
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
            db_info["price"] = "1005.5890 diamonds"
        if "duration" not in db_info:
            db_info["duration"] = "9000 days"
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

        if "req" not in goal[domain]:
            goal[domain]["req"] = {}
        for attr in domain_slots[domain]["attributes"]:
            if random.random() < 0.5:
                # Fake values
                try:
                    if attr.endswith("phone"):
                        goal[domain]["req"]["phone"] = "a-b-c-d-e-1"
                    elif attr.endswith("postcode"):
                        goal[domain]["req"]["postcode"] = "202504DW"
                    elif attr.endswith("address"):
                        goal[domain]["req"]["address"] = "corner of mars, opposite the sun"
                    elif attr == "train-price":
                        goal[domain]["req"]["price"] = "1005.5890 diamonds"
                    elif attr == "train-duration":
                        goal[domain]["req"]["duration"] = "9000 days"
                except Exception as error:
                    print(f"Error while adding attributes: {error}, {goal[domain]}")
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


def generate_and_save(num_samples=120, filename="synthetic_goals_unrealistic.jsonl"):
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