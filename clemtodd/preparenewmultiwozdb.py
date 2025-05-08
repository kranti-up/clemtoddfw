import sqlite3


def get_create_statement(table_name):
    if table_name == "hotel":
        return """
            CREATE TABLE IF NOT EXISTS hotel (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                address TEXT,
                area TEXT,
                internet TEXT,
                parking TEXT,
                single TEXT,
                double TEXT,
                family TEXT,
                name TEXT,
                type TEXT,
                phone TEXT,
                postcode TEXT,
                pricerange TEXT,
                takesbookings TEXT,
                stars TEXT
            )
        """
    elif table_name == "restaurant":
        return """
            CREATE TABLE IF NOT EXISTS restaurant (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                address TEXT,
                area TEXT,
                food TEXT,
                name TEXT,
                phone TEXT,
                postcode TEXT,
                pricerange TEXT
            )
        """
    elif table_name == "train":
        return """
            CREATE TABLE IF NOT EXISTS train (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                arriveby TEXT,
                departure TEXT,
                destination TEXT,
                day TEXT,
                leaveat TEXT,
                trainid TEXT,
                price TEXT,
                duration TEXT
            )
        """


def setup_sqlite_db(table_name, db_name="multiwoz_synthetic.db"):
    #conn = sqlite3.connect(f"/home/admin/Desktop/codebase/cocobots/todsystems/clembench/clemtod/resources/data/en/multiwoz/synthetic/{db_name}")
    conn = sqlite3.connect(f"/home/admin/Desktop/codebase/cocobots/todsystems/clembench/clemtod/resources/data/en/multiwoz/unrealistic/{db_name}")
    cursor = conn.cursor()
    create_statement = get_create_statement(table_name)
    cursor.execute(create_statement)
    conn.commit()
    return conn