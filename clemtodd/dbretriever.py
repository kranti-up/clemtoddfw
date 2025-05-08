import sqlite3

import json
#from clemgame import get_logger

import logging

logger = logging.getLogger(__name__)


class DBRetriever:
    def __init__(self, domains, dbpath):
        logger.error(f"DBRetriever: domain:{domains} dbpath:{dbpath}")
        self.domains = domains
        self.dbpath = dbpath
        self.dbcon = {}
        self._prepare_db_connection(domains)

    def _prepare_db_connection(self, domains):
        for domain in domains:
            self.dbcon[domain] = sqlite3.connect(f"{self.dbpath}/{domain}-dbase.db")

    def getcolumns(self, domains=None):

        dbcolumns = {}

        if domains is None:
            domains = self.domains

        for domain in domains:
            connection = self.dbcon.get(domain)
            if connection is None:
                logger.error(f"Domain {domain} not found in dbcon.")
                continue
            
            dbcolumns[domain] = []
            cursor = connection.cursor()
            cursor.execute(f"PRAGMA table_info({domain})")
            column_names = [row[1] for row in cursor.fetchall()]
            dbcolumns[domain] = column_names

            # Close the connection
            cursor.close()

        return dbcolumns


    def run(self, domain, query, values):

        if domain not in self.domains:
            logger.error(f"Domain {domain} not found in domains.")
            return []
        
        dbcon = self.dbcon[domain]
        dbcon.row_factory = sqlite3.Row
        cursor = dbcon.cursor()

        try:
            cursor.execute(query, values)
            rows = cursor.fetchall()
            result = [dict(row) for row in rows]
            return result# if len(result) <= 5 else result[:5]
        finally:
            # Cleanup
            cursor.close()
            #self.dbcon.close()
        
    
    def reset(self, domain):
        if domain is None:
            for domain in self.domains:
                self.dbcon[domain].close()
        else:
            self.dbcon[domain].close()


if __name__ == "__main__":
    #dbr = DBRetriever(["restaurant"], "games/todsystem/resources/data/en/multiwoz")
    #print(dbr.getcolumns())
    #query = ("SELECT * FROM restaurant WHERE area = ? AND type = ?", ["centre", "turkish"])
    #print(dbretriever.run("restaurant", "SELECT * FROM restaurant WHERE name = ?", ["The Eagle"]))
    #print(dbr.run("restaurant", query[0], query[1]))
    #dbr.reset("restaurant")

    #dbr = DBRetriever(["hotel"], "games/todsystem/resources/data/en/multiwoz")
    #query = ("SELECT * FROM hotel WHERE stars >= ? AND parking = ? AND type = ?", ["3", "yes", "hotel"])
    #query = ("SELECT * FROM hotel WHERE stars = ?", ["3"])
    #print(dbr.run("hotel", query[0], query[1]))
    #dbr.reset("hotel")

    dbr = DBRetriever(["train"], "games/todsystem/resources/data/en/multiwoz")
    query = ("SELECT * FROM train WHERE trainid = ?", ["TR7208"])
    #query = ("SELECT * FROM train WHERE trainid = ? AND departure = ? AND leaveat = ? AND day = ? AND destination = ?", ['tr7208', 'cambridge', '21:01', 'sunday', 'broxbourne'])
    print(dbr.run("train", query[0], query[1]))
    dbr.reset("train")
