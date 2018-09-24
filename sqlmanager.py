import mysql.connector

class SQLManager:

    def __init__(self, pathToConfigFile):
        with open(pathToConfigFile, 'r') as handle:
            self.dbConfig = json.load(handle)
            self.table = self.dbConfig["table"]
            del self.dbConfig["table"]


    def connect(self):
        try:
            self.conn = mysql.connector.connect(**self.dbConfig)
        except mysql.connector.Error as err:
            print(err)


    def executeInsertQuery(self, sqlStatement, data):
        try:
            cursor = self.conn.cursor()
            cursor.executemany(sqlStatement, data)
            self.conn.commit()
        except mysql.connector.Error as err:
            self.connect()
            cursor = self.conn.cursor()
            cursor.executemany(sqlStatement, data)
            self.conn.commit()

        return cursor


    def executeQuery(self, sqlStatement):
        try:
            cursor = self.conn.cursor()
            cursor.execute(sqlStatement)
        except mysql.connector.Error as err:
            self.connect()
            cursor = self.conn.cursor()
            cursor.execute(sqlStatement)

        return cursor


    def getTable(self):
        return self.table


    def closeConnection(self):
        self.cnn.close()
