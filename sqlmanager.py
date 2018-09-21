import mysql.connector

class SQLManager:

    def __init__(self, pathToConfigFile):
        with open(pathToConfigFile, 'r') as handle:
            dbConfig = json.load(handle)


    def connect(self):
        try:
            self.conn = mysql.connector.connect(**dbConfig)
        except mysql.connector.Error as err:
            print(err)


    def executeInsertQuery(self, sqlStatement, data):
        try:
            cursor = self.conn.cursor()
            cursor.executemany(sqlStatement, data)        #We use executemany because we will have more than one insert
            conn.commit()
        except mysql.connector.Error as err:
            self.connect()
            cursor = self.conn.cursor()
            cursor.executemany(sqlStatement, data)
            conn.commit()

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


    def closeConnection(self):
        self.cnn.close()
