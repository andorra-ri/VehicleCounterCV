import mysql.connector

class SQLManager:
    def __init__(self, pathToConfigFile):
        with open(pathToConfigFile, 'r') as handle:
            dbConfig = json.load(handle)


    def initConnection(self):
        try:
            self.cnx = mysql.connector.coonect(**dbConfig)
        except mysql.connector.Error as err:
            print(err)


    def closeConnection(self):
        self.cnx.close()


    def executeInsertQuery(self, sqlStatement):
        self.initConnection()
        cursor = self.cnx.cursor()

        try:
            cursor.execute(sqlStatement)
            self.cnx.commit()
            cursor.close()
            self.closeConnection()
        except MySQLdb.Error as err:
            print(err)


    def executeGetQuery(self, sqlStatement):
        self.initConnection()
        cursor = self.cnx.cursor()

        try:
            values = cursor.execute(sqlStatement)
            self.cnx.commit()
            cursor.close()
            self.closeConnection()

        except MySQLdb.Error as err:
            print(err)


        return values
