import sys
import cloudpickle
import sqlite3 as sqlite

##############
# DB_BACKEND #
##############


class SQLITE_bend:
    
    def __init__(self,pathToDB,tname="data"):
        self.connection = sqlite.connect(pathToDB)
        self.cursor = self.connection.cursor()
        self.kv_table = "data"
        self.columns = ['tag','sire','dam','sex','animalobj']
        self._run(f"create table if not exists {self.kv_table} ({self.columns[0]} TEXT PRIMARY KEY, {self.columns[1]} TEXT, {self.columns[2]} TEXT, {self.columns[3]} INT, {self.columns[4]} BLOB)")
    
    def _run(self,sqlCom):
        try:
            return self.cursor.execute(sqlCom)
        except:
            print(sys.exc_info()[0],sys.exc_info()[1])

    def store_animal(self,tag,sire,dam,sex,animal_obj):
        animal_obj_serialised = sqlite.Binary(cloudpickle.dumps(animal_obj, protocol=None))                   ##NOTE: cloudpickle saves the realised instance of Animal(), which then is stored as BLOB
        comInsert = f"INSERT OR IGNORE INTO {self.kv_table} ({','.join(self.columns)}) VALUES (?,?,?,?,?)"
        self.cursor.execute(comInsert,(tag,sire,dam,sex,animal_obj_serialised))
        
    def get_animal(self,tag):
        query = f"SELECT * FROM {self.kv_table} WHERE tag = {tag}" 
        self._run(query)
        out = {}
        for row in self.cursor.fetchall():
            out = {c:row[i+1] for i,c in enumerate(self.columns[1:])}
            out['animalobj'] = cloudpickle.loads(out['animalobj'])
        return out
    
    def finalise(self):
        """Necessary to register the changes"""
        self.connection.commit()

