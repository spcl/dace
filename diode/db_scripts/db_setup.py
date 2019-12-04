import sqlite3
import os


def db_setup(basepath="."):
    perfdata_path = os.path.join(basepath, "perfdata.db")
    try:
        os.remove(perfdata_path)
    except:
        pass
    # Create the database by connecting to it
    conn = sqlite3.connect(perfdata_path)

    # This is no "production db". On power outage, we WILL lose data.
    conn.execute(
        "PRAGMA JOURNAL_MODE=WAL;"
    )  # WAL probably is not as fast as MEMORY, but does allow more concurrency
    conn.execute("PRAGMA SYNCHRONOUS=OFF;")

    # Get a "Cursor" object for the database
    c = conn.cursor()

    # Create the first table

    # Programs is the table containing the programs, i.e. identical dapp_states
    c.execute('''CREATE TABLE `Programs`
                (programID INTEGER PRIMARY KEY, 
                programHash TEXT UNIQUE, 
                timestamp TIMESTAMP DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')));'''
              )

    # Runs is the table containing the runs and its options for the programs
    c.execute('''CREATE TABLE `Runs`
                (runID INTEGER PRIMARY KEY, 
                program INTEGER,              
                timestamp TIMESTAMP DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                papimode TEXT,
                options TEXT,
                FOREIGN KEY(program) REFERENCES Programs(programID)
                );''')

    c.execute('''
CREATE TABLE `Overheads`
(
    overheadID INTEGER PRIMARY KEY,
    programID INTEGER,
    papiCode INTEGER,
    value REAL,

    FOREIGN KEY(programID) REFERENCES Programs(programID)
)
    
''')

    # SuperSections contains the supersections
    c.execute('''CREATE TABLE `SuperSections`
                (SuperSectionID INTEGER PRIMARY KEY,
                runid INTEGER,
                `order` INTEGER,
                nodeID INTEGER,
                FOREIGN KEY(runid) REFERENCES Runs(runID)
                );''')

    # Same goes for the sections
    c.execute('''CREATE TABLE `Sections`
                (SectionID INTEGER PRIMARY KEY,
                ssid INTEGER,
                `order` INTEGER,
                nodeID INTEGER,
                datasize INTEGER,
                input_datasize INTEGER,
                FOREIGN KEY(ssid) REFERENCES SuperSections(SuperSectionID)
                );
    ''')

    c.execute('''CREATE TABLE `Entries`
                (EntryID INTEGER PRIMARY KEY,
                SectionID INTEGER,
                `order` INTEGER,
                threadID INTEGER,
                iteration INTEGER,
                flags INTEGER,
                FOREIGN KEY(SectionID) REFERENCES Sections(SectionID)
                );
    ''')

    c.execute('''CREATE TABLE `Values`
                (ValueID INTEGER PRIMARY KEY,
                PapiCode INTEGER,
                Value INTEGER,
                entryID INTEGER,
                FOREIGN KEY(entryID) REFERENCES Entries(EntryID))
    ''')

    conn.commit()

    conn.close()


if __name__ == '__main__':
    db_setup()
