import sqlite3
import os

# Create the database by connecting to it
conn = sqlite3.connect('perfdata.db')

# Get a "Cursor" object for the database
c = conn.cursor()

# Insert a program first
c.execute(
    '''INSERT INTO `Programs`
             (programHash)
             VALUES
             (?);
''', ("MyProgram", ))

c.execute(
    '''INSERT INTO `Runs`
             (program, papimode, options)
             VALUES
             (?, ?, ?);

''', (c.lastrowid, "default", "threads=4")
)  # lastrowid is used to simplify things a bit (could also use SELECT instead)

run_id = c.lastrowid
print("New run_id " + str(run_id))
order_entry = 0
# Create a supersection
for i in range(0, 10):
    c.execute(
        '''INSERT INTO `SuperSections`
                (runid, `order`, nodeID)
                VALUES
                (?, ?, ?);
    ''', (run_id, i, 42))

    ssid = c.lastrowid
    print("New ssid " + str(ssid))

    for j in range(0, 10):
        c.execute(
            '''INSERT INTO `Sections`
                    (ssid, `order`, nodeID)
                    VALUES
                    (?, ?, ?);
        ''', (ssid, 123 + j, 42))

        section_id = c.lastrowid
        for k in range(0, 64):
            c.execute(
                '''INSERT INTO `Entries`
                         (SectionID, `order`, threadID, iteration, flags)
                         VALUES
                         (?, ?, ?, ?, ?);
            ''', (section_id, order_entry, k // 16, k, 0))
            order_entry += 1

            entry_id = c.lastrowid
            for l in range(0, 4):
                c.execute(
                    '''INSERT INTO `Values`
                             (PapiCode, Value, entryID)
                             VALUES
                             (?, ?, ?)
                ''', (0xDEADBEEF + l, (l + 1) * 1000, entry_id))

conn.commit()

conn.close()
