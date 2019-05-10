import sqlite3
import os

# Connect to the database
conn = sqlite3.connect('perfdata.db')

# Get a transaction object
c = conn.cursor()

# Run some small queries first
c.execute("SELECT * FROM `Programs`;")
print("All programs:\n" + str(c.fetchall()))

c.execute("SELECT * FROM `Runs`;")
print("All Runs:\n" + str(c.fetchall()))

c.execute("SELECT * FROM `SuperSections`;")
fa = c.fetchall()
print("All SuperSections: " + str(len(fa)) + "\n" + str(fa))

c.execute("SELECT COUNT(*) FROM `Sections`;")
print("All Sections:\n" + str(c.fetchall()))

c.execute("SELECT COUNT(*) FROM `Entries`;")
print("All Entries:\n" + str(c.fetchall()))

q = '''
SELECT
    *
FROM
    (
        SELECT
            *
        FROM `Sections` AS sec
        WHERE
            sec.nodeID = ?
    ) sec
    INNER JOIN `Entries` AS es ON es.SectionID = sec.SectionID
    INNER JOIN `SuperSections` AS ss ON ss.SuperSectionID = sec.ssid
    INNER JOIN `Values` AS vs ON vs.entryID = es.EntryID
    INNER JOIN `Runs` AS rs ON rs.runID = ss.runid
WHERE
    sec.nodeID = ?
    AND rs.runID = 1
LIMIT 4
    ;
'''
c.execute(q, (5, 5))
# This seemingly overcounts by 4 - it actually doesn't. Multiply by the amount of counters!
fa = c.fetchall()
print("All Entries:" + str(len(fa)) + "\n" + str(fa))
for x in fa:
    print(str(x))

q = '''
SELECT
    rs.runID as `runid`, es.threadID as `thread_id`, vs.papicode as `code`, SUM(vs.Value) as `tot_val`
FROM
    (
        SELECT
            *
        FROM `Sections` AS sec
        WHERE
            sec.nodeID = ?
    ) sec
    INNER JOIN `Entries` AS es ON es.SectionID = sec.SectionID
    INNER JOIN `SuperSections` AS ss ON ss.SuperSectionID = sec.ssid
    INNER JOIN `Values` AS vs ON vs.entryID = es.EntryID
    INNER JOIN `Runs` AS rs ON rs.runID = ss.runid
WHERE
    sec.nodeID = ?
    --AND rs.runID = ?
GROUP BY
    rs.runID, es.threadID, vs.papicode, ss.SuperSectionID
    ;
'''
c.execute(q, (1, 1))
fa = c.fetchall()
print("Per-thread:" + str(len(fa)) + "\n")
for x in fa:
    print(str(x))

# Release the connection
conn.close()
