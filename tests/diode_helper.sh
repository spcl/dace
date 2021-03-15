#!/bin/bash
# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

# File helper that creates a diode server for test tasks to run with.
DIODEBASEPATH="../diode/"
SAMPLESBASEPATH="../samples/"
SERVPORT=`expr 6000 + ${CI_CONCURRENT_ID:-0}`

# Remove old config files if they exist
rm ./client_configs/default.conf
# Start the REST server
python3 $DIODEBASEPATH/diode_server.py --localhost --port $SERVPORT &
SERVPID=$!
RETVAL=0

echo "server pid is: $SERVPID, port: $SERVPORT"


# Wait 10 seconds for the server to come online
sleep 10

echo $1
sh -c "$1 > cmdout.txt"
RETVAL=$?

if cat cmdout.txt | grep -q '+ exit 1'; then
    RETVAL=1
elif cat cmdout.txt | grep -q '+ exit 0'; then
    RETVAL=0
else
    echo "Failed to get sensible output"
    RETVAL=1
fi

cat cmdout.txt
    
if [ $RETVAL -eq 0 ]; then
    echo "TEST COMMAND SUCCESSFUL"
fi

# Terminate the server
kill $SERVPID

exit $RETVAL
