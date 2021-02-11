#!/bin/bash
# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
DIODEBASEPATH="$SCRIPTPATH/../../diode/"
SAMPLESBASEPATH="$SCRIPTPATH/../../samples/"
PORT=5002

# Remove old config files if they exist
rm ./client_configs/default.conf
# Start the REST server
python3 $DIODEBASEPATH/diode_server.py --localhost --port=$PORT &
SERVPID=$!
RETVAL=0

TESTFILES=("simple/matmul.py" "sdfg_api/nested_states.py" "sdfg_api/state_fusion.py" "sdfg_api/control_flow.py")
FAILEDFILES=()

echo "server pid is: $SERVPID"

# Wait 10 seconds for the server to come online
sleep 10

for t in "${TESTFILES[@]}"; do


    # Run the client(s) and check output
    cat $SAMPLESBASEPATH/$t | python3 $DIODEBASEPATH/diode_client.py -p $PORT --code --compile --extract outcode > "$(basename $t).from_code"
    if [ $? -ne 0 ]; then
        RETVAL=1
        FAILEDFILES+=("    $t (from code, generate)\n")
    fi

    #echo "Input file $t"
    #cat $SAMPLESBASEPATH/$t
    #echo "Output"
    #cat "$(basename $t).from_code"

    # Execute and check output
    cat $SAMPLESBASEPATH/$t | python3 $DIODEBASEPATH/diode_client.py -p $PORT --code --run > "$(basename $t).txt"
    if [ $? -ne 0 ]; then
        RETVAL=1
        FAILEDFILES+=("    $t (from code, run)\n")
    fi

    #echo "Execution ended, output:"
    #cat "$(basename $t).txt"
    #echo "Failed state:\n${FAILEDFILES[*]}"

    # Compile again and store entire output
    cat $SAMPLESBASEPATH/$t | python3 $DIODEBASEPATH/diode_client.py -p $PORT --code --compile --extract sdfg > "$(basename $t).serialized"

    if [ $? -ne 0 ]; then
        RETVAL=1
        FAILEDFILES+=("    $t (JSON serialized, generate)\n")
    fi

    #echo "Code extracted:"
    #cat "$(basename $t).serialized"

    # Compile from previous output and extract code (expect the same generated code as earlier)
    cat "$(basename $t).serialized" | python3 $DIODEBASEPATH/diode_client.py -p $PORT --compile --extract outcode > "$(basename $t).from_serialized"
    if [ $? -ne 0 ]; then
        RETVAL=1
        FAILEDFILES+=("    $t (JSON serialized, run)\n")
    fi

    #echo "Re-serialized:"
    #cat "$(basename $t).from_serialized"

    diff "$(basename $t).from_code" "$(basename $t).from_serialized"
    if [ $? -ne 0 ]; then
        RETVAL=1
        FAILEDFILES+=("    $t (JSON-code difference)\n")
    fi

done
# Terminate the server
kill $SERVPID

if [ "${#FAILEDFILES[@]}" -ne 0 ]; then
    printf "Failed tests:\n${FAILEDFILES[*]}"
fi

exit $RETVAL
