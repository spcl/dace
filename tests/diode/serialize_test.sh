#!/bin/bash

DIODEBASEPATH="../../diode/"
SAMPLESBASEPATH="../../samples/"

# Remove old config files if they exist
rm ./client_configs/default.conf
# Start the REST server
python3 $DIODEBASEPATH/diode_rest.py --localhost --localdace &
SERVPID=$!
RETVAL=0

TESTFILES=("simple/gemm.py" "sdfg_api/nested_states.py" "sdfg_api/state_fusion.py" "sdfg_api/control_flow.py")
FAILEDFILES=()

echo "server pid is: $SERVPID"

# Wait 10 seconds for the server to come online
sleep 10

for t in "${TESTFILES[@]}"; do


    # Run the client(s) and check output
    cat $SAMPLESBASEPATH/$t | python3 $DIODEBASEPATH/diode_client.py --code --compile --extract outcode > "$(basename $t).from_code"

    #echo "Input file $t"
    #cat $SAMPLESBASEPATH/$t
    #echo "Output"
    #cat "$(basename $t).from_code"

    # Execute and check output
    cat $SAMPLESBASEPATH/$t | python3 $DIODEBASEPATH/diode_client.py --code --run > "$(basename $t).txt"
    if [ $? -ne 0 ]; then
        RETVAL=1
        FAILEDFILES+=("(output) $t")
    fi

    #echo "Execution ended, output:"
    #cat "$(basename $t).txt"
    #echo "Failed state:\n${FAILEDFILES[*]}"

    # Compile again and store entire output
    cat $SAMPLESBASEPATH/$t | python3 $DIODEBASEPATH/diode_client.py --code --compile --extract sdfg > "$(basename $t).serialized"

    #echo "Code extracted:"
    #cat "$(basename $t).serialized"

    # Compile from previous output and extract code (expect the same generated code as earlier)
    cat "$(basename $t).serialized" | python3 $DIODEBASEPATH/diode_client.py --compile --extract outcode > "$(basename $t).from_serialized"

    #echo "Re-serialized:"
    #cat "$(basename $t).from_serialized"

    diff "$(basename $t).from_code" "$(basename $t).from_serialized"
    if [ $? -ne 0 ]; then
        RETVAL=1
        FAILEDFILES+=($t)
    fi

done
# Terminate the server
kill $SERVPID

if [ "${#FAILEDFILES[@]}" -ne 0 ]; then
    echo "Failed tests:\n${FAILEDFILES[*]}"
fi

exit $RETVAL
