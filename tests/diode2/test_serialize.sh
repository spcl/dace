#!/bin/bash

DIODE2BASEPATH="../../diode_2.0/"
SAMPLESBASEPATH="../../samples/"
# Start the REST server
python3 $DIODE2BASEPATH/diode_rest.py &
SERVPID=$!
RETVAL=0

TESTFILES=("simple/gemm.py" "sdfg_api/nested_states.py" "sdfg_api/state_fusion.py" "sdfg_api/control_flow.py")
FAILEDFILES=()

echo "server pid is: $SERVPID"

# Wait 5 seconds for the server to come online
sleep 5

for t in "${TESTFILES[@]}"; do


    # Run the client(s) and check output
    cat $SAMPLESBASEPATH/$t | python3 $DIODE2BASEPATH/diode2_client.py --code --compile --extract outcode > "$(basename $t).from_code"

    # Execute and check output
    cat $SAMPLESBASEPATH/$t | python3 $DIODE2BASEPATH/diode2_client.py --code --run > "$(basename $t).txt"
    if [ $? -ne 0 ]; then
        RETVAL=1
        FAILEDFILES+=("(output) $t")
    fi

    # Compile again and store entire output
    cat $SAMPLESBASEPATH/$t | python3 $DIODE2BASEPATH/diode2_client.py --code --compile --extract sdfg > "$(basename $t).serialized"

    # Compile from previous output and extract code (expect the same generated code as earlier)
    cat "$(basename $t).serialized" | python3 $DIODE2BASEPATH/diode2_client.py --compile --extract outcode > "$(basename $t).from_serialized"

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
