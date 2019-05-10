#!/bin/bash

quit()
{
   echo "script ending for $RESTSERVER and $HTTPSERVER"
   kill $RESTSERVER
   kill $HTTPSERVER
   exit 0
}

trap 'quit' QUIT
trap 'quit'  INT

python3 diode_rest.py&
RESTSERVER=$!
python3 -m http.server&
HTTPSERVER=$!

echo "Created pids $RESTSERVER and $HTTPSERVER"

while true ; do
  sleep 30
done


