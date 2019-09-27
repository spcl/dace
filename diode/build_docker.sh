#!/bin/sh
docker build -t diode -f DockerfileFlask .. && echo "BUILD SUCCESSFUL"
echo "Build done. Run with:"
echo "docker run -dit --name diode_srv -p <yourport>:5000 diode"
echo "Example: 'docker run -dit --name diode_srv -p 12345:5000 diode'"
