#!/bin/sh
docker build -t diode2 -f DockerfileFlask .. && echo "BUILD SUCCESSFUL"
echo "Build done. Run with:"
echo "docker run -dit --name diode2_srv -p <yourport>:5000 diode2"
echo "Example: 'docker run -dit --name diode2_srv -p 12345:5000 diode2'"
