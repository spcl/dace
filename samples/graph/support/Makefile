.PHONY: all

all: libreadgr.so

libreadgr.so: readgr.cpp
	g++ -Wall -O3 readgr.cpp -shared -fPIC -o libreadgr.so

clean:
	rm -rf *~ libreadgr.so

