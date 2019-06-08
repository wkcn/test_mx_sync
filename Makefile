all:
	g++ test_sync.cpp -ldl -fPIC -shared -std=c++11 -o lib.so
