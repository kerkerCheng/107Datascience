all: main.exe

CXX = g++
CXXFLAGS = -O3 --std=c++11

main.exe: main.cpp fp_tree.hpp
	$(CXX) $(CXXFLAGS) main.cpp -o main.exe

.PHONY: clean

clean:
	rm main.exe
	