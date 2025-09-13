CXX      ?= c++
CXXFLAGS ?= -O3
CXXFLAGS += -std=c++17 -Wall -Wextra -pthread

.PHONY: all test clean

all: build/libtm.a

test: build/test
	./build/test

clean:
	$(RM) -r build

build/tm.o: tm.cpp tm.hpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -c -o $@ $<

build/libtm.a: build/tm.o
	$(AR) rcs $@ $^

build/test: test.cpp tm.hpp build/libtm.a
	$(CXX) $(CXXFLAGS) -o $@ test.cpp build/libtm.a
