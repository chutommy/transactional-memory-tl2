CXX      ?= c++
CXXFLAGS ?= -O3
CXXFLAGS += -std=c++17 -Wall -Wextra -pthread

.PHONY: all clean

all: build/libtm.a

clean:
	$(RM) -r build

build/tm.o: tm.cpp tm.hpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -c -o $@ $<

build/libtm.a: build/tm.o
	$(AR) rcs $@ $^
