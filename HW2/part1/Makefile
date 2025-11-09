CXX = g++
CXXFLAGS = -O3 -Wall -mavx2 -Wextra -std=c++17

TARGETS = pi.out test_rng.out
SRC = pi.cpp test_rng.cpp

all: $(TARGETS)

pi.out: pi.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

test_rng.out: test_rng.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)