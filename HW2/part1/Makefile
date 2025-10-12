CXX = g++
CXXFLAGS = -O3 -Wall -mavx2 -march=native -Wextra -std=c++17

TARGET = pi.out
SRC = pi.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGET)