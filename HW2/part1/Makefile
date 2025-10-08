CXX = g++

CXXFLAGS = -Wall -Wextra -std=c++17

TARGET = pi.out
SRC = pi.cpp

# Generate object file names from source files
OBJS = $(SRC:.cpp=.o)

# Default target: build the executable
all: $(TARGET)

# Rule to link object files into the executable
$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET)

# Rule to compile .cpp files into .o files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)