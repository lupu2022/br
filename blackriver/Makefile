.PHONY: all

TARGET = br 

FLAGS = -std=c++11 -Wall -O3 
INC = -I. -I${YAML_PATH}/include 
LINK = -L${YAML_PATH}/lib  -lyaml-cpp 

all: $(TARGET)

$(TARGET): blackriver.cpp
	g++ $(FLAGS) -o $@ $^ $(INC) $(LINK) 

clean:
	rm -f $(TARGET)
