FLAGS=-std=c++11 -O2 -Wall -Wextra -Wnon-virtual-dtor -pedantic
INC=-I/usr/local/Cellar/eigen/3.3.4/include/eigen3

ctxx: main.cpp
	g++ $(INC) $(FLAGS) -o ctxx main.cpp

clean:
	rm ctxx
