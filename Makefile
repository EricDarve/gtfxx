# default
#FLAGS=-std=c++11 -O2 -DNDEBUG
# debug
FLAGS=-std=c++11 -g -Wall -Wextra -Wnon-virtual-dtor -pedantic
# google profiler
#FLAGS=-DPROFILER -DNDEBUG -std=c++11 -O2 -lprofiler -Wl,-no_pie
# gcov
# FLAGS=-std=c++11 -O2 -DNDEBUG -fprofile-arcs -ftest-coverage

INC=-I/usr/local/Cellar/eigen/3.3.4/include/eigen3

ctxx: main.cpp
	g++ $(INC) $(FLAGS) -o ctxx main.cpp

clean:
	rm -f ctxx
