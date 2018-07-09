# default
FLAGS=-std=c++11 -O2 -DNDEBUG
# debug
#FLAGS=-std=c++11 -g -Wall -Wextra -Wnon-virtual-dtor -pedantic
# google profiler
#FLAGS=-DPROFILER -std=c++11 -O2 -Wall -Wextra -pedantic -lprofiler -Wl,-no_pie
#FLAGS=-DPROFILER -std=c++11 -O2 -lprofiler
#FLAGS=-DPROFILER -std=c++11 -lprofiler -Wl,-no_pie -g
# gcov
#FLAGS=-std=c++11 -fprofile-arcs -ftest-coverage

INC=-I/usr/local/Cellar/eigen/3.3.4/include/eigen3

ctxx: main.cpp
	g++ $(INC) $(FLAGS) -o ctxx main.cpp

clean:
	rm ctxx
