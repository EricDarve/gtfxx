# default
FLAGS=-std=c++11 -O2 -DNDEBUG
# debug
#FLAGS=-std=c++11 -g -Wall -Wextra -Wnon-virtual-dtor -pedantic
# google profiler
#FLAGS=-DPROFILER -DNDEBUG -std=c++11 -O2 -lprofiler -Wl,-no_pie
# gcov
#FLAGS=-std=c++11 -O2 -DNDEBUG -fprofile-arcs -ftest-coverage

INC=-I/usr/local/Cellar/eigen/3.3.4/include/eigen3

CPP = icpc
#CPP = g++

.PHONY : clean test

all: ctxx

test : ctxx
	./ctxx --gtest_repeat=1 --gtest_break_on_failure

ctxx : main.cpp libgtest.a
	$(CPP) $(INC) $(FLAGS) -isystem googletest/include -pthread main.cpp libgtest.a -o ctxx

deque : deque.cpp threadpool.hpp libgtest.a
	$(CPP) $(FLAGS) -isystem googletest/include -pthread deque.cpp libgtest.a -o deque

libgtest.a :
	cd googletest; $(CPP) -isystem ./include -I. -pthread -c ./src/gtest-all.cc; ar -rv ../libgtest.a gtest-all.o

clean:
	rm -f ctxx libgtest.a
