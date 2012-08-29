#
# Makefile for svd test code
#

CXX = mpic++
CC = mpicc

LDFLAGS = -L ~/local/lib -Wl,-rpath=${HOME}/local/lib -fopenmp -ldoubanm -llapack
CFLAGS = -fopenmp -Wall -O3 -fpermissive
CXXFLAGS = -std=c++0x -std=gnu++0x -fopenmp -Wall -O3
CPPFLAGS = -I /home/wuhong_intern/local/include -I ../include -I/usr/include/mysql -DNDEBUG

all: main_svd_tr

main_svd_tr: main_svd_tr.o

clean:
	/bin/rm -f main_svd_tr *.o
