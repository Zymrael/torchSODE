CC=gcc
CFLAGS=-I.
DEPS = lorenz.h

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

run_lorenz: main.o lorenz.o 
	$(CC) -o run_lorenz main.o lorenz.o 

main.o: main.c lorenz.h lorenz.c
	$(CC) -o main main.o lorenz.o 
lorenz.o: lorenz.c lorenz.h
	$(CC) -o test1 main.o lorenz.o 
