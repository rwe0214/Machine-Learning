CC	= gcc
CFLAGS	=-g -Wall

all: online_learning

%o: %c
	$(CC) -c $(CFLAGS) $<

online_learning: online_learning.o
	$(CC) -o $@ $^ -lm

run:
	./online_learning
	python3 plot.py

clean:
	$(RM) online_learning *.o output/*.txt output/*.png
	rmdir output
