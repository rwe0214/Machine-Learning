CC	= gcc
CFLAGS	=-g -Wall

all: linear_regression

%o: %c
	$(CC) -c $(CFLAGS) $<

linear_regression: matrix.o linear_regression.o
	$(CC) -o $@ $^

run:
	./linear_regression

clean:
	$(RM) linear_regression *.o