CC	= gcc
CFLAGS	=-g -Wall

all: naive_bayes_classifier

%o: %c
	$(CC) -c $(CFLAGS) $<


naive_bayes_classifier: mnist.o naive_bayes.o main.o
	$(CC) -o $@ $^ -lm

run:
	./naive_bayes_classifier

clean:
	$(RM) *.o naive_bayes_classifier

