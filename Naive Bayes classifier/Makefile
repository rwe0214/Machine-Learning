CC	= gcc
CFLAGS	=-g -Wall

all: naive_bayes_classifier

%o: %c
	$(CC) -c $(CFLAGS) $<

naive_bayes_classifier: mnist.o naive_bayes_classifier.o
	$(CC) -o $@ $^ -lm

clean:
	$(RM) *.o naive_bayes_classifier
