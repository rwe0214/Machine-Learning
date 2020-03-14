CC	= gcc
CFLAGS	=-g -Wall

all: linear_regression

%o: %c
	$(CC) -c $(CFLAGS) $<

linear_regression: matrix.o linear_regression.o
	$(CC) -o $@ $^

# If the first argument is "run"...
ifeq (run,$(firstword $(MAKECMDGOALS)))
  # use the rest as arguments for "run"
  RUN_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  # ...and turn them into do-nothing targets
  $(eval $(RUN_ARGS):;@:)
endif

run:
	./linear_regression $(RUN_ARGS)
	python3 plot.py

clean:
	$(RM) linear_regression *.o output/*.txt output/*.png