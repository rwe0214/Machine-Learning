OUTDIR = .out
CC	= g++
CFLAGS	=-g -Wall

SRCS = $(wildcard ./*.cpp)
OBJS = $(patsubst ./%.cpp, $(OUTDIR)/%.o, $(SRCS)) 
EXE = sequential_estimator

all: $(OUTDIR) $(EXE)

$(OUTDIR):
	mkdir $(OUTDIR)

$(EXE) : $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ 

$(OUTDIR)/%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

.PHONY: all clean run debug

format:
	clang-format -i *.cpp *.h

run:
	./$(EXE)
	python3 plot.py

debug:
	gdb -q ./$(OUTDIR)/$(EXE)

clean:
	rm -f $(OUTDIR)/* ./$(EXE) ./output/*.data ./output/*.png
	rmdir ./$(OUTDIR) ./output
