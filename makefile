XX=g++ -std=c++11
SRCS=test.cpp\
     Detection.cpp\

OBJS=$(SRCS:.cpp=.o)

EXEC=detection

start:$(OBJS)
	$(XX) -o $(EXEC) $(OBJS) `pkg-config --cflags --libs opencv4`
.cpp.o:
	$(XX) -o $@ -c $< `pkg-config --cflags --libs opencv4`

clean:
	rm -rf $(OBJS)
