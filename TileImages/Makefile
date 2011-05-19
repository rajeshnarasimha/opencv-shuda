CC      = g++ #gcc
CFLAGS	= -I$(LIB)/opencv/include/ -I$(LIB)/boost/include 
LIBS    = -L$(LIB)/opencv/lib -lopencv_core -lopencv_highgui 

example: main.cpp
	$(CC) -o Exe  main.cpp $(CFLAGS) $(LIBS)
