CC      = g++ #gcc
CFLAGS	= -I$(LIB)/opencv/include/  
LIBS    = -L$(LIB)/opencv/lib -lopencv_core -lopencv_highgui -lopencv_imgproc

example: main.cpp
	$(CC) -o Exe  main.cpp $(CFLAGS) $(LIBS)
