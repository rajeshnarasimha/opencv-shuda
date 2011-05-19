CC      = colorgcc #g++ #gcc
CFLAGS	= -I$(SRC)/btl-shuda/btl/include -I$(LINK)/include -I$(LINK)/include/eigen3
LIBS    = -L$(LIB)/opencv/lib -lopencv_core -lopencv_highgui -lopencv_calib3d -L$(LIB)/boost/lib -lboost_filesystem -lboost_serialization -L/urs/lib -lglut -lGLU 
DEPS    = calibrationthroughimages.hpp 

%.o: %.cpp $(DEPS) 
	$(CC) -o $@ -c $< $(CFLAGS)

exe: main.o calibrationthroughimages.o
	$(CC) -o Exe  main.o calibrationthroughimages.o $(CFLAGS) $(LIBS)
