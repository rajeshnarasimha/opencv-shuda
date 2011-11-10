CC      = colorgcc #g++ #gcc
CFLAGS	= -I$(SRC)/opencv-shuda/ShudaLib -I/usr/local/include/eigen3
LIBS    = -L/usr/local/lib -lopencv_core -L$(SRC)/opencv-shuda/ShudaLib -lShudaLib -lopencv_highgui -lopencv_calib3d -lopencv_imgproc -lopencv_features2d -L$(LIB)/boost/lib -lboost_system -lboost_filesystem -lboost_serialization -L/urs/lib -lglut -lGLU 
DEPS    = calibrationthroughimages.hpp 

%.o: %.cpp $(DEPS) 
	$(CC) -o $@ -c $< $(CFLAGS)

exe: main.o calibrationthroughimages.o
	$(CC) -o Exe  main.o calibrationthroughimages.o $(CFLAGS) $(LIBS)
