OBJDIR  = obj

HEADERS = calibratekinectextrinsics.hpp optimdepth.hpp
SOURCES = calibratekinectextrinsics.cpp optimdepth.cpp main.cpp 
OBJECTS = $(OBJDIR)/calibratekinectextrinsics.o $(OBJDIR)/main.o $(OBJDIR)/optimdepth.o 

CC      = colorgcc #g++ #gcc
INCLUDES= -I$(SRC)/btl-shuda/btl/include -I$(SRC)/btl-shuda/btl-extra/include -I$(LINK)/include -I/usr/local/include/eigen3 -I$(SRC)/opencv-shuda/Lib/CalibKinect/include
CCFLAGS = -g
LIBS    = -L/$(LINK)/lib  -lopencv_core -lopencv_highgui -lopencv_calib3d -lboost_filesystem -lboost_serialization -L/urs/lib -lglut -lGLU -L$(SRC)/btl-shuda/build/lib -lBtlExtraVideoSource -lyaml-cpp

all: dir1 main.o calibratekinectextrinsics.o optimdepth.o exe

dir1:
	mkdir -p $(OBJDIR)

main.o: main.cpp
	$(CC) -o $(OBJDIR)/main.o -c main.cpp $(INCLUDES) $(CCFLAGS)

calibratekinectextrinsics.o: calibratekinectextrinsics.cpp
	$(CC) -o $(OBJDIR)/calibratekinectextrinsics.o -c calibratekinectextrinsics.cpp $(INCLUDES) $(CCFLAGS)

optimdepth.o: optimdepth.cpp
	$(CC) -o $(OBJDIR)/optimdepth.o -c optimdepth.cpp $(INCLUDES) $(CCFLAGS)

exe: $(OBJECTS)
	$(CC) -o Exe  $(OBJECTS) $(INCLUDES) $(LIBS) $(CCFLAGS)
