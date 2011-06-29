CC      = g++ #gcc
INCLUDES= -I$(SRC)/btl-shuda/btl/include -I$(SRC)/btl-shuda/btl-extra/include -I$(LINK)/include -I/usr/local/include/eigen3 
CCFLAGS = -g
LIBS    = -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_calib3d -lboost_filesystem -lboost_serialization  -L/urs/lib -lglut -lGLU -L$(SRC)/btl-shuda/build/lib -lBtlExtraVideoSource -L$(LIB)/yaml-cpp/lib -lyaml-cpp
OBJDIR  = obj

all: dir1 exe

dir1:
	mkdir -p $(OBJDIR)

%.o: %.cpp $(DEPS) 
	$(CC) -o $(OBJDIR)/$@ -c $< $(INCLUDES) $(CCFLAGS)

exe: main.o 
	$(CC) -o Exe  $(OBJDIR)/main.o $(INCLUDES) $(LIBS) $(CCFLAGS)
