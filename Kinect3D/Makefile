CC      = g++ #colorgcc
INCLUDES= -I$(USR)/include -I$(SRC)/btl-shuda/btl-extra/include -I$(SRC)/btl-shuda/btl/include -I$(USR)/include/eigen3 -I$(USR)/include/openni 
LIBS    = -L$(USR)/lib -lopencv_core -lopencv_highgui -lopencv_calib3d -L$(SRC)/btl-shuda/build/lib -lBtlExtraVideoSource -L$(LIB)/kinect/OpenNI-OpenNI-unstable/Platform/Linux-x86/Bin/Release -lOpenNI -lboost_filesystem -lboost_serialization -lyaml-cpp -L/usr/lib -lglut -lGLU
CCFLAGS = -g -lstdc++

Exe: main.cpp
	$(CC) -o  Exe main.cpp $(INCLUDES) $(LIBS) $(CCFLAGS) 
