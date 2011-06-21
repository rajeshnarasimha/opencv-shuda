CC      = colorgcc
INCLUDES= -I$(LINK)/include -I$(SRC)/btl-shuda/btl-extra/include -I$(SRC)/btl-shuda/btl/include -I$(LINK)/include/eigen3 -I$(LIB)/kinect/OpenNI-OpenNI-unstable/Include 
LIBS    = -L$(LINK)/lib -lopencv_core -lopencv_highgui -lopencv_calib3d -lopencv_features2d -lopencv_video -L$(SRC)/btl-shuda/build/lib -lBtlExtraVideoSource -L$(LIB)/kinect/OpenNI-OpenNI-unstable/Platform/Linux-x86/Bin/Release -lOpenNI -lboost_filesystem -lboost_serialization -lyaml-cpp -L/usr/lib -lglut -lGLU
CCFLAGS = -g -lstdc++

Exe: main.cpp
	$(CC) -o  Exe main.cpp $(INCLUDES) $(LIBS) $(CCFLAGS) 

Debug: debug.cpp
	$(CC) -o  Debug debug.cpp $(INCLUDES) $(LIBS) $(CCFLAGS) 

