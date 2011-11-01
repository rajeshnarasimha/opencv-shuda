CC      = g++#colorgcc
INCLUDES= -I/usr/local/include/eigen3 -I$(SRC)/btl-shuda/btl-extra/include -I$(SRC)/btl-shuda/btl/include -I/usr/include/ni  
LIBS    = -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_calib3d -lopencv_features2d -lopencv_video -lopencv_imgproc -L$(SRC)/btl-shuda/build/lib -lBtlExtraVideoSource  -lOpenNI -lboost_filesystem -lboost_serialization -L$(LIB)/yaml-cpp/lib -lyaml-cpp -L/usr/lib -lglut -lGLU
CCFLAGS = -g -lstdc++

Exe: main.cpp
	$(CC) -o  Exe main.cpp $(INCLUDES) $(LIBS) $(CCFLAGS) 

Debug: debug.cpp
	$(CC) -o  Debug debug.cpp $(INCLUDES) $(LIBS) $(CCFLAGS) 

