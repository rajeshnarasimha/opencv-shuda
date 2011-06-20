CC      = colorgcc
INCLUDES= -I$(LINK)/include -I$(LINK)/include/eigen3  -I$(SRC)/btl-shuda/btl/include 
LIBS    = -L$(LINK)/lib -lopencv_core -lopencv_highgui -lopencv_calib3d -lopencv_flann -lopencv_features2d
CCFLAGS = -g -lstdc++

Exe: find_obj.cpp
	$(CC) -o  find_obj find_obj.cpp $(INCLUDES) $(LIBS) $(CCFLAGS)
