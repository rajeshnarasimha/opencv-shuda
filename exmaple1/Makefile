CC      = g++ #gcc
CFLAGS	= -I$(LINK)/include -I$(SRC)/btl-shuda/btl-extra/include -I$(SRC)/btl-shuda/btl/include -I$(LINK)/include/eigen3
LIBS    = -L$(LINK)/lib -lopencv_core -lopencv_highgui -lopencv_calib3d -L$(SRC)/btl-shuda/build/lib -lBtlExtraVideoSource -lyaml-cpp -lboost_system -lboost_serialization -lboost_filesystem

Exe: main.cpp
	$(CC) -o  Exe main.cpp $(CFLAGS) $(LIBS)
