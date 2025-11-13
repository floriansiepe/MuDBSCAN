MPICXX ?= mpicxx
# Generic C/C++ compiler (may be set by environment); we avoid using it for MPI-linked builds
CC ?= cc

# Set CXX to MPICXX so implicit make rules use the MPI wrapper for C++ files
CXX := $(MPICXX)

# Verify MPICXX is available; if not, provide a helpful message when running make
ifeq ($(shell which $(MPICXX) 2>/dev/null),)
$(error "MPICXX ($(MPICXX)) not found in PATH. Load your MPI module or set MPICXX to the MPI compiler wrapper.")
endif

# Compiler flags for compilation of .cpp -> .o
CXXFLAGS ?= -c -std=c++11 -Wall -Wextra

# Use MPICXX for compilation/linking so MPI include paths/libs are used automatically
COMPILER := $(MPICXX)

# Generic pattern rule for building .o from .cpp using the chosen COMPILER
%.o: %.cpp
	$(COMPILER) $(CXXFLAGS) $< -o $@

output: vectorc.o Data.o GList.o RList.o MuC.o RTree.o MuC_RTree.o partition.o clustering.o main.o
	$(COMPILER) -o output vectorc.o Data.o GList.o RList.o MuC.o RTree.o MuC_RTree.o partition.o clustering.o main.o -lm

main.o: main.cpp
	$(COMPILER) $(CXXFLAGS) main.cpp -lm

vectorc.o: vectorc.cpp
	$(COMPILER) $(CXXFLAGS) vectorc.cpp -lm

Data.o: Data.cpp
	$(COMPILER) $(CXXFLAGS) Data.cpp -lm

MuC_RTree.o: MuC_RTree.cpp
	$(COMPILER) $(CXXFLAGS) MuC_RTree.cpp  -lm

GList.o: GList.cpp
	$(COMPILER) $(CXXFLAGS) GList.cpp  -lm 

RList.o: RList.cpp
	$(COMPILER) $(CXXFLAGS) RList.cpp  -lm 

MuC.o: MuC.cpp
	$(COMPILER) $(CXXFLAGS) MuC.cpp  -lm
	 
RTree.o: RTree.cpp
	$(COMPILER) $(CXXFLAGS) RTree.cpp  -lm

partition.o: partition.cpp
	$(COMPILER) $(CXXFLAGS) partition.cpp  -lm

clustering.o: clustering.cpp
	$(COMPILER) $(CXXFLAGS) clustering.cpp  -lm

clean:
	rm *.o
	rm -f test*