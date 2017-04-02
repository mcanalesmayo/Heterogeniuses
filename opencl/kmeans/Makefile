include ../../common/make.config
# C compiler
CC = g++
CC_FLAGS = -g -O2 -fopenmp
# CUDA
CUDA_OCL_LIB=/usr/local/cuda/lib64
CUDA_OCL_INC=/usr/local/cuda/include

# Intel Altera
# OpenCL compile and link flags.
AOCL_COMPILE_CONFIG := $(shell aocl compile-config )
AOCL_LINK_CONFIG := $(shell aocl link-config )
ALTERA_LIB_DIRS=
ALTERA_LIBS=rt
ALTERA_INC_DIRS := ../../common/inc
ALTERA_SRCS := ../../common/src/AOCLUtils/*.cpp

kmeans: cluster.o getopt.o read_input.o kmeans_clustering.o rmse.o
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) kmeans.cpp cluster.o getopt.o read_input.o kmeans_clustering.o rmse.o $(ALTERA_SRCS) -o kmeans.o -lOpenCL $(AOCL_COMPILE_CONFIG) $(AOCL_LINK_CONFIG) -I$(CUDA_OCL_INC) $(foreach D,$(ALTERA_INC_DIRS),-I$D) -L$(CUDA_OCL_LIB) $(foreach D,$(ALTERA_LIB_DIRS),-L$D) $(foreach L,$(ALTERA_LIBS),-l$L)

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

cluster.o: cluster.c 
	$(CC) $(CC_FLAGS) cluster.c -c

getopt.o: getopt.c 
	$(CC) $(CC_FLAGS) getopt.c -c

kmeans_clustering.o: kmeans_clustering.c
	$(CC) $(CC_FLAGS) kmeans_clustering.c -c

kmeans.o: kmeans.c 
	$(CC) $(CC_FLAGS) read_input.c -c

rmse.o: rmse.c
	$(CC) $(CC_FLAGS) rmse.c -c

clean:
	rm -f *.o *~ *.linkinfo
