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
CURR_DIR = $(shell pwd )
ALTERA_LIB_DIRS=
ALTERA_LIBS=rt
ALTERA_INC_DIRS := ../../common/inc
ALTERA_SRCS := ../../common/src/AOCLUtils/*.cpp

kmeans: cluster.o getopt.o kmeans_clustering.o read_input.o rmse.o load_balancer.o
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) -D NPOINTS=$(NPOINTS) -D NFEATURES=$(NFEATURES) -D NCLUSTERS=$(NCLUSTERS) -D AOCX_PATH=$(CURR_DIR)/kmeans kmeans.cpp cluster.o getopt.o kmeans_clustering.o read_input.o rmse.o load_balancer.o $(ALTERA_SRCS) -o kmeans.o -lOpenCL $(AOCL_COMPILE_CONFIG) $(AOCL_LINK_CONFIG) -I$(CUDA_OCL_INC) $(foreach D,$(ALTERA_INC_DIRS),-I$D) -L$(CUDA_OCL_LIB) $(foreach D,$(ALTERA_LIB_DIRS),-L$D) $(foreach L,$(ALTERA_LIBS),-l$L)

%.o: %.[ch]
	$(CC) $(CC_FLAGS) -D NPOINTS=$(NPOINTS) -D NFEATURES=$(NFEATURES) -D NCLUSTERS=$(NCLUSTERS) $< -c

cluster.o: cluster.c 
	$(CC) $(CC_FLAGS) -D NPOINTS=$(NPOINTS) -D NFEATURES=$(NFEATURES) -D NCLUSTERS=$(NCLUSTERS) cluster.c -c

getopt.o: getopt.c 
	$(CC) $(CC_FLAGS) -D NPOINTS=$(NPOINTS) -D NFEATURES=$(NFEATURES) -D NCLUSTERS=$(NCLUSTERS) getopt.c -c

kmeans_clustering.o: kmeans_clustering.c
	$(CC) $(CC_FLAGS) -D NPOINTS=$(NPOINTS) -D NFEATURES=$(NFEATURES) -D NCLUSTERS=$(NCLUSTERS) kmeans_clustering.c -c

read_input.o: read_input.c 
	$(CC) $(CC_FLAGS) -D NPOINTS=$(NPOINTS) -D NFEATURES=$(NFEATURES) -D NCLUSTERS=$(NCLUSTERS) read_input.c -c

rmse.o: rmse.c
	$(CC) $(CC_FLAGS) -D NPOINTS=$(NPOINTS) -D NFEATURES=$(NFEATURES) -D NCLUSTERS=$(NCLUSTERS) rmse.c -c

load_balancer.o: load_balancer.c
	$(CC) $(CC_FLAGS) -D NPOINTS=$(NPOINTS) -D NFEATURES=$(NFEATURES) -D NCLUSTERS=$(NCLUSTERS) load_balancer.c -c

kmeans.o: kmeans.cpp
	$(CC) $(CC_FLAGS) -lOpenCL kmeans.cpp -I$(OPENCL_INC) -L$(OPENCL_LIB) -c

clean:
	rm -f *.o *~ *.linkinfo