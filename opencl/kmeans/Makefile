include ../../common/make.config
# C compiler
CC = g++
CC_FLAGS = -g -O2 -fopenmp 
OPENCL_LIB=/usr/local/cuda/lib64
OPENCL_INC=/usr/local/cuda/include

kmeans: cluster.o getopt.o read_input.o kmeans_clustering.o rmse.o kmeans.o
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) -lOpenCL kmeans.o cluster.o getopt.o read_input.o kmeans_clustering.o rmse.o -o kmeans -I$(OPENCL_INC) -L$(OPENCL_LIB)

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

cluster.o: cluster.c 
	$(CC) $(CC_FLAGS) cluster.c -c

getopt.o: getopt.c 
	$(CC) $(CC_FLAGS) getopt.c -c

rmse.o: rmse.c
	$(CC) $(CC_FLAGS) rmse.c -c

kmeans.o: kmeans.cpp
	$(CC) $(CC_FLAGS) -lOpenCL kmeans.cpp -I$(OPENCL_INC) -L$(OPENCL_LIB) -c

clean:
	rm -f *.o *~ kmeans *.linkinfo
