#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string>
#include "kmeans.h"
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include <omp.h>

#ifdef WIN
	#include <windows.h>
#else
	#include <pthread.h>
	#include <sys/time.h>
	double gettime() {
		struct timeval t;
		gettimeofday(&t,NULL);
		return t.tv_sec+t.tv_usec*1e-6;
	}
#endif


#ifdef NV 
	#include <oclUtils.h>
#else
	#include <CL/cl.h>
#endif

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

#ifdef RD_WG_SIZE_0_0
        #define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define BLOCK_SIZE RD_WG_SIZE
#else
        #define BLOCK_SIZE 512
#endif

#ifdef RD_WG_SIZE_1_0
     #define BLOCK_SIZE2 RD_WG_SIZE_1_0
#elif defined(RD_WG_SIZE_1)
     #define BLOCK_SIZE2 RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)
     #define BLOCK_SIZE2 RD_WG_SIZE
#else
     #define BLOCK_SIZE2 512
#endif



// local variables
static cl_platform_id  *platform_ids;
static cl_context	    context;
static cl_command_queue cmd_queue;
static cl_device_type   device_type;
static cl_device_id    *device_list;
static cl_int           num_devices;

static int initialize()
{
	cl_int result;
	size_t size;
	cl_uint num_platforms;

	printf("Initializing\n");
	if (clGetPlatformIDs(0, NULL, &num_platforms) != CL_SUCCESS) { printf("ERROR: clGetPlatformIDs(0,NULL,&num_platforms) failed\n"); return -1; }
	printf("Number of platforms: %d\n", num_platforms);
	platform_ids = (cl_platform_id *) malloc(num_platforms * sizeof(cl_platform_id));
	if (clGetPlatformIDs(num_platforms, platform_ids, NULL) != CL_SUCCESS) { printf("ERROR: clGetPlatformIDs(num_platforms,platform_ids,NULL) failed\n"); return -1; }

	{
		char char_buffer[512]; 
		printf("Querying platform for info:\n");
		printf("==========================\n");
		clGetPlatformInfo(platform_ids[0], CL_PLATFORM_NAME, 512, char_buffer, NULL);
		printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
		clGetPlatformInfo(platform_ids[0], CL_PLATFORM_VENDOR, 512, char_buffer, NULL);
		printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
		clGetPlatformInfo(platform_ids[0], CL_PLATFORM_VERSION, 512, char_buffer, NULL);
		printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);
	}

	// Intel Altera is idx=0
	// cl_context_properties:
	// Specifies a list of context property names and their corresponding values. Each property name is immediately followed by the corresponding desired value.
	// The list is terminated with 0. properties can be NULL in which case the platform that is selected is implementation-defined.
	// The list of supported properties is described in the table below.
	cl_context_properties ctxprop[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform_ids[0], 0};
	context = clCreateContextFromType(ctxprop, CL_DEVICE_TYPE_ACCELERATOR, NULL, NULL, NULL);
	if( !context ) { printf("ERROR: clCreateContextFromType(%s) failed\n", "FPGA"); return -1; }

	// get the list of FPGAs
	result = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &size);

	num_devices = (int) (size / sizeof(cl_device_id));
	
	if( result != CL_SUCCESS || num_devices < 1 ) { printf("ERROR: clGetContextInfo() failed\n"); return -1; }
	device_list = new cl_device_id[num_devices];
	if( !device_list ) { printf("ERROR: new cl_device_id[] failed\n"); return -1; }
	result = clGetContextInfo( context, CL_CONTEXT_DEVICES, size, device_list, NULL );
	if( result != CL_SUCCESS ) { printf("ERROR: clGetContextInfo() failed\n"); return -1; }

	// create command queue for the first device
	cmd_queue = clCreateCommandQueue(context, device_list[0], 0, NULL);
	if( !cmd_queue ) { printf("ERROR: clCreateCommandQueue() FPGA failed\n"); return -1; }

	free(platform_ids);

	return 0;
}

static int shutdown()
{
	// release resources
	if( cmd_queue ) clReleaseCommandQueue( cmd_queue );
	if( context ) clReleaseContext( context );
	if( device_list ) delete device_list;

	// reset all variables
	cmd_queue = 0;
	context = 0;
	device_list = 0;
	num_devices = 0;
	device_type = 0;

	return 0;
}

cl_mem d_feature;
/*cl_mem d_feature_swap;*/
cl_mem d_cluster;
cl_mem d_membership;

cl_kernel kernel_s;
/*cl_kernel kernel2;*/

int   *membership_OCL;
int   *membership_d;
float *feature_d;
float *clusters_d;
float *center_d;

int allocate(float features[][NFEATURES])
{
	// OpenCL initialization
	if(initialize()) return -1;

	// compile kernel
	cl_int err = 0;

	/* ------------ */
	/* FPGA program */
	/* ------------ */

	cl_device_id device = device_list[0];

	// Create the FPGA program.
  	std::string binary_file = aocl_utils::getBoardBinaryFile("/home/mcanales/Heterogeniuses/opencl/kmeans/kmeans", device);
  	printf("Using AOCX: %s\n", binary_file.c_str());
  	cl_program prog = aocl_utils::createProgramFromBinary(context, binary_file.c_str(), &device, 1);
  	err = clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);
  	if (err != CL_SUCCESS) { printf("ERROR: FPGA clBuildProgram() => %d\n", err); return -1; }
	
	char * kernel_kmeans_c  = "kmeans_assign";
	/*char * kernel_swap  = "kmeans_swap";*/	
		
	kernel_s = clCreateKernel(prog, kernel_kmeans_c, &err);  
	if(err != CL_SUCCESS) { printf("ERROR: clCreateKernel() 0 => %d\n", err); return -1; }
/*	kernel2 = clCreateKernel(prog, kernel_swap, &err);  
	if(err != CL_SUCCESS) { printf("ERROR: clCreateKernel() 0 => %d\n", err); return -1; }*/
		
	clReleaseProgram(prog);	
	
	d_feature = clCreateBuffer(context, CL_MEM_READ_WRITE, NPOINTS * NFEATURES * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer d_feature (size:%d) => %d\n", NPOINTS * NFEATURES, err); return -1;}
	/*d_feature_swap = clCreateBuffer(context, CL_MEM_READ_WRITE, NPOINTS * NFEATURES * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer d_feature_swap (size:%d) => %d\n", NPOINTS * NFEATURES, err); return -1;}*/
	d_cluster = clCreateBuffer(context, CL_MEM_READ_WRITE, NCLUSTERS * NFEATURES  * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer d_cluster (size:%d) => %d\n", NCLUSTERS * NFEATURES, err); return -1;}
	d_membership = clCreateBuffer(context, CL_MEM_READ_WRITE, NPOINTS * sizeof(int), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer d_membership (size:%d) => %d\n", NPOINTS, err); return -1;}
		
	//write buffers
	err = clEnqueueWriteBuffer(cmd_queue, d_feature, 1, 0, NPOINTS * NFEATURES * sizeof(float), features[0], 0, 0, 0);
	if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer d_feature (size:%d) => %d\n", NPOINTS * NFEATURES, err); return -1; }
	
	/*clSetKernelArg(kernel2, 0, sizeof(void *), (void*) &d_feature);
	clSetKernelArg(kernel2, 1, sizeof(void *), (void*) &d_feature_swap);
	clSetKernelArg(kernel2, 2, sizeof(cl_int), (void*) &NPOINTS);
	clSetKernelArg(kernel2, 3, sizeof(cl_int), (void*) &NFEATURES);*/
	
	size_t global_work[3] = { NPOINTS, 1, 1 };
	/// Ke Wang adjustable local group size 2013/08/07 10:37:33
	/*size_t local_work_size= BLOCK_SIZE; // work group size is defined by RD_WG_SIZE_0 or RD_WG_SIZE_0_0 2014/06/10 17:00:51
	if(global_work[0]%local_work_size !=0) global_work[0]=(global_work[0]/local_work_size+1)*local_work_size;

	err = clEnqueueNDRangeKernel(cmd_queue, kernel2, 1, NULL, global_work, &local_work_size, 0, 0, 0);*/
	/*err = clEnqueueNDRangeKernel(cmd_queue, kernel2, 1, NULL, global_work, NULL, 0, 0, 0);
	if(err != CL_SUCCESS) { printf("ERROR: clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }*/
	
	posix_memalign((void **) &membership_OCL, ALIGNMENT, NPOINTS * sizeof(int));
}

void deallocateMemory()
{
	clReleaseMemObject(d_feature);
	//clReleaseMemObject(d_feature_swap);
	clReleaseMemObject(d_cluster);
	clReleaseMemObject(d_membership);
	free(membership_OCL);

}


int main( int argc, char** argv) 
{
	//printf("WG size of kernel_swap = %d, WG size of kernel_kmeans = %d \n", BLOCK_SIZE, BLOCK_SIZE2);

	setup(argc, argv);
	shutdown();
}

int	kmeansOCL(float features[][NFEATURES],    /* in: [npoints][nfeatures] */
           int    *membership,
		   float **clusters,
		   int     *new_centers_len,
           float  **new_centers)	
{
	double start = omp_get_wtime();
	double end;
	int delta = 0;
	int i, j, k;
	cl_int err = 0;
	
	err = clEnqueueWriteBuffer(cmd_queue, d_cluster, 1, 0, NCLUSTERS * NFEATURES * sizeof(float), clusters[0], 0, 0, 0);
	if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer d_cluster (size:%d) => %d\n", NPOINTS, err); return -1; }

	int size = 0; int offset = 0;

	//clSetKernelArg(kernel_s, 0, sizeof(void *), (void*) &d_feature_swap);
	clSetKernelArg(kernel_s, 0, sizeof(void *), (void*) &d_feature);
	clSetKernelArg(kernel_s, 1, sizeof(void *), (void*) &d_cluster);
	clSetKernelArg(kernel_s, 2, sizeof(void *), (void*) &d_membership);

	size_t global_work[3] = { NPOINTS, 1, 1 }; 

	/// Ke Wang adjustable local group size 2013/08/07 10:37:33
	size_t local_work_size=NFEATURES*NCLUSTERS; // work group size is defined by RD_WG_SIZE_1 or RD_WG_SIZE_1_0 2014/06/10 17:00:41
	if(global_work[0]%local_work_size !=0) global_work[0]=(global_work[0]/local_work_size+1)*local_work_size;

	err = clEnqueueNDRangeKernel(cmd_queue, kernel_s, 1, NULL, global_work, &local_work_size, 0, 0, 0);
	/*err = clEnqueueNDRangeKernel(cmd_queue, kernel_s, 1, NULL, global_work, NULL, 0, 0, 0);*/
	if(err != CL_SUCCESS) { printf("ERROR: clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }
	clFinish(cmd_queue);
	err = clEnqueueReadBuffer(cmd_queue, d_membership, 1, 0, NPOINTS * sizeof(int), membership_OCL, 0, 0, 0);
	if(err != CL_SUCCESS) { printf("ERROR: Memcopy Out\n"); return -1; }
	
	delta = 0;
	omp_set_num_threads(8);
	int cluster_id;
	end = omp_get_wtime();
	printf("kernel time: %lf\n", end - start);
	start = end;
	#pragma omp parallel for schedule(static) private(i,j,cluster_id) shared(membership_OCL,membership,new_centers,new_centers_len) reduction(+:delta)
	for (i = 0; i < NPOINTS; i++)
	{
		cluster_id = membership_OCL[i];
		new_centers_len[cluster_id]++;
		if (membership_OCL[i] != membership[i])
		{
			delta++;
			membership[i] = membership_OCL[i];
		}
		for (j = 0; j < NFEATURES; j++)
		{
			new_centers[cluster_id][j] += features[i][j];
		}
	}
	end = omp_get_wtime();
	printf("omp reduction time: %lf\n", end - start);

	return delta;
}
