#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string>
#include "load_balancer.h"
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

#define XSTR(x) #x
#define STR(x) XSTR(x)

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
	#define BLOCK_SIZE 128
#endif

#ifdef RD_WG_SIZE_1_0
	#define BLOCK_SIZE2 RD_WG_SIZE_1_0
#elif defined(RD_WG_SIZE_1)
	#define BLOCK_SIZE2 RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)
	#define BLOCK_SIZE2 RD_WG_SIZE
#else
	#define BLOCK_SIZE2 128
#endif

#define TWO_GPUS


// local variables
static cl_platform_id  *platform_ids;
static cl_context	    context_fpga;
static cl_context	    context_gpus;
//static cl_command_queue cmd_queue_fpga;
static cl_command_queue cmd_queue;
#ifdef TWO_GPUS
static cl_command_queue cmd_queue2;
#endif
static cl_device_id    *device_list_gpu;
//static cl_device_id    *device_list_fpga;
static cl_int           num_devices_gpu;
//static cl_int           num_devices_fpga;

//static float feature_swap[NFEATURES][NPOINTS] __attribute__ ((aligned (ALIGNMENT)));

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

	/* **** */
	/* GPUs */
	/* **** */

	// NVIDIA CUDA is idx=1
	cl_context_properties ctxprop_gpu[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform_ids[1], 0};
	context_gpus = clCreateContextFromType(ctxprop_gpu, CL_DEVICE_TYPE_GPU, NULL, NULL, NULL);
	if( !context_gpus ) { printf("ERROR: clCreateContextFromType(%s) failed\n", "FPGA"); return -1; }

	// {
	// 	char char_buffer[512]; 
	// 	printf("Querying platform for info:\n");
	// 	printf("==========================\n");
	// 	clGetPlatformInfo(platform_ids[1], CL_PLATFORM_NAME, 512, char_buffer, NULL);
	// 	printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
	// 	clGetPlatformInfo(platform_ids[1], CL_PLATFORM_VENDOR, 512, char_buffer, NULL);
	// 	printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
	// 	clGetPlatformInfo(platform_ids[1], CL_PLATFORM_VERSION, 512, char_buffer, NULL);
	// 	printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);
	// }

	// get the list of GPUs
	result = clGetContextInfo(context_gpus, CL_CONTEXT_DEVICES, 0, NULL, &size);
	num_devices_gpu = (int) (size / sizeof(cl_device_id));
	if( result != CL_SUCCESS || num_devices_gpu < 1 ) { printf("ERROR: clGetContextInfo() failed\n"); return -1; }
	device_list_gpu = new cl_device_id[num_devices_gpu];
	
	result = clGetContextInfo( context_gpus, CL_CONTEXT_DEVICES, size, device_list_gpu, NULL );
	if( result != CL_SUCCESS ) { printf("ERROR: clGetContextInfo() failed\n"); return -1; }

	// create command queue for the first device
	cmd_queue = clCreateCommandQueue( context_gpus, device_list_gpu[0], 0, NULL );
	if( !cmd_queue ) { printf("ERROR: clCreateCommandQueue() failed\n"); return -1; }
#ifdef TWO_GPUS
	cmd_queue2 = clCreateCommandQueue( context_gpus, device_list_gpu[1], 0, NULL );
	if( !cmd_queue2 ) { printf("ERROR: clCreateCommandQueue() failed\n"); return -1; }
#endif

	// {
	// 	char char_buffer[512]; 
	// 	printf("Querying platform for info:\n");
	// 	printf("==========================\n");
	// 	clGetPlatformInfo(platform_ids[0], CL_PLATFORM_NAME, 512, char_buffer, NULL);
	// 	printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
	// 	clGetPlatformInfo(platform_ids[0], CL_PLATFORM_VENDOR, 512, char_buffer, NULL);
	// 	printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
	// 	clGetPlatformInfo(platform_ids[0], CL_PLATFORM_VERSION, 512, char_buffer, NULL);
	// 	printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);
	// }

	// cl_context_properties:
	// Specifies a list of context property names and their corresponding values. Each property name is immediately followed by the corresponding desired value.
	// The list is terminated with 0. properties can be NULL in which case the platform that is selected is implementation-defined.
	// The list of supported properties is described in the table below.

	/* **** */
	/* FPGA */
	/* **** */

	// Intel Altera is idx=0
	/*cl_context_properties ctxprop_fpga[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform_ids[0], 0};
	context_fpga = clCreateContextFromType(ctxprop_fpga, CL_DEVICE_TYPE_ACCELERATOR, NULL, NULL, NULL);
	if( !context_fpga ) { printf("ERROR: clCreateContextFromType(%s) failed\n", "FPGA"); return -1; }

	// get the list of GPUs
	result = clGetContextInfo(context_gpus, CL_CONTEXT_DEVICES, 0, NULL, &size);
	num_devices_fpga = (int) (size / sizeof(cl_device_id));
	if( result != CL_SUCCESS || num_devices_fpga < 1 ) { printf("ERROR: clGetContextInfo() failed\n"); return -1; }
	device_list_fpga = new cl_device_id[num_devices_fpga];
	if( !device_list_fpga ) { printf("ERROR: new cl_device_id[] failed\n"); return -1; }
	result = clGetContextInfo( context_fpga, CL_CONTEXT_DEVICES, size, device_list_fpga, NULL );
	if( result != CL_SUCCESS ) { printf("ERROR: clGetContextInfo() failed\n"); return -1; }

	// create command queue for the first device
	cmd_queue_fpga = clCreateCommandQueue( context_fpga, device_list_fpga[0], 0, NULL );
	if( !cmd_queue_fpga ) { printf("ERROR: clCreateCommandQueue() failed\n"); return -1; }*/

	free(platform_ids);

	return 0;
}

static int shutdown()
{
	// release resources
	//if( cmd_queue_fpga ) clReleaseCommandQueue( cmd_queue_fpga );
	if( cmd_queue ) clReleaseCommandQueue( cmd_queue );
#ifdef TWO_GPUS	
	if( cmd_queue2 ) clReleaseCommandQueue( cmd_queue2 );
#endif
	if( context_gpus ) clReleaseContext( context_gpus );
	//if( context_fpga ) clReleaseContext( context_fpga );
	if( device_list_gpu ) delete device_list_gpu;
	//if( device_list_fpga ) delete device_list_fpga;

	// reset all variables
	//cmd_queue_fpga = 0;
	cmd_queue = 0;
#ifdef TWO_GPUS	
	cmd_queue2 = 0;
#endif
	context_gpus = 0;
	//context_fpga = 0;
	device_list_gpu = 0;
	//device_list_fpga = 0;
	num_devices_gpu = 0;
	//num_devices_fpga = 0;

	return 0;
}

//cl_mem d_feature_fpga;
/*cl_mem d_feature_swap_fpga;
cl_mem d_cluster_fpga;
cl_mem d_distances_fpga;
cl_mem d_membership_fpga;*/

cl_mem d_feature_gpu0;
cl_mem d_feature_swap_gpu0;
cl_mem d_cluster_gpu0;
cl_mem d_membership_gpu0;

#ifdef TWO_GPUS
cl_mem d_feature_gpu1;
cl_mem d_feature_swap_gpu1;
cl_mem d_cluster_gpu1;
cl_mem d_membership_gpu1;
#endif

//cl_kernel kernel_fpga;

cl_kernel kernel_assign_gpu1;
cl_kernel kernel_swap1;

#ifdef TWO_GPUS
cl_kernel kernel_assign_gpu2;
cl_kernel kernel_swap2;
#endif

float my_new_centers[NCLUSTERS][NFEATURES];
int my_new_centers_len[NCLUSTERS];
int *membership_OCL;
float *feature_d;
float *clusters_d;
float *center_d;

int npoints_fpga;
int npoints_gpu;

#ifdef TWO_GPUS
	int divider=2;
#else
	int divider=1; //TODO: ONLY WORKS IF NPOINTS is a multiple of 2
#endif

int allocate(float feature[][NFEATURES])
{
	/* ************ */
	/* GPUs program */
	/* ************ */

	int sourcesize = 1024*1024;
	char *source = (char *) calloc(sourcesize, sizeof(char)); 
	if(!source) { printf("ERROR: calloc(%d) failed\n", sourcesize); return -1; }

	// read the kernel core source
	char const *tempchar = "./kmeans_gpu.cl";
	FILE *fp = fopen(tempchar, "rb"); 
	if(!fp) { printf("ERROR: unable to open '%s'\n", tempchar); return -1; }
	fread(source + strlen(source), sourcesize, 1, fp);
	fclose(fp);
		
	// OpenCL initialization
	if(initialize()) return -1;

	// compile kernel
	cl_int err = 0;
	const char *slist[2] = { source, 0 };
	cl_program prog_gpus = clCreateProgramWithSource(context_gpus, 1, slist, NULL, &err);
	if(err != CL_SUCCESS) { printf("ERROR: clCreateProgramWithSource() => %d\n", err); return -1; }
	char options[64];
	sprintf(options, "-D NPOINTS=%d -D NFEATURES=%d -D NCLUSTERS=%d", NPOINTS, NFEATURES, NCLUSTERS);
	err = clBuildProgram(prog_gpus, 0, NULL, options, NULL, NULL);

	/*{ //show warnings/errors
		static char log[65536]; memset(log, 0, sizeof(log));
		cl_device_id device_id = 0;
		err = clGetContextInfo(context_gpus, CL_CONTEXT_DEVICES, sizeof(device_id), &device_id, NULL);
		clGetProgramBuildInfo(prog_gpus, device_id, CL_PROGRAM_BUILD_LOG, sizeof(log)-1, log, NULL);
		if(err || strstr(log,"warning:") || strstr(log, "error:")) printf("<<<<\n%s\n>>>>\n", log);
	}*/
	if(err != CL_SUCCESS) { printf("ERROR: GPU clBuildProgram() => %d\n", err); return -1; }
	
	char const *kernel_assign_gpu_name  = "kmeans_kernel_assign";
	char const *kernel_swap_name  = "kmeans_swap";	
		
	kernel_assign_gpu1 = clCreateKernel(prog_gpus, kernel_assign_gpu_name, &err);  
	if(err != CL_SUCCESS) { printf("ERROR: clCreateKernel() 0 => %d\n", err); return -1; }
	kernel_swap1 = clCreateKernel(prog_gpus, kernel_swap_name, &err);  
	if(err != CL_SUCCESS) { printf("ERROR: clCreateKernel() 0 => %d\n", err); return -1; }

#ifdef TWO_GPUS
	kernel_assign_gpu2 = clCreateKernel(prog_gpus, kernel_assign_gpu_name, &err);  
	if(err != CL_SUCCESS) { printf("ERROR: clCreateKernel() 0 => %d\n", err); return -1; }
	kernel_swap2 = clCreateKernel(prog_gpus, kernel_swap_name, &err);  
	if(err != CL_SUCCESS) { printf("ERROR: clCreateKernel() 0 => %d\n", err); return -1; }
#endif

	free(source);
	clReleaseProgram(prog_gpus);

	//int div_points=NPOINTS;
	npoints_fpga = bestFpgaWorkload(NPOINTS, NFEATURES, NCLUSTERS);
	npoints_gpu = (NPOINTS - npoints_fpga)/divider;
	
	printf("Points for FPGA: %d, Points for each GPU: %d\n", npoints_fpga, npoints_gpu);
	/* Whole swap on a single GPU */
	d_feature_gpu0 = clCreateBuffer(context_gpus, CL_MEM_READ_ONLY, NPOINTS * NFEATURES * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer d_feature_gpu0 (size:%d) => %d\n", NPOINTS * NFEATURES, err); return -1;}
	d_feature_swap_gpu0 = clCreateBuffer(context_gpus, CL_MEM_READ_WRITE, npoints_gpu * NFEATURES * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer d_feature_swap_gpu0 (size:%d) => %d\n", NPOINTS * NFEATURES, err); return -1;}
	d_cluster_gpu0 = clCreateBuffer(context_gpus, CL_MEM_READ_ONLY, NCLUSTERS * NFEATURES * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer d_cluster_gpu0 (size:%d) => %d\n", NCLUSTERS * NFEATURES, err); return -1;}
	d_membership_gpu0 = clCreateBuffer(context_gpus, CL_MEM_WRITE_ONLY, npoints_gpu * sizeof(int), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer d_membership_gpu0 (size:%d) => %d\n", npoints_gpu, err); return -1;}

#ifdef TWO_GPUS		
	d_feature_gpu1 = clCreateBuffer(context_gpus, CL_MEM_READ_ONLY, NPOINTS * NFEATURES * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer d_feature_gpu1 (size:%d) => %d\n", NPOINTS * NFEATURES, err); return -1;}
	d_feature_swap_gpu1 = clCreateBuffer(context_gpus, CL_MEM_READ_WRITE, npoints_gpu * NFEATURES * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer d_feature_swap_gpu1 (size:%d) => %d\n", NPOINTS * NFEATURES, err); return -1;}
	d_cluster_gpu1 = clCreateBuffer(context_gpus, CL_MEM_READ_ONLY, NCLUSTERS * NFEATURES  * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer d_cluster_gpu1 (size:%d) => %d\n", NCLUSTERS * NFEATURES, err); return -1;}
	d_membership_gpu1 = clCreateBuffer(context_gpus, CL_MEM_WRITE_ONLY, npoints_gpu * sizeof(int), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer d_membership_gpu1 (size:%d) => %d\n", npoints_gpu, err); return -1;}
#endif

	//CHANGED TO NON-BLOCKING
	//write buffers
	err = clEnqueueWriteBuffer(cmd_queue, d_feature_gpu0, 0, 0, npoints_gpu * NFEATURES * sizeof(float), feature[0], 0, 0, 0);
	if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer d_feature_gpu0 (size:%d) => %d\n", npoints_gpu * NFEATURES, err); return -1; }
#ifdef TWO_GPUS
	err = clEnqueueWriteBuffer(cmd_queue2, d_feature_gpu1, 0, 0, npoints_gpu * NFEATURES * sizeof(float), feature[npoints_gpu], 0, 0, 0);
	if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer d_feature_gpu1 (size:%d) => %d\n", npoints_gpu * NFEATURES, err); return -1; }
#endif

	// Let GPUs manage the feature swap, adding another kernel to the FPGA device would incur in overheading
	clSetKernelArg(kernel_swap1, 0, sizeof(void *), (void*) &d_feature_gpu0);
	clSetKernelArg(kernel_swap1, 1, sizeof(void *), (void*) &d_feature_swap_gpu0);
	clSetKernelArg(kernel_swap1, 2, sizeof(cl_int), (void*) &npoints_gpu);

#ifdef TWO_GPUS
	clSetKernelArg(kernel_swap2, 0, sizeof(void *), (void*) &d_feature_gpu1);
	clSetKernelArg(kernel_swap2, 1, sizeof(void *), (void*) &d_feature_swap_gpu1);
	clSetKernelArg(kernel_swap2, 2, sizeof(cl_int), (void*) &npoints_gpu);
#endif

	size_t global_work_gpus[3] = { npoints_gpu, 1, 1 };
	size_t local_work_size_gpus = BLOCK_SIZE;
	if(global_work_gpus[0]%local_work_size_gpus != 0) global_work_gpus[0] = (global_work_gpus[0]/local_work_size_gpus+1) * local_work_size_gpus;

	printf("%lu, %lu\n", global_work_gpus[0], local_work_size_gpus);
	err = clEnqueueNDRangeKernel(cmd_queue, kernel_swap1, 1, NULL, global_work_gpus, &local_work_size_gpus, 0, 0, 0);
	if(err != CL_SUCCESS) { printf("ERROR: GPU0 kernel_swap clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }

#ifdef TWO_GPUS
	err = clEnqueueNDRangeKernel(cmd_queue2, kernel_swap2, 1, NULL, global_work_gpus, &local_work_size_gpus, 0, 0, 0);
	if(err != CL_SUCCESS) { printf("ERROR: GPU1 kernel_swap clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }	
#endif



	/* ************ */
	/* FPGA program */
	/* ************ */

	/*cl_device_id device = device_list_fpga[0];

	// Create the FPGA program.
  	std::string binary_file = aocl_utils::getBoardBinaryFile(STR(AOCX_PATH), device);
  	
  	printf("Using AOCX: %s\n", binary_file.c_str());
  	cl_program prog_fpga = aocl_utils::createProgramFromBinary(context_fpga, binary_file.c_str(), &device, 1);
  	err = clBuildProgram(prog_fpga, 0, NULL, NULL, NULL, NULL);
  	if (err != CL_SUCCESS) { printf("ERROR: FPGA clBuildProgram() => %d\n", err); return -1; }
	
	char const *kernel_assign_fpga_name  = "kmeans_assign";*/
	/*char const *kernel_swap  = "kmeans_swap";*/	
		
	/*kernel_fpga = clCreateKernel(prog_fpga, kernel_assign_fpga_name, &err);  
	if(err != CL_SUCCESS) { printf("ERROR: clCreateKernel() 0 => %d\n", err); return -1; }*/
	/*kernel2 = clCreateKernel(prog_fpga, kernel_swap, &err);  
	if(err != CL_SUCCESS) { printf("ERROR: clCreateKernel() 0 => %d\n", err); return -1; }*/
		
	/*clReleaseProgram(prog_fpga);*/
	
	/*d_feature_fpga = clCreateBuffer(context_fpga, CL_MEM_READ_ONLY, npoints_fpga * NFEATURES * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer d_feature_fpga (size:%d) => %d\n", npoints_fpga * NFEATURES, err); return -1;}*/
	/*d_feature_swap_fpga = clCreateBuffer(context_fpga, CL_MEM_READ_ONLY, NPOINTS * NFEATURES * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer d_feature_swap_fpga (size:%d) => %d\n", NPOINTS * NFEATURES, err); return -1;}
	d_cluster_fpga = clCreateBuffer(context_fpga, CL_MEM_READ_ONLY, NCLUSTERS * NFEATURES  * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer d_cluster_fpga (size:%d) => %d\n", NCLUSTERS * NFEATURES, err); return -1;}
	d_distances_fpga = clCreateBuffer(context_fpga, CL_MEM_WRITE_ONLY, npoints_fpga * NCLUSTERS * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer d_distances_fpga (size:%d) => %d\n", npoints_fpga * NCLUSTERS, err); return -1;}*/


	/* Write feature_swap buffers to devices */
/*#ifdef TWO_GPUS
	err = clEnqueueReadBuffer(cmd_queue2, d_feature_swap_gpu1, 0, 0, div_points * NFEATURES * sizeof(float), feature_swap[div_points], 0, 0, 0);
#endif*/
	/*err = clEnqueueReadBuffer(cmd_queue, d_feature_swap_gpu0, 0, 0, NPOINTS * NFEATURES * sizeof(float), feature_swap[0], 0, 0, 0);

	clFinish(cmd_queue);
#ifdef TWO_GPUS
	//clFinish(cmd_queue2);
	err = clEnqueueWriteBuffer(cmd_queue2, d_feature_swap_gpu1, 0, 0, NPOINTS * NFEATURES * sizeof(float), feature_swap[0], 0, 0, 0);
	if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer d_feature_swap_gpu1 (size:%d) => %d\n", npoints_gpu * NFEATURES, err); return -1; }
#endif
	err = clEnqueueWriteBuffer(cmd_queue, d_feature_swap_gpu0, 0, 0, NPOINTS * NFEATURES * sizeof(float), feature_swap[0], 0, 0, 0);
	if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer d_feature_swap_gpu0 (size:%d) => %d\n", npoints_gpu * NFEATURES, err); return -1; }
	err = clEnqueueWriteBuffer(cmd_queue_fpga, d_feature_swap_fpga, 0, 0, NPOINTS * NFEATURES * sizeof(float), feature_swap[0], 0, 0, 0);
	if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer d_feature_swap_fpga (size:%d) => %d\n", NPOINTS * NFEATURES, err); return -1; }*/
	
	// point to clusters distances
	posix_memalign((void **) &membership_OCL, ALIGNMENT, NPOINTS * sizeof(int));
}

void deallocateMemory()
{
	clReleaseMemObject(d_feature_gpu0);
	clReleaseMemObject(d_feature_swap_gpu0);
	clReleaseMemObject(d_cluster_gpu0);
	clReleaseMemObject(d_membership_gpu0);

#ifdef TWO_GPUS
	clReleaseMemObject(d_feature_gpu1);
	clReleaseMemObject(d_feature_swap_gpu1);
	clReleaseMemObject(d_cluster_gpu1);
	clReleaseMemObject(d_membership_gpu1);
#endif

	//clReleaseMemObject(d_feature_fpga);
	/*clReleaseMemObject(d_feature_swap_fpga);
	clReleaseMemObject(d_cluster_fpga);
	clReleaseMemObject(d_distances_fpga);*/

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
	// double start = omp_get_wtime();
	// double end;
	int delta = 0;
	int i, j, k;
	cl_int err = 0;

	/* **** */
	/* GPUs */
	/* **** */

	size_t global_work_gpus[3] = { npoints_gpu, 1, 1 }; 
	size_t local_work_size_gpus = BLOCK_SIZE2;
	if(global_work_gpus[0]%local_work_size_gpus !=0) global_work_gpus[0]=(global_work_gpus[0]/local_work_size_gpus+1)*local_work_size_gpus;

	//CHANGED TO NON BLOCKING
	err = clEnqueueWriteBuffer(cmd_queue, d_cluster_gpu0, 0, 0, NCLUSTERS * NFEATURES * sizeof(float), clusters[0], 0, 0, 0);
	if(err != CL_SUCCESS) { printf("ERROR: GPU0 clEnqueueWriteBuffer d_cluster_gpu0 (size:%d) => %d\n", npoints_gpu, err); return -1; }
#ifdef TWO_GPUS
	err = clEnqueueWriteBuffer(cmd_queue2, d_cluster_gpu1, 0, 0, NCLUSTERS * NFEATURES * sizeof(float), clusters[0], 0, 0, 0);
	if(err != CL_SUCCESS) { printf("ERROR: GPU1 clEnqueueWriteBuffer d_cluster_gpu0 (size:%d) => %d\n", npoints_gpu, err); return -1; }
#endif
	int div_points=npoints_gpu;
					
	clSetKernelArg(kernel_assign_gpu1, 0, sizeof(void *), (void*) &d_feature_swap_gpu0);
	clSetKernelArg(kernel_assign_gpu1, 1, sizeof(void *), (void*) &d_cluster_gpu0);
	clSetKernelArg(kernel_assign_gpu1, 2, sizeof(void *), (void*) &d_membership_gpu0);
	clSetKernelArg(kernel_assign_gpu1, 3, sizeof(cl_int), (void*) &div_points);

#ifdef TWO_GPUS
	clSetKernelArg(kernel_assign_gpu2, 0, sizeof(void *), (void*) &d_feature_swap_gpu1);
	clSetKernelArg(kernel_assign_gpu2, 1, sizeof(void *), (void*) &d_cluster_gpu1);
	clSetKernelArg(kernel_assign_gpu2, 2, sizeof(void *), (void*) &d_membership_gpu1);
	clSetKernelArg(kernel_assign_gpu2, 3, sizeof(cl_int), (void*) &div_points);
#endif


	/* ************ */
	/* Enqueue jobs */
	/* ************ */

	err = clEnqueueNDRangeKernel(cmd_queue, kernel_assign_gpu1, 1, NULL, global_work_gpus, &local_work_size_gpus, 0, 0, 0);
	if(err != CL_SUCCESS) { printf("ERROR: GPU0 kernel_assign clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }

#ifdef TWO_GPUS
	err = clEnqueueNDRangeKernel(cmd_queue2, kernel_assign_gpu2, 1, NULL, global_work_gpus, &local_work_size_gpus, 0, 0, 0);
	if(err != CL_SUCCESS) { printf("ERROR: GPU1 kernel_assign clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }

	err = clEnqueueReadBuffer(cmd_queue2, d_membership_gpu1, 0, 0, npoints_gpu * sizeof(int), &membership_OCL[npoints_gpu], 0, 0, 0);
	if(err != CL_SUCCESS) { printf("ERROR: GPU1 Memcopy Out-> %d\n", err); return -1; }
#endif
	err = clEnqueueReadBuffer(cmd_queue, d_membership_gpu0, 0, 0, npoints_gpu * sizeof(int), membership_OCL, 0, 0, 0);
	if(err != CL_SUCCESS) { printf("ERROR: GPU0 Memcopy Out-> %d\n", err); return -1; }

#ifdef TWO_GPUS
	clFinish(cmd_queue2);
#endif
	clFinish(cmd_queue);


	/* ********* */
	/* Reduction */
	/* ********* */

	omp_set_num_threads(8);
	// end = omp_get_wtime();
	// printf("kernel time: %lf\n", end - start);
	// start = end;
	int cluster_id;

	#pragma omp parallel for private(i,j)
	for (i = 0; i < NCLUSTERS; i++){
		my_new_centers_len[i] = 0;
		for (j = 0; j < NFEATURES; j++){
			my_new_centers[i][j] = 0.0;
		}
	}

	#pragma omp parallel private(i,j,cluster_id) firstprivate(my_new_centers,my_new_centers_len) shared(features,membership_OCL,membership,new_centers,new_centers_len,delta)
	{
		#pragma omp for schedule(static) reduction(+:delta)
		for (i=0; i<NPOINTS; i++)
		{
			// get closest cluster
			cluster_id = membership_OCL[i];
			// check membership
			my_new_centers_len[cluster_id]++;
			if (cluster_id != membership[i])
			{
				// update delta if changed
				delta++;
				membership[i] = cluster_id;
			}

			for (j = 0; j < NFEATURES; j++)
			{
				my_new_centers[cluster_id][j] += features[i][j];
			}
		}

		for(i = 0; i < NCLUSTERS; i++){
			#pragma omp atomic
			new_centers_len[i] += my_new_centers_len[i];
			for(j=0; j<NFEATURES; j++){
				#pragma omp atomic
				new_centers[i][j] += my_new_centers[i][j];	
			}
		}
	}

	// end = omp_get_wtime();
	// printf("omp reduction time: %lf\n", end - start);

	return delta;
}
