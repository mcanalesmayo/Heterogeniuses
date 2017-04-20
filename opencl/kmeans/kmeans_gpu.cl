#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

__kernel void
kmeans_kernel_assign(__global float* restrict feature,
					__global float* restrict clusters,
					__global float* restrict distances,
					int npoints
					)
{
	unsigned int gid = get_global_id(0);
	int index = 0;
	if (gid < npoints)
	{
		float min_dist=FLT_MAX;
		for (int c=0; c < NCLUSTERS; c++) {
			float dist = 0.0;
			for (int f=0; f<NFEATURES; f++){
				float diff = feature[f * NPOINTS + gid]-clusters[c * NFEATURES + f];
				dist += pown(diff, 2);
			}

			distances[gid * NCLUSTERS + c] = dist;
		}
	}
	
	return;
}

__kernel void
kmeans_swap(__global float* restrict feature,
			__global float* restrict feature_swap,
			int npoints
			)
{
	unsigned int gid = get_global_id(0);
	if (gid < npoints){
		for(int f = 0; f < NFEATURES; f++) feature_swap[f * NPOINTS + gid] = feature[gid * NFEATURES + f];
	}
}