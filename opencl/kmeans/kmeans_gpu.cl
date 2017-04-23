#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

__kernel void
kmeans_kernel_assign(__global float* restrict feature,
					__global float* restrict clusters,
					__global int* restrict membership,
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
			float ans  = 0.0;
			for (int f=0; f<NFEATURES; f++){
				float diff = feature[f * npoints + gid]-clusters[c * NFEATURES + f];
				ans += pown(diff, 2);
			}

			dist = ans;
			if (dist < min_dist) {
				min_dist = dist;
				index = c;
			}
		}
		membership[gid] = index;
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
		for(int f = 0; f < NFEATURES; f++) feature_swap[f * npoints + gid] = feature[gid * NFEATURES + f];
	}
}