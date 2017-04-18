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
    int index;
	if (gid < npoints)
	{
		float min_dist=FLT_MAX;
		for (int i=0; i < NCLUSTERS; i++) {
			float dist = 0;
			float ans  = 0;
			for (int l=0; l<NFEATURES; l++){
					ans += (feature[l * npoints + gid]-clusters[i* NFEATURES + l])* 
						   (feature[l * npoints + gid]-clusters[i* NFEATURES + l]);
			}

			dist = ans;
			if (dist < min_dist) {
				min_dist = dist;
				index    = i;
			}
		}
		membership[gid] = index;
	}	
	
	return;
}

__kernel void
kmeans_swap(__global float  *feature,   
			__global float  *feature_swap,
			int npoints
){

	unsigned int gid = get_global_id(0);
    if (gid < npoints){
	    for(int i = 0; i <  NFEATURES; i++)
		    feature_swap[i * npoints + gid] = feature[gid * NFEATURES + i];
    }
} 
