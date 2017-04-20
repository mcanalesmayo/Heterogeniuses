#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

__attribute__((reqd_work_group_size(NCLUSTERS*NFEATURES,1,1)))
__kernel void kmeans_assign(__global float* restrict feature,   
              __global float* restrict clusters,
              __global float* restrict distances
              )
{
    __local float clusters_local[NCLUSTERS*NFEATURES];
    unsigned int gid = get_global_id(0);
    unsigned int lid = get_local_id(0);

    clusters_local[lid] = clusters[lid];
    barrier(CLK_LOCAL_MEM_FENCE);

    float min_dist=FLT_MAX;
    for (int c=0; c < NCLUSTERS; c++) {
        float dist = 0.0;
        #pragma unroll
        for (int f=0; f<NFEATURES; f++){
        	float diff = feature[f * NPOINTS + gid] - clusters_local[c * NFEATURES + f];
            dist += pown(diff, 2);
        }

        distances[gid * NCLUSTERS + c] = dist;
    }

    return;
}