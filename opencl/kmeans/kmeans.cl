#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

#define SIZEOF_FLOAT 4

__attribute__((reqd_work_group_size(NFEATURES*NCLUSTERS,1,1)))
__kernel void kmeans_assign(__global float* restrict feature,   
              __global float* restrict clusters,
              __global int* restrict membership
              )
{
    __local float clusters_local[NFEATURES*NCLUSTERS*SIZEOF_FLOAT];
    __local float features_local[NFEATURES*NCLUSTERS*SIZEOF_FLOAT];
    int index;
    unsigned int gid = get_global_id(0);
    unsigned int lid = get_local_id(0);

    printf("Lid ->  %d   ---  Gid -> %d \n", gid, lid);

    clusters_local[lid] = clusters[lid];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int f=0; f<NFEATURES; f++){
        features_local[lid * NFEATURES + f] = feature[gid * NFEATURES + f];
    }

    float min_dist=FLT_MAX;
    for (int c=0; c < NCLUSTERS; c++) {
        float dist = 0.0;
        for (int f=0; f<NFEATURES; f++){
            dist += (features_local[lid * NFEATURES + f] - clusters_local[c * NFEATURES + f])* 
                   (features_local[lid * NFEATURES + f] - clusters_local[c * NFEATURES + f]);
        }

        if (dist < min_dist) {
            min_dist = dist;
            index = c;
        }
    }
    membership[gid] = index;
}