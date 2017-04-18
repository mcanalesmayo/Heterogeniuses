/*************************************************************************/
/**   File:         rmse.c												**/
/**   Description:  calculate root mean squared error of particular     **/
/**                 clustering.											**/
/**   Author:  Sang-Ha Lee												**/
/**            University of Virginia.									**/
/**																		**/
/**   Note: euclid_dist_2() and find_nearest_point() adopted from       **/
/**			Minebench code.												**/
/**                                                                     **/
/*************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <omp.h>

#include "kmeans.h"

extern double wtime(void);

/*----< euclid_dist_2() >----------------------------------------------------*/
/* multi-dimensional spatial Euclid distance square */
__inline
float euclid_dist_2(float *pt1,
                    float *pt2)
{
    int i;
    float ans=0.0;

    for (i=0; i<NFEATURES; i++)
        ans += (pt1[i]-pt2[i]) * (pt1[i]-pt2[i]);

    return(ans);
}

/*----< find_nearest_point() >-----------------------------------------------*/
__inline
int find_nearest_point(float  *pt,
                       float  **pts)
{
    int index, i;
    float max_dist=FLT_MAX;

    /* find the cluster center id with min distance to pt */
    for (i=0; i<NCLUSTERS; i++) {
        float dist;
        dist = euclid_dist_2(pt, pts[i]);  /* no need square root */
        if (dist < max_dist) {
            max_dist = dist;
            index    = i;
        }
    }
    return(index);
}

/*----< rms_err(): calculates RMSE of clustering >-------------------------------------*/
float rms_err	(float feature[][NFEATURES],   /* [NPOINTS][NFEATURES] */
                 float **cluster_centres              /* [NCLUSTERS][NFEATURES] */
                )
{
    int    i;
	int	   nearest_cluster_index;	/* cluster center id with min distance to pt */
    float  sum_euclid = 0.0;		/* sum of Euclidean distance squares */
    float  ret;						/* return value */
    
    /* calculate and sum the sqaure of euclidean distance*/	
    #pragma omp parallel for \
                shared(feature,cluster_centres) \
                private(i, nearest_cluster_index) \
                schedule (static)	
    for (i=0; i<NPOINTS; i++) {
        nearest_cluster_index = find_nearest_point(feature[i],
													cluster_centres);

		sum_euclid += euclid_dist_2(feature[i],
									cluster_centres[nearest_cluster_index]);
		
    }	
	/* divide by n, then take sqrt */
	ret = sqrt(sum_euclid / NPOINTS);

    return(ret);
}

