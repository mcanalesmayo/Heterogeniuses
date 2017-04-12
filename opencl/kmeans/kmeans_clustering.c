/*****************************************************************************/
/*IMPORTANT:  READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.         */
/*By downloading, copying, installing or using the software you agree        */
/*to this license.  If you do not agree to this license, do not download,    */
/*install, copy or use the software.                                         */
/*                                                                           */
/*                                                                           */
/*Copyright (c) 2005 Northwestern University                                 */
/*All rights reserved.                                                       */

/*Redistribution of the software in source and binary forms,                 */
/*with or without modification, is permitted provided that the               */
/*following conditions are met:                                              */
/*                                                                           */
/*1       Redistributions of source code must retain the above copyright     */
/*        notice, this list of conditions and the following disclaimer.      */
/*                                                                           */
/*2       Redistributions in binary form must reproduce the above copyright   */
/*        notice, this list of conditions and the following disclaimer in the */
/*        documentation and/or other materials provided with the distribution.*/ 
/*                                                                            */
/*3       Neither the name of Northwestern University nor the names of its    */
/*        contributors may be used to endorse or promote products derived     */
/*        from this software without specific prior written permission.       */
/*                                                                            */
/*THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS    */
/*IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED      */
/*TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT AND         */
/*FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL          */
/*NORTHWESTERN UNIVERSITY OR ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT,       */
/*INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES          */
/*(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR          */
/*SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)          */
/*HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,         */
/*STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN    */
/*ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             */
/*POSSIBILITY OF SUCH DAMAGE.                                                 */
/******************************************************************************/

/*************************************************************************/
/**   File:         kmeans_clustering.c                                 **/
/**   Description:  Implementation of regular k-means clustering        **/
/**                 algorithm                                           **/
/**   Author:  Wei-keng Liao                                            **/
/**            ECE Department, Northwestern University                  **/
/**            email: wkliao@ece.northwestern.edu                       **/
/**                                                                     **/
/**   Edited by: Jay Pisharath                                          **/
/**              Northwestern University.                               **/
/**                                                                     **/
/**   ================================================================  **/
/**																		**/
/**   Edited by: Shuai Che, David Tarjan, Sang-Ha Lee					**/
/**				 University of Virginia									**/
/**																		**/
/**   Description:	No longer supports fuzzy c-means clustering;	 	**/
/**					only regular k-means clustering.					**/
/**					No longer performs "validity" function to analyze	**/
/**					compactness and separation crietria; instead		**/
/**					calculate root mean squared error.					**/
/**                                                                     **/
/*************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include "kmeans.h"

#define RANDOM_MAX 2147483647

extern double wtime(void);

/*----< kmeans_clustering() >---------------------------------------------*/
float** kmeans_clustering(float feature[][NFEATURES],    /* in: [NPOINTS][NFEATURES] */
                          float   threshold,
                          int    *membership) /* out: [NPOINTS] */
{    
    int      i, j, n = 0;				/* counters */
	int		 loop=0, temp;
    int     *new_centers_len;	/* [NCLUSTERS]: no. of points in each cluster */
    float    delta;				/* if the point moved */
    float  **clusters;			/* out: [NCLUSTERS][NFEATURES] */
    float  **new_centers;		/* [NCLUSTERS][NFEATURES] */

	int     initial[NPOINTS];		/* used to hold the index of points not yet selected
								   prevents the "birthday problem" of dual selection (?)
								   considered holding initial cluster indices, but changed due to
								   possible, though unlikely, infinite loops */
	int      initial_points;
	int		 c = 0;

	/* Visualization */
	char buffer[32]; // The filename buffer.

    /* allocate space for and initialize returning variable clusters[] */
    posix_memalign((void **) &clusters, ALIGNMENT, NCLUSTERS * sizeof(float *));
    posix_memalign((void **) &clusters[0], ALIGNMENT, NCLUSTERS * NFEATURES * sizeof(float));

    for (i=1; i<NCLUSTERS; i++)
        clusters[i] = clusters[i-1] + NFEATURES;

	/* initialize the random clusters */
	#pragma omp parallel for private(i) schedule(static)
	for (i = 0; i < NPOINTS; i++)
	{
		initial[i] = i;
	}
	initial_points = NPOINTS;

    /* randomly pick cluster centers */
    for (i=0; i<NCLUSTERS && initial_points >= 0; i++) {
		//n = (int)rand() % initial_points;		
		
        for (j=0; j<NFEATURES; j++)
            clusters[i][j] = feature[initial[n]][j];	// remapped

		/* swap the selected index to the end (not really necessary,
		   could just move the end up) */
		temp = initial[n];
		initial[n] = initial[initial_points-1];
		initial[initial_points-1] = temp;
		initial_points--;
		n++;
    }

	/* initialize the membership to -1 for all */
	#pragma omp parallel for private(i) schedule(guided)
    for (i=0; i < NPOINTS; i++) membership[i] = -1;

    /* allocate space for and initialize new_centers_len and new_centers */
    posix_memalign((void **) &new_centers_len, ALIGNMENT, NCLUSTERS * sizeof(int));
    posix_memalign((void **) &new_centers, ALIGNMENT, NCLUSTERS * sizeof(float *));
    posix_memalign((void **) &new_centers[0], ALIGNMENT, NCLUSTERS * NFEATURES * sizeof(float));

    // first iter
    new_centers_len[0] = 0;
    for (j=0; j<NFEATURES; j++){
        new_centers[0][j] = 0;
    }
    // remaining iters
    for (i=1; i<NCLUSTERS; i++){
        new_centers_len[i] = 0;
        new_centers[i] = new_centers[i-1] + NFEATURES;
        #pragma omp for private(i) schedule(guided)
        for(j=0; j<NFEATURES; j++){
            new_centers[i][j] = 0;
        }
    }

	/* iterate until convergence */
	do {
        delta = 0.0;
		// CUDA
		delta = (float) kmeansOCL(feature,			/* in: [NPOINTS][NFEATURES] */
								   membership,		/* which cluster the point belongs to */
								   clusters,		/* out: [NCLUSTERS][NFEATURES] */
								   new_centers_len,	/* out: number of points in each cluster */
								   new_centers		/* sum of points in each cluster */
								   );

		/* replace old cluster centers with new_centers */
		/* CPU side of reduction */
		#pragma omp parallel for schedule(guided) private(i,j) shared(new_centers_len,clusters,new_centers)
		for (i=0; i<NCLUSTERS; i++) {
			for (j=0; j<NFEATURES; j++) {
				if (new_centers_len[i] > 0)
					clusters[i][j] = new_centers[i][j] / new_centers_len[i];	/* take average i.e. sum/n */
				new_centers[i][j] = 0.0;	/* set back to 0 */
			}
			printf("Cluster %d: %d points\n", i, new_centers_len[i]);
			new_centers_len[i] = 0;			/* set back to 0 */
		}
		
		// Put "file" then k then ".txt" in to filename.
		snprintf(buffer, sizeof(char) * 32, "./files/file%i.txt",c);


		/* Saving the output */		
		FILE *f = fopen(buffer, "w");
		if (f == NULL)
		{
			printf("Error opening file!\n");
			exit(1);
		}		

		for(int i = 0; i < NPOINTS; i++) {
			/* Print membership and features */
			fprintf(f, "%d %f %f\n", membership[i], feature[i][0], feature[i][1]);
		} 

		fclose(f);

		c++;
    } while ((delta > threshold) && (loop++ < 500));	/* makes sure loop terminates */



	/* Saving the output */		
	FILE *f = fopen("file.txt", "w");
	if (f == NULL)
	{
	    printf("Error opening file!\n");
	    exit(1);
	}
	

	for(int i = 0; i < NPOINTS; i++) {
		/* print Membership and features */
		fprintf(f, "%d %f %f\n", membership[i], feature[i][0], feature[i][1]);
	} 

	fclose(f);

	printf("iterated %d times\n", c);
    free(new_centers[0]);
    free(new_centers);
    free(new_centers_len);

    return clusters;
}

