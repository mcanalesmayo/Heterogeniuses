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
/**   File:         cluster.c                                           **/
/**   Description:  Takes as input a file, containing 1 data point per  **/
/**                 per line, and performs a fuzzy c-means clustering   **/
/**                 on the data. Fuzzy clustering is performed using    **/
/**                 min to max clusters and the clustering that gets    **/
/**                 the best score according to a compactness and       **/
/**                 separation criterion are returned.                  **/
/**   Author:  Brendan McCane                                           **/
/**            James Cook University of North Queensland.               **/
/**            Australia. email: mccane@cs.jcu.edu.au                   **/
/**                                                                     **/
/**   Edited by: Jay Pisharath, Wei-keng Liao                           **/
/**              Northwestern University.                               **/
/**																		**/
/**   ================================================================  **/
/**																		**/
/**   Edited by: Shuai Che, David Tarjan, Sang-Ha Lee					**/
/**				 University of Virginia									**/
/**																		**/
/**   Description:	No longer supports fuzzy c-means clustering;	 	**/
/**					only regular k-means clustering.					**/
/**					No longer performs "validity" function to analyze	**/
/**					compactness and separation criteria; instead		**/
/**					calculate root mean squared error.					**/
/**                                                                     **/
/*************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <float.h>
#include <omp.h>
#include "kmeans.h"

float	min_rmse_ref = FLT_MAX;		
extern double wtime(void);
	/* reference min_rmse value */

// large variables in the heap rather than the stack
static int membership[NPOINTS];				/* which cluster a data point belongs to */

/*---< cluster() >-----------------------------------------------------------*/
int cluster(float  features[][NFEATURES],				/* array: [NPOINTS][NFEATURES] */
            int    	threshold,				/* loop terminating factor */
            float ***cluster_centres,		/* out: [best_nclusters][NFEATURES] */
			float	*min_rmse,				/* out: minimum RMSE */
			int		 isRMSE,				/* calculate RMSE */
			int		 nloops					/* number of iteration for each number of clusters */
			)
{
	int		index =0;						/* number of iteration to reach the best RMSE */
	int		rmse;							/* RMSE for each clustering */
    float **tmp_cluster_centres;			/* hold coordinates of cluster centers */
	int		i;

	/* allocate device memory, invert data array (@ kmeans_cuda.cu) */
	allocate(features);

	/* iterate nloops times for each number of clusters */
	for(i = 0; i < nloops; i++)
	{
		/* initialize initial cluster centers, CUDA calls (@ kmeans_cuda.cu) */
		tmp_cluster_centres = kmeans_clustering(features,
												threshold,
												membership);
		if (*cluster_centres) {
			free((*cluster_centres)[0]);
			free(*cluster_centres);
		}

		*cluster_centres = tmp_cluster_centres;
        
				
		/* find the number of clusters with the best RMSE */
		if(isRMSE)
		{
			rmse = rms_err(features,
						   tmp_cluster_centres);
			
			if(rmse < min_rmse_ref){
				min_rmse_ref = rmse;			//update reference min RMSE
				*min_rmse = min_rmse_ref;		//update return min RMSE
				index = i;						//update number of iteration to reach best RMSE
			}
		}			
	}
	
	deallocateMemory();							/* free device memory (@ kmeans_cuda.cu) */

    return index;
}

