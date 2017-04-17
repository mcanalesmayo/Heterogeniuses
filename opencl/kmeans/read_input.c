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
/**   File:         example.c                                           **/
/**   Description:  Takes as input a file:                              **/
/**                 ascii  file: containing 1 data point per line       **/
/**                 binary file: first int is the number of objects     **/
/**                              2nd int is the no. of features of each **/
/**                              object                                 **/
/**                 This example performs a fuzzy c-means clustering    **/
/**                 on the data. Fuzzy clustering is performed using    **/
/**                 min to max clusters and the clustering that gets    **/
/**                 the best score according to a compactness and       **/
/**                 separation criterion are returned.                  **/
/**   Author:  Wei-keng Liao                                            **/
/**            ECE Department Northwestern University                   **/
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
#define _CRT_SECURE_NO_DEPRECATE 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <fcntl.h>
#include <omp.h>
#include "kmeans.h"
#include <unistd.h>

extern double wtime(void);



/*---< usage() >------------------------------------------------------------*/
void usage(char *argv0) {
    char *help =
        "\nUsage: %s [switches] -i filename\n\n"
		"    -i filename      :file containing data to be clustered\n"		
		"    -m max_nclusters :maximum number of clusters allowed    [default=5]\n"
        "    -n min_nclusters :minimum number of clusters allowed    [default=5]\n"
		"    -t threshold     :threshold value                       [default=0.001]\n"
		"    -l nloops        :iteration for each number of clusters [default=1]\n"
		"    -b               :input file is in binary format\n"
        "    -r               :calculate RMSE                        [default=off]\n"
		"    -o               :output cluster center coordinates     [default=off]\n";
    fprintf(stderr, help, argv0);
    exit(-1);
}

/*---< main() >-------------------------------------------------------------*/
int setup(int argc, char **argv) {
		int		opt;
 extern char   *optarg;
		char   *filename = 0;
		char	line[20480];
		int		isBinaryFile = 0;

		int	    threshold = 0;		/* default value */
		float	len;

		float   features[NPOINTS][NFEATURES] __attribute__ ((aligned (ALIGNMENT)));
		float **cluster_centres=NULL;
		int		i, j, index;
		int		nloops = 1;				/* default value */
				
		int		isRMSE = 0;		
		float	rmse;
		
		int		isOutput = 0;
		float	cluster_timing, io_timing;		

		/* obtain command line arguments and change appropriate options */
		while ( (opt=getopt(argc,argv,"i:t:m:n:l:bro"))!= EOF) {
        switch (opt) {
            case 'i': filename=optarg;
                      break;
            case 'b': isBinaryFile = 1;
                      break;            
            case 't': threshold=atoi(optarg);
                      break;
			case 'r': isRMSE = 1;
                      break;
			case 'o': isOutput = 1;
					  break;
		    case 'l': nloops = atoi(optarg);
					  break;
            case '?': usage(argv[0]);
                      break;
            default: usage(argv[0]);
                      break;
        }
    }

    if (filename == 0) usage(argv[0]);
		
	/* ============== I/O begin ==============*/
    /* get NFEATURES and NPOINTS */
    io_timing = omp_get_wtime();
    if (isBinaryFile) {		//Binary file input
        int infile;
        if ((infile = open(filename, O_RDONLY, "0600")) == -1) {
            fprintf(stderr, "Error: no such file (%s)\n", filename);
            exit(1);
        }

        read(infile, features, NPOINTS*NFEATURES*sizeof(float));

        close(infile);
    }
    else {
        FILE *infile;
        if ((infile = fopen(filename, "r")) == NULL) {
            fprintf(stderr, "Error: no such file (%s)\n", filename);
            exit(1);
		}

        for(i=0; i<NPOINTS; i++) {
            fgets(line, 20480, infile);
            if (strtok(line, " \t\n") == NULL) continue;            
            for (j=0; j<NFEATURES; j++) {
                features[i][j] = atof(strtok(NULL, " ,\t\n"));
            }
        }
        fclose(infile);
    }
    io_timing = omp_get_wtime() - io_timing;
	
	printf("\nI/O completed\n");
	printf("\nNumber of objects: %d\n", NPOINTS);
	printf("Number of features: %d\n", NFEATURES);	
	/* ============== I/O end ==============*/

	srand(7);												/* seed for future random number generator */

	/* ======================= core of the clustering ===================*/

    cluster_timing = omp_get_wtime();		/* Total clustering time */
	cluster_centres = NULL;
    index = cluster(features,				/* array: [NPOINTS][NFEATURES] */
					threshold,				/* loop termination factor */
				   &cluster_centres,		/* return: [best_nclusters][NFEATURES] */  
				   &rmse,					/* Root Mean Squared Error */
					isRMSE,					/* calculate RMSE */
					nloops);				/* number of iteration for each number of clusters */		
    
	cluster_timing = omp_get_wtime() - cluster_timing;

    /* =============== Command Line Output =============== */

    /* cluster center coordinates
       :displayed only for when k=1*/
    if(isOutput == 1) {
        printf("\n================= Centroid Coordinates =================\n");
        for(i = 0; i < NCLUSTERS; i++){
            printf("%d:", i);
            for(j = 0; j < NFEATURES; j++){
                printf(" %.2f", cluster_centres[i][j]);
            }
            printf("\n\n");
        }
    }

	/* =============== Command Line Output =============== */

	printf("Number of Iteration: %d\n", nloops);
	printf("Time for I/O: %.5fsec\n", io_timing);
	printf("Time for Entire Clustering: %.5fsec\n", cluster_timing);
	
	if(nloops != 1){									// single k, multiple iteration
		if(isRMSE)										// if calculated RMSE
			printf("Number of trials to approach the best RMSE of %.3f is %d\n", rmse, index + 1);
	}
	else{												// single k, single iteration				
		if(isRMSE)										// if calculated RMSE
			printf("Root Mean Squared Error: %.3f\n", rmse);
	}
	
    return(0);
}

