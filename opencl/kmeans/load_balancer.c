#include "load_balancer.h"

int bestFpgaWorkload(int npoints, int nfeatures, int nclusters){
	return npoints;
	//return (npoints/4)-((npoints/4)%(nfeatures*nclusters));
}