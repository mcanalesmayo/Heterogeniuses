#include "load_balancer.h"

int bestFpgaWorkload(int npoints, int nfeatures, int nclusters){
	return (npoints/2)-((npoints/2)%(nfeatures*nclusters));
}