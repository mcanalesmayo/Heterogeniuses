#include "load_balancer.h"

int bestFpgaWorkload(int npoints, int nfeatures, int nclusters){
	return (npoints/128)-((npoints/128)%(nfeatures*nclusters));
}