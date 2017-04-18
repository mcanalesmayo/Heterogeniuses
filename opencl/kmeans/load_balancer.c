#include "load_balancer.h"

int bestFpgaWorkload(int npoints, int nfeatures, int nclusters){
	return (npoints/3)-((npoints/3)%(nfeatures*nclusters));
}