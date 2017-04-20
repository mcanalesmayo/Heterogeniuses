#include "load_balancer.h"

int bestFpgaWorkload(int npoints, int nfeatures, int nclusters){
	return (npoints/512)-((npoints/512)%(nfeatures*nclusters));
}