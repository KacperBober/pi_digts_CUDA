
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

/*this template function allocates array memory and fills it with 
  float/double precision leibnitz elements
  */
template <class precisionType>
precisionType* fill_leibnitz_array(size_t free_mem)
{	
	int elements_number = (int)free_mem/sizeof(precisionType);

	precisionType* p_arr = new precisionType[elements_number];

	for (int i = 0; i < elements_number; i++) {
		precisionType sign;
		if (i % 2)  { sign = -1; }
		else	{ sign = 1; }

		p_arr[i] = (precisionType) 4 * (sign / (2 * (i + 1) - 1));
	}
	return p_arr;
}

template <class precisionType>
precisionType add_with_GPU(precisionType *leibnitz_arr, size_t free_mem) {

	unsigned long total_threads = free_mem / sizeof(precisionType);
	const int threads_per_block = 256;
	int blocks_number = (total_threads + threads_per_block - 1) / threads_per_block;
	
	precisionType * d_leibnitz;
	cudaMalloc(&d_leibnitz, free_mem);
	cudaMemcpy(d_leibnitz, leibnitz_arr, free_mem, cudaMemcpyHostToDevice);
	
	/*
	for ()... {
		sum_leibnitz_elements << < blocks_number, threads_per_block >> > ()
	}*/
	
}

template <class precisionType>
__global__ void sum_leibnitz_elements(precisionType *pointer, long int summations_number, bool parity) {

	long int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < summations_number) {
		pointer[index] = pointer[index] + pointer[summations_number + index];
	}

	__syncthreads();	//avoid two threads trying to access one memory address at the same time 
	/* if number of elements is not even, we have to add last matrix element*/
	if (index == summations_number && !parity) {
		pointer[index - 1] = pointer[index - 1] + pointer[index];
	} 
}

template <class precisionType>
precisionType add_with_CPU(precisionType *leibnitz_arr, size_t free_mem) {
	
	precisionType sum = 0;
	for (int i = 0; i < free_mem / sizeof(precisionType); i++) {
		sum += leibnitz_arr[i];
	}

	return sum;
}


int main()
{
	size_t free_mem, total_mem;
	cudaMemGetInfo(&free_mem, &total_mem);

	free_mem = free_mem - free_mem % 8; //making sure we have memory for multiple of 8

	float* leibnitz_f = fill_leibnitz_array<float>(free_mem);
	float sum_f = add_with_CPU<float>(leibnitz_f, free_mem);

	std::cout.precision(15);
	std::cout << std::fixed << sum_f;
	delete[] leibnitz_f;

	double* leibnitz_d = fill_leibnitz_array<double>(free_mem);
	double sum_d = add_with_CPU<double>(leibnitz_d, free_mem);

	std::cout.precision(15);
	std::cout << std::endl << std::fixed << sum_d;

	 
	delete[] leibnitz_d;


    return 0;
}


