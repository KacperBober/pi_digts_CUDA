
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

	for (int i = 1; i <= elements_number; i++) {
		precisionType sign;
		if (i % 2 == 0)  { sign = -1; }
		else	{ sign = 1; }

		p_arr[i-1] = (precisionType) 4*(sign / (2*i - 1));
	}
	return p_arr;
}

template <class precisionType>
__global__ void sum_leibnitz_elements(precisionType *pointer, unsigned long summations_number) {

	unsigned long index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index < summations_number) {
			pointer[index] += pointer[summations_number + index];
		}
	/* if number of elements is not even, we have to add last matrix element*/
}

template <class precisionType>
void add_with_GPU(precisionType *leibnitz_arr, size_t free_mem) {

	unsigned long total_threads = free_mem / sizeof(precisionType);
	const int threads_per_block = 256;

	precisionType x = 0;
	precisionType *h_sum = &x;

	precisionType oszustwo = 0;
	
	precisionType * d_leibnitz;
	cudaMalloc(&d_leibnitz, free_mem);
	cudaMemcpy(d_leibnitz, leibnitz_arr, free_mem, cudaMemcpyHostToDevice);


	unsigned long summation_threads = total_threads;
	while(summation_threads >= 2) {

		if (summation_threads % 2 == 1) {
			oszustwo += leibnitz_arr[summation_threads];
		}

		summation_threads /= 2;
		int blocks_number = (summation_threads + threads_per_block - 1) / threads_per_block;
		sum_leibnitz_elements<precisionType><< < blocks_number, threads_per_block >> >(d_leibnitz, summation_threads);
	}

	cudaMemcpy(h_sum, d_leibnitz, sizeof(precisionType), cudaMemcpyDeviceToHost);
	*h_sum = *h_sum + oszustwo;

	std::cout << std::endl << *h_sum;
	
	cudaFree(d_leibnitz);
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
	free_mem = free_mem - 100000;
	//free_mem = free_mem - free_mem % 16; //making sure we have memory for multiple of 8
	
	
	float* leibnitz_f = fill_leibnitz_array<float>(free_mem);
	float sum_f = add_with_CPU<float>(leibnitz_f, free_mem);

	std::cout.precision(15);
	std::cout << std::fixed << sum_f;
	delete[] leibnitz_f;
	
	double* leibnitz_d = fill_leibnitz_array<double>(free_mem);
	double sum_d = add_with_CPU<double>(leibnitz_d, free_mem);

	std::cout.precision(15);
	std::cout << std::endl << std::fixed << sum_d;

	add_with_GPU<double>(leibnitz_d, free_mem);
	 
	delete[] leibnitz_d;


    return 0;
}


