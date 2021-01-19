
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <chrono>

double pi = 3.1415926535897932;
const int threads_per_block = 256;

/*this template function allocates array memory and fills it with 
  float/double precision leibnitz elements
  */

void print_stats(double time_cpu, double time_gpu) {
	double speed = time_cpu / time_gpu;
	if (speed < 1) {
		std::cout << "CPU byl szybszy " << 1 / speed << " razy\n\n";
	}
	else {
		std::cout << "GPU byl szybszy " << speed << " razy\n\n";
	}
}

double error(double answer) {
	double err = abs(answer - pi) * 100;
	return err;
}

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


/*adds left threads when number of threads is not a divisor of total array elements*/
template <class precisionType>
__global__ void addRestGpu(precisionType* leibnitz_ele_in, precisionType* data_left, unsigned long threads_number, unsigned long blocks_number) {
	__shared__ precisionType shared_data[threads_per_block];
	unsigned int t_local = threadIdx.x; //local thread in block

		shared_data[t_local] = data_left[t_local]; //each thread loads data to shared memory
		__syncthreads();

		unsigned int shared = threads_number;
		bool parity = false;
		while (shared > 1) {
			if (shared % 2 == 1) parity = false;
			else parity = true;

			shared = shared >>= 1;
			if (t_local < shared) {
				shared_data[t_local] += shared_data[t_local + shared];
			}
			if (!parity) {
				if (t_local == 0) {
					shared_data[0] += shared_data[shared * 2];

				}
			}
			__syncthreads();
		}

		if (t_local == 0) {
			if (blocks_number == 0) {
				leibnitz_ele_in[0] = shared_data[0]; //copying data to global memor
			}
			else {
				leibnitz_ele_in[0] += shared_data[0]; //copying data to global memor
			}

		}

	}


/*summs blocks of shared memory and returns values to working array*/
template <class precisionType>
__global__ void sum_leibnitz_shared(precisionType* leibnitz_ele_in, unsigned long threads_number, unsigned long blocks_number) {
	__shared__ precisionType shared_data[threads_per_block];

	unsigned int t_local = threadIdx.x; //local thread in block
	unsigned int t_global = blockIdx.x * blockDim.x + threadIdx.x; //global thread index

		shared_data[t_local] = leibnitz_ele_in[t_global]; //each thread loads data into shared memory
		__syncthreads();

			
		for (unsigned int shared = blockDim.x / 2; shared > 0; shared >>= 1) { //divide by 2 and add 
				if (t_local < shared) {
					shared_data[t_local] += shared_data[t_local + shared];
				}
				__syncthreads();
			}

			if (t_local == 0) leibnitz_ele_in[blockIdx.x] = shared_data[0]; //copying data to global memor
}


template <class precisionType>
__global__ void sum_leibnitz_elements(precisionType *pointer, unsigned long summations_number, bool parity) {

	unsigned long index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index < summations_number) {
			pointer[index] += pointer[summations_number + index];
		}

		if (!parity) {
			if (index == 0) {
				pointer[index] += pointer[summations_number *2];
			}
		}
}

template <class precisionType>
void add_with_GPU(precisionType *leibnitz_arr, size_t free_mem, double czas) {

	unsigned long total_threads = free_mem / sizeof(precisionType);

	precisionType x = 0;
	precisionType *h_sum = &x;
	
	//////////////////////////////GPU no shared//////////////////////////////////////////

	precisionType * d_leibnitz;
	cudaMalloc(&d_leibnitz, free_mem);
	cudaMemcpy(d_leibnitz, leibnitz_arr, free_mem, cudaMemcpyHostToDevice);

	std::chrono::steady_clock::time_point beginGPU = std::chrono::steady_clock::now();
	bool parity = false;
	unsigned long summation_threads = total_threads;

	while(summation_threads > 1) {

		if (summation_threads % 2 == 1) {
			parity = false;
		}
		else parity = true;

		summation_threads /= 2;
		int blocks_number = (summation_threads + threads_per_block - 1) / threads_per_block;
		sum_leibnitz_elements<precisionType><< < blocks_number, threads_per_block >> >(d_leibnitz, summation_threads, parity);

	}

	std::chrono::steady_clock::time_point endGPU = std::chrono::steady_clock::now();
	double gpuTime = std::chrono::duration_cast<std::chrono::microseconds>(endGPU - beginGPU).count();

	cudaMemcpy(h_sum, d_leibnitz, sizeof(precisionType), cudaMemcpyDeviceToHost);
	std::cout << "Czas GPU = " << gpuTime << " mikrosekund, wynik GPU = " << *h_sum << ", blad = " << error(*h_sum) << "%" << std::endl;
	std::cout << "GPU bez shared byl szybszy " << czas / gpuTime << " razy od CPU" << std::endl;
	cudaFree(d_leibnitz);

	////////////////////////////////////GPU with shared//////////////////////////////////

	precisionType * ds_leibnitz;
	cudaMalloc(&ds_leibnitz, free_mem);
	cudaMemcpy(ds_leibnitz, leibnitz_arr, free_mem, cudaMemcpyHostToDevice);

	std::chrono::steady_clock::time_point beginGPUshared = std::chrono::steady_clock::now();
	summation_threads = total_threads;
	unsigned long rest = summation_threads % threads_per_block;
	while (summation_threads > 0) {

		int blocks_number = (summation_threads-rest + threads_per_block - 1) / threads_per_block;
		sum_leibnitz_shared<precisionType> << < blocks_number, threads_per_block >> > (ds_leibnitz, summation_threads-rest, blocks_number);
		addRestGpu<precisionType> << <1, threads_per_block >> > (ds_leibnitz, ds_leibnitz + summation_threads - rest, rest, blocks_number);

		summation_threads /= threads_per_block;
		rest = summation_threads % threads_per_block;

	}


	std::chrono::steady_clock::time_point endGPUshared = std::chrono::steady_clock::now();
	double gpu_time_shared = std::chrono::duration_cast<std::chrono::microseconds>(endGPUshared - beginGPUshared).count();

	cudaMemcpy(h_sum, ds_leibnitz, sizeof(precisionType), cudaMemcpyDeviceToHost);
	std::cout << "Czas GPU shared = " << gpu_time_shared << " mikrosekund, wynik GPU = " << *h_sum << ", blad = " << error(*h_sum) << "%" << std::endl;
	std::cout << "GPU z shared byl szybszy " << czas / gpu_time_shared << "razy od CPU oraz o " << gpuTime / gpu_time_shared << " razy od GPU bez uzycia shared" << std::endl;
	cudaFree(ds_leibnitz);
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
	free_mem = free_mem/3 * 2;
	free_mem = free_mem - free_mem % 16; //making sure we have memory for multiple of 8
	
	
	float* leibnitz_f = fill_leibnitz_array<float>(free_mem);

	std::chrono::steady_clock::time_point beginCPUf = std::chrono::steady_clock::now();
	float sum_f = add_with_CPU<float>(leibnitz_f, free_mem);
	std::chrono::steady_clock::time_point endCPUf = std::chrono::steady_clock::now();
	int cpu_f = std::chrono::duration_cast<std::chrono::microseconds>(endCPUf - beginCPUf).count();

	std::cout.precision(16);
	std::cout << "Wzor PI = " << pi << std::endl;
	std::cout << "-----------float-------------" << std::endl;
	std::cout << "Czas CPU = " << cpu_f << " mikrosekund, wynik CPU = " << sum_f << ", blad = " << error(sum_f) <<"%"<< std::endl;

	add_with_GPU<float>(leibnitz_f, free_mem, cpu_f);

	std::cout.precision(16);



	delete[] leibnitz_f;
	
	double* leibnitz_d = fill_leibnitz_array<double>(free_mem);

	std::chrono::steady_clock::time_point beginCPUd = std::chrono::steady_clock::now();
	double sum_d = add_with_CPU<double>(leibnitz_d, free_mem);
	std::chrono::steady_clock::time_point endCPUd = std::chrono::steady_clock::now();
	int cpu_d = std::chrono::duration_cast<std::chrono::microseconds>(endCPUd - beginCPUd).count();

	std::cout <<std::endl<< "----------double-------------" << std::endl;
	std::cout << "Czas CPU = " << cpu_d << " mikrosekund, wynik CPU = " << sum_d << ", blad = " << error(sum_d) << "%" << std::endl;

	add_with_GPU<double>(leibnitz_d, free_mem, cpu_d);

	 
	delete[] leibnitz_d;


    return 0;
}


