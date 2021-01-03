
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

/*this template function allocates array memory and fills it with 
  float/double precision leibnitz elements
  */
template <class myType>
myType* fill_leibnitz_array(size_t free_mem)
{	
	int elements_number = (int)free_mem/sizeof(myType);

	myType* p_arr = new myType[elements_number];

	for (int i = 0; i < elements_number; i++) {
		myType sign;
		if (i % 2)  { sign = -1; }
		else	{ sign = 1; }

		p_arr[i] = (myType) 4 * (sign / (2 * (i + 1) - 1));
	}
	return p_arr;
}

int main()
{
	size_t free_mem, total_mem;
	cudaMemGetInfo(&free_mem, &total_mem);

	float* leibnitz_d = fill_leibnitz_array<float>(free_mem);

	float sum = 0;
	for (int i = 0; i < free_mem / sizeof(float); i++) {
		sum += leibnitz_d[i];
	}

	std::cout.precision(15);
	std::cout <<std::fixed << sum;


	delete[] leibnitz_d;

    return 0;
}


