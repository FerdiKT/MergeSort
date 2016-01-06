// http://stackoverflow.com/questions/3557221/how-do-i-measure-time-in-c
// http://geeksquiz.com/merge-sort/
// http://mc.stanford.edu/cgi-bin/images/3/34/Darve_cme343_cuda_3.pdf


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <windows.h>

#define N 131072
#define threadSize 1
//#define blockSize N/2


void merge(long arr[], int l, int m, int r)
{
	int i, j, k;
	int n1 = m - l + 1;
	int n2 = r - m;

	/* create temp arrays */
	long L[N/2], R[N/2];

	/* Copy data to temp arrays L[] and R[] */
	for (i = 0; i < n1; i++)
		L[i] = arr[l + i];
	for (j = 0; j < n2; j++)
		R[j] = arr[m + 1 + j];

	/* Merge the temp arrays back into arr[l..r]*/
	i = 0;
	j = 0;
	k = l;
	while (i < n1 && j < n2)
	{
		if (L[i] <= R[j])
		{
			arr[k] = L[i];
			i++;
		}
		else
		{
			arr[k] = R[j];
			j++;
		}
		k++;
	}

	/* Copy the remaining elements of L[], if there are any */
	while (i < n1)
	{
		arr[k] = L[i];
		i++;
		k++;
	}

	/* Copy the remaining elements of R[], if there are any */
	while (j < n2)
	{
		arr[k] = R[j];
		j++;
		k++;
	}
}

void mergeSort(long arr[], int l, int r)
{
	if (l < r)
	{
		int m = l + (r - l) / 2; //Same as (l+r)/2, but avoids overflow for large l and h
		mergeSort(arr, l, m);
		mergeSort(arr, m + 1, r);
		merge(arr, l, m, r);
	}
}


void printArray(long A[], int size)
{
	int i;
	for (i = 0; i < size; i++)
		printf("%d ", A[i]);
	printf("\n");
}

__global__ void gpu_MergeSort(long* source, long *dest, long size) {
	long index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < N)
	{
		long start = index * size;

		long middle = start + size / 2;

		long end = start + size;
		if (end > N)
		{
			end = N;
		}

		//printf("start: %d - Middle: %d - End: %d\n", start, middle, end);

		long i = start, j = middle;

		long k = start;
		while (i < middle && j < end) {
			if (source[i] <= source[j]) {
				dest[k] = source[i];
				i++;
			}
			else {
				dest[k] = source[j];
				j++;
			}
			k++;
		}
		while (i < middle) {
			dest[k] = source[i];
			k++;
			i++;
		}
		while (j < end) {
			dest[k] = source[j];
			k++;
			j++;
		}
	}
	__syncthreads();
}


int main()
{
	DWORD dwStartTime, dwElapsed;

    long a[N], b[N], *d_A, *d_B;
	
	for (size_t i = 0; i < N; i++)
	{
		a[i] = N - i;
		b[i] = N - i;
	}

	int arr_size = sizeof(b) / sizeof(b[0]);

	//printf("Given array is \n");
	//printArray(b, arr_size);
	dwStartTime = GetTickCount();
	// MERGE SORT WITH CPU
	mergeSort(b, 0, arr_size - 1);

	dwElapsed = GetTickCount() - dwStartTime;
	//printf("\nSorted array is \n");
	//printArray(b, arr_size);
	printf("Calculations with CPU took %d.%3d seconds to complete\n", dwElapsed / 1000, dwElapsed - dwElapsed / 1000);

	int size = N * sizeof(long);

	cudaMalloc((void**)&d_A, size);
	cudaMalloc((void**)&d_B, size);
	cudaMemcpy(d_A, a, size, cudaMemcpyHostToDevice);
	
	long blockSize = 0;
	
	dwStartTime = GetTickCount();
	// MERGE SORT WITH GPU
	for (size_t i = 2; i <= N; i=i*2)
	{
		blockSize = N / (threadSize * i);
		//printf("block: % d - thd: %d - i: %d\n", blockSize, threadSize, i);
		gpu_MergeSort <<<blockSize, threadSize>> >(d_A, d_B, i);
		cudaDeviceSynchronize();
		//cudaMemcpy(a, d_B, size, cudaMemcpyDeviceToHost);
		//printArray(a, arr_size);
		// Swap source with destination array
		long *temp = d_A;
		d_A = d_B;
		d_B = temp;
	}
	dwElapsed = GetTickCount() - dwStartTime;
	//printArray(a, arr_size);

	printf("Calculations with GPU took %d.%3d seconds to complete\n", dwElapsed / 1000, dwElapsed - dwElapsed / 1000);
    return 0;
}
