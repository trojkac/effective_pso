
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

__global__ void initialize_particles(double** d_particles, double** d_velocities, double** d_personal_bests_locations, double* d_personal_bests_values, int dim, int nr_of_particles, double** d_neighbor_bests_locations, double* d_neighbor_bests_values)
{
	int particle_nr = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = 0; i < dim; i++)
	{
		d_particles[particle_nr][i] = 0.0;
		d_velocities[particle_nr][i] = 1.0;
		d_personal_bests_locations[particle_nr][i] = 0.0;
		d_personal_bests_values[particle_nr] = INT_MAX;
		d_neighbor_bests_locations[particle_nr][i] = 0.0;
		d_neighbor_bests_values[particle_nr] = INT_MAX;
	}
}

__global__ void compute_function(double** d_particles, double** d_velocities, double** d_personal_bests_locations, double* d_personal_bests_values, int dim, int nr_of_particles, double** d_neighbor_bests_locations, double* d_neighbor_bests_values)
{
	int particle_nr = blockIdx.x * blockDim.x + threadIdx.x;
	double value = 0.0;
	for (int i = 0; i < dim; i++)
	{
		value += d_particles[particle_nr][i] * d_particles[particle_nr][i];
	}

	if (value < d_personal_bests_values[particle_nr])
	{
		d_personal_bests_values[particle_nr] = value;
		d_personal_bests_locations[particle_nr] = d_particles[particle_nr];
	}
}

__global__ void actualize_bests(double** d_particles, double** d_velocities, double** d_personal_bests_locations, double* d_personal_bests_values, int dim, int nr_of_particles, double** d_neighbor_bests_locations, double* d_neighbor_bests_values)
{
	//s¹siadami s¹ cz¹stki z tego samego bloku
	int particle_nr = blockIdx.x * blockDim.x + threadIdx.x;
	int first_particle_in_block = blockIdx.x * blockDim.x;

	double* new_best_neighbor_location = new double[dim], new_best_neighbor_value = INT_MAX;
	for (int i = first_particle_in_block; i < first_particle_in_block + blockDim.x; i++)
	{
		if (d_neighbor_bests_values[particle_nr] < d_personal_bests_values[i])
		{
			new_best_neighbor_location = d_personal_bests_locations[i];
			new_best_neighbor_value - d_personal_bests_values[i];
		}
	}

	if (new_best_neighbor_value != INT_MAX)
	{
		d_neighbor_bests_values[particle_nr] = new_best_neighbor_value;
		for (int i = 0; i < dim; i++)
		{
			d_neighbor_bests_locations[particle_nr][i] = new_best_neighbor_location[i];
		}
	}
}

__global__ void compute_velocities(double** d_particles, double** d_velocities, double** d_personal_bests_locations, double* d_personal_bests_values, int dim, int nr_of_particles, double** d_neighbor_bests_locations, double* d_neighbor_bests_values)
{
	int particle_nr = blockIdx.x * blockDim.x + threadIdx.x;
	//random
	double r1 = 0.5;
	double r2 = 0.5;

	for (int i = 0; i < dim; i++)
	{
		d_velocities[particle_nr][i] += r1 * (d_personal_bests_locations[particle_nr][i] - d_particles[particle_nr][i]) +
			r1 * (d_neighbor_bests_locations[particle_nr][i] - d_particles[particle_nr][i]);
	}
}

__global__ void actualize_locations(double** d_particles, double** d_velocities, double** d_personal_bests_locations, double* d_personal_bests_values, int dim, int nr_of_particles, double** d_neighbor_bests_locations, double* d_neighbor_bests_values)
{
	int particle_nr = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = 0; i < dim; i++)
	{
		d_particles[particle_nr][i] += d_velocities[particle_nr][i];
	}
}

//particle = location
int main()
{
	int dim, iterations, nr_of_particles, particle_size;

	dim = 1;
	nr_of_particles = 100;
	iterations = 1000;
	particle_size = sizeof(double);

	dim3 threads_per_block(16);
	dim3 num_of_blocks(nr_of_particles / (threads_per_block.x * threads_per_block.y * threads_per_block.z));

	//CPU
	double **h_particles, **h_personal_bests_locations, *h_personal_bests_values, **h_velocities, **h_neighbor_bests_locations, *h_neighbor_bests_values;

	//GPU
	double **d_particles, **d_personal_bests_locations, *d_personal_bests_values, **d_velocities, **d_neighbor_bests_locations, *d_neighbor_bests_values;

	cudaError_t cudaStatus;

	//alokuj pamiêæ na GPU	
	cudaStatus = cudaMalloc(&d_particles, nr_of_particles * particle_size * dim);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&d_personal_bests_locations, nr_of_particles*particle_size * dim);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&d_personal_bests_values, nr_of_particles * particle_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&d_velocities, nr_of_particles * particle_size * dim);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&d_neighbor_bests_locations, nr_of_particles * particle_size * dim);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&d_neighbor_bests_values, nr_of_particles * particle_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	//inicjalizuj po³o¿enia, prêdkoœci oraz bests
	initialize_particles << <num_of_blocks, threads_per_block >> >(d_particles, d_velocities, d_personal_bests_locations, d_personal_bests_values, dim, nr_of_particles, d_neighbor_bests_locations, d_neighbor_bests_values);

	for (int i = 1; i <= iterations; i++)
	{
		compute_function << <num_of_blocks, threads_per_block >> >(d_particles, d_velocities, d_personal_bests_locations, d_personal_bests_values, dim, nr_of_particles, d_neighbor_bests_locations, d_neighbor_bests_values);
		actualize_bests << <num_of_blocks, threads_per_block >> >(d_particles, d_velocities, d_personal_bests_locations, d_personal_bests_values, dim, nr_of_particles, d_neighbor_bests_locations, d_neighbor_bests_values);
		compute_velocities << <num_of_blocks, threads_per_block >> >(d_particles, d_velocities, d_personal_bests_locations, d_personal_bests_values, dim, nr_of_particles, d_neighbor_bests_locations, d_neighbor_bests_values);
		actualize_locations << <num_of_blocks, threads_per_block >> >(d_particles, d_velocities, d_personal_bests_locations, d_personal_bests_values, dim, nr_of_particles, d_neighbor_bests_locations, d_neighbor_bests_values);
	}

	//alokuj pamiêæ na CPU
	h_particles = new double*[nr_of_particles];
	for (int i = 0; i < nr_of_particles; i++)
	{
		h_particles[i] = new double[dim];
	}

	h_personal_bests_locations = new double*[nr_of_particles];
	for (int i = 0; i < nr_of_particles; i++)
	{
		h_personal_bests_locations[i] = new double[dim];
	}

	h_personal_bests_values = new double[nr_of_particles];

	h_velocities = new double*[nr_of_particles];
	for (int i = 0; i < nr_of_particles; i++)
	{
		h_velocities[i] = new double[dim];
	}

	h_neighbor_bests_locations = new double*[nr_of_particles];
	for (int i = 0; i < nr_of_particles; i++)
	{
		h_neighbor_bests_locations[i] = new double[dim];
	}

	h_neighbor_bests_values = new double[nr_of_particles];

	//kopiuj wyniki na GPU
	cudaStatus = cudaMemcpy(h_particles, d_particles, nr_of_particles * particle_size * dim, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(h_personal_bests_locations, d_personal_bests_locations, nr_of_particles * particle_size * dim, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(h_personal_bests_values, d_personal_bests_values, nr_of_particles * particle_size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(h_neighbor_bests_locations, d_neighbor_bests_locations, nr_of_particles * particle_size * dim, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(h_neighbor_bests_values, d_neighbor_bests_values, nr_of_particles * particle_size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		goto Error;
	}

Error:
	cudaFree(d_particles);
	cudaFree(d_neighbor_bests_locations);
	cudaFree(d_personal_bests_locations);
	cudaFree(d_velocities);
	cudaFree(d_neighbor_bests_values);
	cudaFree(d_personal_bests_values);

	return cudaStatus;
}