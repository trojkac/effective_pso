using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;

namespace PSOCudafy
{
    public class PsoAlgorithm
    {
        public static void Execute()
        {
            CudafyModule km = CudafyModule.TryDeserialize();
            if (km == null || !km.TryVerifyChecksums())
            {
                km = CudafyTranslator.Cudafy();
                km.Serialize();
            }

            GPGPU gpu = CudafyHost.GetDevice(CudafyModes.Target);
            gpu.LoadModule(km);

            var dim = 1;
            var particlesCount = 100;
            var iterations = 1000;
            
            var threadsPerBlock = 16;
            var blocksCount = (particlesCount / threadsPerBlock);

            var h_particles = new double[particlesCount, dim];
            var d_particles = gpu.CopyToDevice(h_particles);
            
            var h_velocities = new double[particlesCount, dim];
            var d_velocities = gpu.CopyToDevice(h_velocities);

            var h_personal_bests_locations = new double[particlesCount, dim];
            var d_personal_bests_locations = gpu.CopyToDevice(h_personal_bests_locations);

            var h_personal_bests_values = new double[particlesCount];
            var d_personal_bests_values = gpu.Allocate<double>(particlesCount);
            
            var h_neighbor_bests_locations = new double[particlesCount, dim];
            var d_neighbor_bests_locations = gpu.CopyToDevice(h_neighbor_bests_locations);
            
            var h_neighbor_bests_values = new double[particlesCount];
            var d_neighbor_bests_values = gpu.Allocate<double>(particlesCount);

            gpu.Launch(blocksCount, threadsPerBlock, "InitializeParticles",
                d_particles, d_velocities, d_personal_bests_locations, 
                d_personal_bests_values, d_neighbor_bests_locations, 
                d_neighbor_bests_values, dim, particlesCount);

            var new_best_neighbor_location = gpu.Allocate<double>(particlesCount);

            for (var i = 0; i < iterations; ++i)
            {
                gpu.Launch(blocksCount, threadsPerBlock, "ComputeFunction",
                    d_particles, d_velocities, d_personal_bests_locations, 
                    d_personal_bests_values, d_neighbor_bests_locations, 
                    d_neighbor_bests_values, dim, particlesCount);

                gpu.Launch(blocksCount, threadsPerBlock, "UpdateBests",
                    d_particles, d_velocities, d_personal_bests_locations, 
                    d_personal_bests_values, d_neighbor_bests_locations, 
                    d_neighbor_bests_values, new_best_neighbor_location, dim, particlesCount);

                gpu.Launch(blocksCount, threadsPerBlock, "ComputeVelocities",
                    d_particles, d_velocities, d_personal_bests_locations, 
                    d_personal_bests_values, d_neighbor_bests_locations, 
                    d_neighbor_bests_values, dim, particlesCount);

                gpu.Launch(blocksCount, threadsPerBlock, "UpdateLocations",
                    d_particles, d_velocities, d_personal_bests_locations, 
                    d_personal_bests_values, d_neighbor_bests_locations, 
                    d_neighbor_bests_values, dim, particlesCount);
            }

            gpu.CopyFromDevice(d_particles, h_particles);
            gpu.CopyFromDevice(d_personal_bests_locations, h_personal_bests_locations);
            gpu.CopyFromDevice(d_personal_bests_values, h_neighbor_bests_values);
            gpu.CopyFromDevice(d_neighbor_bests_locations, h_neighbor_bests_locations);
            gpu.CopyFromDevice(d_neighbor_bests_values, h_neighbor_bests_values);

            gpu.FreeAll();
        }

        [Cudafy]
        public static void InitializeParticles(GThread thread,
                                               double[,] d_particles,
                                               double[,] d_velocities,
                                               double[,] d_personal_best_locations,
                                               double[] d_personal_bests_values,
                                               double[,] d_neighbor_bests_locations,
                                               double[] d_neighbor_bests_values,
                                               int dim, int particlesCount)
        {
            int idx = thread.blockIdx.x * thread.blockDim.x + thread.threadIdx.x;

            for (int i = 0; i < dim; i++)
            {
                d_particles[idx, i] = 0.0;
                d_velocities[idx, i] = 1.0;
                d_personal_best_locations[idx, i] = 0.0;
                d_personal_bests_values[idx] = int.MaxValue;
                d_neighbor_bests_locations[idx, i] = 0.0;
                d_neighbor_bests_values[idx] = int.MaxValue;
            }
        }

        [Cudafy]
        public static void ComputeFunction(GThread thread,
                                           double[,] d_particles,
                                           double[,] d_velocities,
                                           double[,] d_personal_best_locations,
                                           double[] d_personal_bests_values,
                                           double[,] d_neighbor_bests_locations,
                                           double[] d_neighbor_bests_values,
                                           int dim, int particlesCount)
        {
            int particleIdx = thread.blockIdx.x * thread.blockDim.x + thread.threadIdx.x;
            double value = 0.0;

            for (int i = 0; i < dim; i++)
            {
                value += d_particles[particleIdx, i] * d_particles[particleIdx, i];
            }

            if (value < d_personal_bests_values[particleIdx])
            {
                d_personal_bests_values[particleIdx] = value;

                for (int i = 0; i < dim; i++)
                {
                    d_personal_best_locations[particleIdx, i] = d_particles[particleIdx, i];
                }
            }
        }

        [Cudafy]
        public static void UpdateBests(GThread thread,
                                       double[,] d_particles,
                                       double[,] d_velocities,
                                       double[,] d_personal_best_locations,
                                       double[] d_personal_bests_values,
                                       double[,] d_neighbor_bests_locations,
                                       double[] d_neighbor_bests_values,
                                       double[] new_best_neighbor_location,
                                       int dim, int particlesCount)
        {
            int particleIdx = thread.blockIdx.x * thread.blockDim.x + thread.threadIdx.x;
            int firstInBlockIdx = thread.blockIdx.x * thread.blockDim.x;

            double new_best_neighbor_value = double.MaxValue;

            for (int i = firstInBlockIdx; i < firstInBlockIdx + thread.blockDim.x; ++i)
            {
                if (d_neighbor_bests_values[particleIdx] < d_personal_bests_values[i])
                {
                    for (int j = 0; j < dim; j++)
                    {
                        new_best_neighbor_location[j] = d_personal_best_locations[i, j];
                    }

                    new_best_neighbor_value = d_personal_bests_values[i];
                }
            }

            if (new_best_neighbor_value < double.MaxValue)
            {
                d_neighbor_bests_values[particleIdx] = new_best_neighbor_value;

                for (int i = 0; i < dim; ++i) 
                {
                    d_neighbor_bests_locations[particleIdx, i] = new_best_neighbor_location[i];
                }
            }
        }

        [Cudafy]
        public static void ComputeVelocities(GThread thread,
                                             double[,] d_particles,
                                             double[,] d_velocities,
                                             double[,] d_personal_best_locations,
                                             double[] d_personal_bests_values,
                                             double[,] d_neighbor_bests_locations,
                                             double[] d_neighbor_bests_values,
                                             int dim, int particlesCount)
        {
            int particleIdx = thread.blockIdx.x * thread.blockDim.x + thread.threadIdx.x;

            double r1 = 0.5;
            double r2 = 0.5;

            for (int i = 0; i < dim; i++)
            {
                d_velocities[particleIdx, i] +=
                    r1 * (d_personal_best_locations[particleIdx, i] - d_particles[particleIdx, i]) +
                    r2 * (d_neighbor_bests_locations[particleIdx, i] - d_particles[particleIdx, i]);
            }
        }

        [Cudafy]
        public static void UpdateLocations(GThread thread,
                                           double[,] d_particles,
                                           double[,] d_velocities,
                                           double[,] d_personal_best_locations,
                                           double[] d_personal_bests_values,
                                           double[,] d_neighbor_bests_locations,
                                           double[] d_neighbor_bests_values,
                                           int dim, int particlesCount)
        {
            int particleIdx = thread.blockIdx.x * thread.blockDim.x + thread.threadIdx.x;

            for (int i = 0; i < dim; i++)
            {
                d_particles[particleIdx, i] += d_velocities[particleIdx, i];
            }
        }
    }
}
