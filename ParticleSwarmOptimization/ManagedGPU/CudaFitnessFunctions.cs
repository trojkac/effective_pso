namespace ManagedGPU
{
    public class CudaFitnessFunctions
    {
        public static ICudaFitnessFunction Quadratic = new QuadraticFitnessFunction();

        public static ICudaFitnessFunction Rosenbrock = new RosenbrockFitnessFunction();

        public static ICudaFitnessFunction Rastrigin = new RastriginFitnessFunction();
    }
}
