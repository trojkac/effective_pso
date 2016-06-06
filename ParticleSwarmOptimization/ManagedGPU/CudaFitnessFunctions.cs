namespace ManagedGPU
{
    public class CudaFitnessFunctions
    {
        public static ICudaFitnessFunction Quadratic = new QuadraticFitnessFunction();

        public static ICudaFitnessFunction Rosenbrock = new RosenbrockFitnessFunction();

        public static ICudaFitnessFunction SimpleRastrigin = new SimpleRastriginFitnessFunction();

        public static ICudaFitnessFunction Sphere = new SphereFitnessFunction();

        public static ICudaFitnessFunction Ellipsoidal = new EllipsoidalFitnessFunction();

        public static ICudaFitnessFunction Rastrigin = new RastriginFitnessFunction();

        public static ICudaFitnessFunction BucheRastrigin = new BucheRastriginFitnessFunction();

        public static ICudaFitnessFunction LinearSlope = new LinearSlopeFitnessFunction();
    }
}
