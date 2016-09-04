namespace ManagedGPU
{
    class CudaAlgorithmFactory
    {
        public static GenericCudaAlgorithm AlgorithmForFunction(CudaParams parameters, StateProxy proxy)
        {
            switch (parameters.FunctionNumber)
            {
                case 1:
                    return new SphereAlgorithm(parameters, proxy);

                case 2:
                    return new EllipsoidalAlgorithm(parameters, proxy);

                case 3:
                    return new RastriginAlgorithm(parameters, proxy);

                case 4:
                    return new BuecheRastriginAlgorithm(parameters, proxy);

                case 5:
                    return new LinearSlopeAlgorithm(parameters, proxy);

                case 6:
                    return new AttractiveSectorAlgorithm(parameters, proxy);

                case 7:
                    return new StepEllipsoidAlgorithm(parameters, proxy);

                case 8:
                    return new RosenbrockAlgorithm(parameters, proxy);

                case 9:
                    return new RosenbrockRotatedAlgorithm(parameters, proxy);

                case 10:
                    return new EllipsoidalRotatedAlgorithm(parameters, proxy);

                case 11:
                    return new DiscusAlgorithm(parameters, proxy);

                case 12: 
                    return new BentCigarAlgorithm(parameters, proxy);

                case 14:
                    return new DifferentPowersAlgorithm(parameters, proxy);

                case 15:
                    return new RastriginRotatedAlgorithm(parameters, proxy);

                case 16: 
                    return new WeierstrassAlgorithm(parameters, proxy);

                case 17:
                    return new SchaffersAlgorithm(parameters, proxy)
                    {
                        Conditioning = 10.0,
                        IllformedSeed = false
                    };

                case 18: 
                    return new SchaffersAlgorithm(parameters, proxy)
                    {
                        Conditioning = 1000.0,
                        IllformedSeed = true
                    };

                case 19:
                    return new GriewankRosenbrockAlgorithm(parameters, proxy);

                case 20:
                    return new SchwefelAlgorithm(parameters, proxy);

                case 21:
                    return new GallagherAlgorithm(parameters, proxy)
                    {
                        PeaksCount = 101
                    };

                case 22:
                    return new GallagherAlgorithm(parameters, proxy)
                    {
                        PeaksCount = 21
                    };

                case 23:
                    return new KatsuuraAlgorithm(parameters, proxy);

                case 24:
                    return new LunacekBiRastriginAlgorithm(parameters, proxy);

                default:
                    return null;
            }
        }
    }
}
