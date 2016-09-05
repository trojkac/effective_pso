namespace Common
{
    public class ParticlesCount
    {
        public PsoParticleType ParticleType;
        public int Count;

        public ParticlesCount(PsoParticleType psoParticleType, int p)
        {
            // TODO: Complete member initialization
            ParticleType = psoParticleType;
            Count = p;
        }

        public ParticlesCount()
        {
            Count = 0;
            ParticleType = PsoParticleType.Standard;
        }
    }
}