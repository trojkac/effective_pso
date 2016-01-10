namespace Common
{
    public class UserPsoParameters
    {
        public bool IterationsLimitCondition { get; set; }
        public int Iterations { get; set; }
        public double TargetValue { get; set; }
        public bool TargetValueCondition { get; set; }
        public double Epsilon { get; set; }
        public int StandardParticles { get; set; }
        public int FullyInformedParticles { get; set; }
    }
}
