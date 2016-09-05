namespace Common
{
    public struct DimensionBound
    {
        public double Min;
        public double Max;

        public DimensionBound(double min, double max)
        {
            Min = min;
            Max = max;
        }
    }
}