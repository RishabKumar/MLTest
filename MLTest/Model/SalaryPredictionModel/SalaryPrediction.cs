using Microsoft.ML.Data;

namespace MLTest.Model
{
    public class SalaryPrediction
    {
        [ColumnName("Score")]
        public float PredictedSalary;
    }
}