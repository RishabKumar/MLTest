using Microsoft.ML.Data;

namespace MLTest.Model
{
    public class SalaryModel
    {
        [LoadColumn(0)]
        public float YearsOfExperience;

        [LoadColumn(1)]
        public float Salary;
    }
}