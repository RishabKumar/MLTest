using Microsoft.ML.Data;
using System;

namespace MLTest.Model.TextModel
{
    public class ArticlePrediction
    {
        [ColumnName("Classification")]
        public string Classification { get; set; }

        [ColumnName("Score")]
        [VectorType]
        public float[] Score { get; set; }

    }
}
