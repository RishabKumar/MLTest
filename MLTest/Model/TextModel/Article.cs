using Microsoft.ML.Data;
using System.Collections.Generic;

namespace MLTest.Model.TextModel
{
    public static class Collection
    {
        public static HashSet<string> Categories = new HashSet<string>();
    }

    public class Article
    {
        [LoadColumn(1)]
        public string Text { get; set; }

        [LoadColumn(0)]
        public string Category { get; set; }

        [LoadColumn(2)]
        public string Category1 { get; set; }
    }
}