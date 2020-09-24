using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;
using MLTest.Model;
using MLTest.Model.TextModel;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace MLTest
{
    class NewsData
    {
        public News[] Data { get; set; }
    }

    class News
    {
        public string Category { get; set; }
        public string Headline { get; set; }
    }

    class Program
    {
        static void Main()
        {
            CategoryPrediction.Execute();
            CategoryPrediction.Train();
            //using (var writer = new StreamWriter(File.OpenWrite("D://Test.json")))
            //{
            //    foreach (var line in File.ReadAllLines("D://News_Category_Dataset_v2.json"))
            //    {
            //        writer.WriteLine($"{line},");
            //    }
            //}
            //var json = JsonConvert.DeserializeObject<NewsData>(File.ReadAllText("D://Test.json"));
            //using (var writer = new StreamWriter(File.OpenWrite("D://NewsData.json")))
            //{
            //    foreach (var news in json.Data)
            //    {
            //        writer.WriteLine($"{news.Category}|{news.Headline}");
            //    }
            //}
            //var t = 0;

        }
    }

    class CategoryPrediction
    {
        private static MLContext mlContext = new MLContext(seed: 0);
        public static void Execute()
        {
            Train();
            var model = mlContext.Model.Load("D://trainedModel.data", out var schema);
            Console.WriteLine("Training done");

            // create prediction engine
            var predictionEngine = mlContext.Model.CreatePredictionEngine<Article, ArticlePrediction>(model);

            while (true)
            {
                var article = predictionEngine.Predict(new Article { Text = GetText() });
                Console.WriteLine("Predicted Category: " + article.Classification);
                var category = Collection.Categories.ElementAt(article.Score.ToList().IndexOf(article.Score.Max()));
                Console.WriteLine("Max probability:" + category);
            }
        }

        static string GetText()
        {
            return Console.ReadLine();
        }

        public static void Train(string trainingDataPath = "D:\\AssetData.csv", char separator = '|', bool hasHeader = false)
        {
            var pipelineCategory = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", inputColumnName: "Category") // Map the desired output column in dataset to a hypothetical column
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Text", outputColumnName: "TextFeaturized"))  // Map the input column(s) in dataset to a hypothetical column
                .Append(mlContext.Transforms.Concatenate("Features", "TextFeaturized"))  // Map all the hypothetical input columns to a hypothetical input column
                .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))  // Define what algorithm to be used on hypothetical output, input column
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("Classification", "PredictedLabel"));  // Map the output to the property in the Article Prediction Model

            var data = mlContext.Data.LoadFromTextFile<Article>(trainingDataPath, separator, hasHeader, trimWhitespace: true);

            var categories = data.GetColumn<string>("Category");
            foreach (var category in categories)
            {
                Collection.Categories.Add(category);
            }
            var model = pipelineCategory.Fit(data);
            mlContext.Model.Save(model, data.Schema, "D://trainedModel.data");
        }
    }

    class SalaryPredictor
    {
        public static void Execute()
        {
            var features = "Features";
            var mlContext = new MLContext(seed: 0);

            //create pipeline
            var pipeline = mlContext.Transforms.Concatenate(features, new[] { "YearsOfExperience" })
                .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Salary", featureColumnName: features, maximumNumberOfIterations: 100));
            //.Append(mlContext.Transforms.NormalizeMeanVariance("Features"));

            // load training data
            var data = mlContext.Data.LoadFromTextFile<SalaryModel>("D:\\SalaryData.csv", ',', true);

            // create training model
            var model = pipeline.Fit(data);

            var dataView = model.Transform(data);

            var metrices = mlContext.Regression.Evaluate(dataView, "Salary");

            Console.WriteLine(metrices.RSquared);
            Console.WriteLine(metrices.RootMeanSquaredError);

            // create prediction engine
            var predictionEngine = mlContext.Model.CreatePredictionEngine<SalaryModel, SalaryPrediction>(model);

            while (true)
            {
                var salary = predictionEngine.Predict(new SalaryModel() { YearsOfExperience = GetExperience() });
                Console.WriteLine("Predicted Salary: " + salary.PredictedSalary);
            }
        }

        static float GetExperience()
        {
            Console.WriteLine("Enter years of expereience:");
            return float.Parse(Console.ReadLine());
        }
    }


    class Sample
    {
        public class HouseData { public float Size { get; set; } public float Price { get; set; } }
        public class Prediction {[ColumnName("Score")] public float Price { get; set; } }
        static void Execute()
        {
            MLContext mlContext = new MLContext();
            // 1. Import or create training data
            HouseData[] houseData = {
                new HouseData() { Size = 1.1F, Price = 1.2F },
                new HouseData() { Size = 1.9F, Price = 2.3F },
                new HouseData() { Size = 2.8F, Price = 3.0F },
                new HouseData() { Size = 3.4F, Price = 3.7F } };
            IDataView trainingData = mlContext.Data.LoadFromEnumerable(houseData);

            // 2. Specify data preparation and model training pipeline           
            var pipeline = mlContext.Transforms.Concatenate("Features", new[] { "Size" })
                .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Price", maximumNumberOfIterations: 100));

            // 3. Train model           
            var model = pipeline.Fit(trainingData);

            // 4. Make a prediction          
            var size = new HouseData() { Size = 2.5F };
            var price = mlContext.Model.CreatePredictionEngine<HouseData, Prediction>(model).Predict(size);
            Console.WriteLine($"Predicted price for size: {size.Size * 1000} sq ft= {price.Price * 100:C}k");
            Console.Read();
            // Predicted price for size: 2500 sq ft= $261.98k       }   }
        }
    }
}