using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace DieHardML
{
    class Program // https://blog.elmah.io/predicting-die-hard-fans-with-ml-net-and-csharp/
    {
        private const string __PipelineFile = "./diehard-pipeline.zip";
        private const string __ModelFile = "./diehard-model.zip";

        static void Main(string[] args)
        {
            var context = new MLContext();

            //var training_data = new List<MovePreferenceInput>();

            var fanat = new MovePreference
            {
                StarWars = 8,
                Armageddon = 10,
                SleeplessInSeattle = 1,
                ILikeDieHard = true
            };

            var hater = new MovePreference
            {
                StarWars = 1,
                Armageddon = 1,
                SleeplessInSeattle = 9,
                ILikeDieHard = false
            };

            var training_data = Enumerable.Repeat(fanat, 50)
               .Concat(Enumerable.Repeat(hater, 50));

            var training_data_view = context.Data.LoadFromEnumerable(training_data);

            var model = !File.Exists(__ModelFile) && !File.Exists(__PipelineFile)
                ? TrainNewModel(context, training_data_view)
                : RetrainModel(context, training_data_view);

            var test1 = new MovePreference
            {
                StarWars = 7,
                Armageddon = 9,
                SleeplessInSeattle = 0
            };

            var test2 = new MovePreference
            {
                StarWars = 0,
                Armageddon = 0,
                SleeplessInSeattle = 10
            };

            var engine = context.Model.CreatePredictionEngine<MovePreference, LikePrediction>(model);

            var prediction1 = engine.Predict(test1);
            Console.WriteLine("For data {0}\r\n\tprediction {1}", test1, prediction1);

            var prediction2 = engine.Predict(test2);
            Console.WriteLine("For data {0}\r\n\tprediction {1}", test2, prediction2);
        }

        private static ITransformer TrainNewModel(MLContext Context, IDataView Data)
        {
            var base_pipeline = Context
                .Transforms
                .Concatenate(
                    "Features",
                    nameof(MovePreference.StarWars),
                    nameof(MovePreference.Armageddon),
                    nameof(MovePreference.SleeplessInSeattle))
                .AppendCacheCheckpoint(Context);

            var data_pipeline = base_pipeline.Fit(Data);

            Context.Model.Save(data_pipeline, Data.Schema, __PipelineFile);

            var trainer = base_pipeline.Append(Context
                .BinaryClassification
                .Trainers
                .AveragedPerceptron(
                    labelColumnName: nameof(MovePreference.ILikeDieHard),
                    numberOfIterations: 10,
                    featureColumnName: "Features"));

            var preprocessed_data = data_pipeline.Transform(Data);
            var model = trainer.Fit(preprocessed_data);
            Context.Model.Save(model, Data.Schema, __ModelFile);
            return model;
        }

        private static ITransformer RetrainModel(MLContext Context, IDataView Data)
        {
            var trained_model = Context.Model.Load(__ModelFile, out _);
            var data_pipeline = Context.Model.Load(__PipelineFile, out _);

            var transformed_data = data_pipeline.Transform(Data);
            var chain = trained_model as IEnumerable<ITransformer>;
            var prediction_transformer = chain!.Last() as ISingleFeaturePredictionTransformer<object>;

            var original_model_parameters = prediction_transformer!.Model as LinearBinaryModelParameters;

            var transformer = Context
               .BinaryClassification
               .Trainers
               .AveragedPerceptron(
                    nameof(MovePreference.ILikeDieHard),
                    "Features",
                    numberOfIterations: 10)
               .Fit(transformed_data, original_model_parameters);
            var model = data_pipeline.Append(transformer);

            Context.Model.Save(model, Data.Schema, __ModelFile);

            return model;
        }
    }

    record MovePreference
    {
        public float StarWars { get; init; }
        public float Armageddon { get; init; }
        public float SleeplessInSeattle { get; init; }

        public bool ILikeDieHard { get; init; }
    }

    record LikePrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; init; }
    }
}
