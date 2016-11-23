using neural.Class;
using neural.Helper;
using System;

namespace neural
{
    class Program
    {
        static Perceptron.myDelegate activationFunction = new Perceptron.myDelegate(Perceptron.sigmodFunction);
        // H2H results (HW, D, AW), Last 10 host matches (W, D, L), Last 10 guest matches (W,D,L) = 9 numbers as input data
        static NeuralNetwork nn = new NeuralNetwork(9, 3, activationFunction);

        static void Main(string[] args)
        {
            int i = 0;
            while (i < 10)
            {
                nn = new NeuralNetwork(9, 3, activationFunction);
                // preparing neural network
                //dla 9 elementowej sieci
                nn.AppendLayer(9);
                nn.AppendLayer(7);
                nn.AppendLayer(5);

                // SOU - EVE
                double[] trainingInput = new double[] { 2, 2, 1, 3, 4, 3, 4, 3, 3 };
                double[] trainingOutput = new double[] { .48, .27, .25 };
                Training(trainingInput, trainingOutput);

                // HUL - WBA
                trainingInput = new double[] { 1, 2, 1, 1, 1, 9, 3, 3, 4 };
                trainingOutput = new double[] { .31, .29, .42 };
                Training(trainingInput, trainingOutput);

                //BUR -MCI
                trainingInput = new double[] { 1, 0, 4, 3, 2, 5, 6, 3, 1 };
                trainingOutput = new double[] { .1, 0.15, 0.75 };
                Training(trainingInput, trainingOutput);

                ////LEI - MID
                //trainingInput = new double[] { 3, 2, 0, 3, 2, 5, 1, 4, 5 };
                //trainingOutput = new double[] { .52, 0.26, 0.22 };
                //Training(trainingInput, trainingOutput);


                ////BUR-MCI
                //double[] inputData = new double[] { 1, 0, 4, 3, 2, 5, 6, 3, 1 };

                //LEI - MID
                double[] inputData = new double[] { 3, 2, 0, 3, 2, 5, 1, 4, 5 };

                Predict(inputData);
                i++;
            }
            Console.ReadKey();
        }

        /// <summary>
        /// Nauka sieci neuronowej
        /// </summary>
        /// Ostatnie 5 meczów H2H na danym terenie (DW, R, WW), ostatnie 10 meczów gospodarza (W,R,P), ostatnie 10 meczów gościa (W,R,P)
        /// <param name="trainingInput">Dane wejściowe</param>
        /// <param name="trainingOutput"></param>
        static void Training(double[] trainingInput, double[] trainingOutput)
        {
            for (int i = 0; i < 1000; i++)
            {
                nn.PropagateBack(trainingInput, trainingOutput, activationFunction);
            }
        }

        /// <summary>
        /// Przewidywanie szans osiągnięcia danego rezultatu w danym meczu
        /// </summary>
        static void Predict(double[] inputData)
        {
            try
            {
                Console.WriteLine("Result:");
                MatrixHelper.MatrixDisplay(nn.Propagate(inputData));
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
        }
    }
}
