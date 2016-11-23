using neural.Helper;
using System;
using System.Linq;

namespace neural.Class
{
    class NeuralNetwork
    {
        public TrainingData trData;
        public layer[] layerList;
        private int inputSize;
        private int outputSize;
        private Perceptron.myDelegate activationFunction;

        public NeuralNetwork(int inputSize, int outputSize, Perceptron.myDelegate activationFunction)
        {
            this.inputSize = inputSize;
            this.outputSize = outputSize;
            this.activationFunction = activationFunction;
            LayerCreateInputOutput(inputSize, outputSize);
        }

        // TODO
        /// <summary>
        /// Rozszerzenie TrainingData
        /// </summary>
        /// <param name="data">Wartości startowe</param>
        public void Populate(double[] data)
        {
            // TODO
            //trData = new TrainingData(data, new double[] { });
        }

        /// <summary>
        /// Tworzenie warstwy wejściowej i wyjściowej
        /// </summary>
        /// <param name="inputSize">Liczba neuronów warstwy wejściowej</param>
        /// <param name="outputSize">Liczba neuronów warstwy wyjściowej</param>
        private void LayerCreateInputOutput(int inputSize, int outputSize)
        {
            layerList = new layer[2];
            layerList[0] = new layer(0, inputSize, activationFunction);
            layerList[1] = new layer(inputSize, outputSize, activationFunction);
        }

        /// <summary>
        /// Dodawanie nowej warstwy
        /// </summary>
        /// <param name="n">liczba elementów nowej warstwy</param>
        public void AppendLayer(int n)
        {
            int lastInputLayerSize = 0;
            var tempLayerList = layerList;
            layerList = new layer[tempLayerList.Length + 1];
            for (int i = 0; i < tempLayerList.Length - 1; i++)
            {
                layerList[i] = tempLayerList[i];
                lastInputLayerSize = tempLayerList[i].size;
            }
            layerList[tempLayerList.Length - 1] = new layer(lastInputLayerSize, n, activationFunction);
            layerList[tempLayerList.Length] = new layer(n, outputSize, activationFunction);
        }

        /// <summary>
        /// Metoda propagacji - sporządza predykcję wektora wejściowego
        /// </summary>
        /// <param name="td"></param>
        /// <returns></returns>
        public double[,] Propagate(params double[] data)
        {
            trData = new TrainingData(data, new double[] { });
            double[,] inputData = MatrixHelper.ConvertToMatrix(trData.input);
            layerList[0].values = inputData;
            for (int i = 1; i < layerList.Length; i++)
            {
                inputData = MatrixHelper.MatrixMultiply(inputData, layerList[i].perceptron.scalesMatrix);
                inputData = ExecuteActivationFunction(inputData, layerList[i].perceptron.activationFunction);
                layerList[i].values = inputData;
            }
            return inputData;
        }

        /// <summary>
        /// Użycie funkcji aktywacji na każdym elemencie macierzy
        /// </summary>
        /// <param name="input">macierz</param>
        /// <returns></returns>
        private double[,] ExecuteActivationFunction(double[,] input, Perceptron.myDelegate activationFunction, bool deriv = false)
        {
            for (int i = 0; i < input.GetLength(0); i++)
            {
                for (int j = 0; j < input.GetLength(1); j++)
                {
                    input[i, j] = activationFunction(input[i, j], deriv);
                }
            }
            return input;
        }

        /// <summary>
        /// Obliczanie błędu
        /// </summary>
        /// <param name="trData">Dane treningowe</param>
        /// <returns>Macierz</returns>
        private double[,] ErrorCalculate(TrainingData trData)
        {
            double[,] y = MatrixHelper.ConvertToMatrix(trData.output);
            var ysim = Propagate(trData.input);
            return MatrixHelper.MatrixSubstraction(y, ysim);
        }

        /// <summary>
        /// Propagacja wsteczna
        /// </summary>
        /// <param name="input">Dane wejściowe</param>
        /// <param name="output">Dane wyjściowe</param>
        public void PropagateBack(double[] input, double[] output, Perceptron.myDelegate activationFunction)
        {
            var error = ErrorCalculate(new TrainingData(input, output));
            int numberOfLayers = layerList.Length - 1;
            for (int i = 0; i < numberOfLayers; i++)
            {
                var valuesActivationFunction = ExecuteActivationFunction(layerList[numberOfLayers - i].values, activationFunction, true);
                var delta = MatrixHelper.VectorScalar(error, valuesActivationFunction); 

                var scalesMatrixTransposition = MatrixHelper.MatrixTransposition(layerList[numberOfLayers - i].perceptron.scalesMatrix);
                error = MatrixHelper.MatrixMultiply(delta, scalesMatrixTransposition);
               
                var matrixMultiplied = MatrixHelper.MatrixScalar(layerList[numberOfLayers - i - 1].values, MatrixHelper.MatrixTransposition(delta));
                layerList[numberOfLayers - i].perceptron.scalesMatrix = MatrixHelper.MatrixSubstraction(layerList[numberOfLayers - i].perceptron.scalesMatrix, MatrixHelper.MatrixTransposition(matrixMultiplied));
            }
        }
    }
}
