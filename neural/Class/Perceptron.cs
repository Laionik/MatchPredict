using System;

namespace neural.Class
{

    class Perceptron
    {
        public double[,] scalesMatrix;
        public delegate double myDelegate(double data, bool deriv=false);
        public myDelegate activationFunction;
        public Perceptron(int sizeX, int sizeY, myDelegate activationFunction)
        {
            scalesMatrix = new double[sizeX, sizeY];
            this.activationFunction = activationFunction;
        }

        public Perceptron(myDelegate activationFunction)
        {
            this.activationFunction = activationFunction;
        }

        /// <summary>
        /// Funkcja aktywacyjna - binarna
        /// </summary>
        /// <param name="data">Element wektora/macierzy</param>
        /// <returns>Wynik funkcji aktywacji - double</returns>
        public static double binaryFunction(double data)
        {
            return data < 0 ? 0 : 1;
        }

        /// <summary>
        /// Funkcja aktywacyjna - sigmoidalna
        /// </summary>
        /// <param name="data">Element wektora/macierzy</param>
        /// <returns>Wynik funkcji aktywacji - double</returns>
        public static double sigmodFunction(double data, bool deriv=false)
        {
            if (deriv)
                return data * (1 - data);
            return 1 / (1 + Math.Pow(Math.E, data));
        }

        /// <summary>
        /// Generowanie losowych wartości wag
        /// </summary>
        public void randomScalesGenerate()
        {
            Random rnd = new Random();
            for (int first = 0; first < scalesMatrix.GetLength(0); first++)
                for (int second = 0; second < scalesMatrix.GetLength(1); second++)
                    scalesMatrix[first, second] = rnd.NextDouble() * 10 - 5;
        }
    }
}
