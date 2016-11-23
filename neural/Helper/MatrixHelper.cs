using System;

namespace neural.Helper
{
    static class MatrixHelper
    {

        /// <summary>
        /// Mnożenie macierzy
        /// </summary>
        /// <param name="firstMatrix">Macierz</param>
        /// <param name="secondMatrix">Macierz</param>
        /// <returns>Macierz</returns>
        public static double[,] MatrixMultiply(double[,] firstMatrix, double[,] secondMatrix)
        {
            int firstColumns = firstMatrix.GetLength(1);
            int firstRows = firstMatrix.GetLength(0);
            int secondColumns = secondMatrix.GetLength(1);
            int secondRows = secondMatrix.GetLength(0);
            if (firstColumns != secondRows)
            {
                if (firstColumns < secondRows)
                {
                    firstMatrix = MatrixComplementColumns(firstMatrix, secondRows);
                    firstColumns = secondRows;
                }
                else
                {
                    secondMatrix = MatrixComplementRows(secondMatrix, firstColumns);
                    secondRows = firstColumns;
                }
            }
            double[,] resultMatrix = new double[firstRows, secondColumns];
            for (int i = 0; i < firstRows; i++)
            {
                for (int j = 0; j < secondColumns; j++)
                {
                    for (int k = 0; k < firstColumns; k++)
                    {
                        resultMatrix[i, j] += firstMatrix[i, k] * secondMatrix[k, j];
                    }
                }
            }
            return resultMatrix;
        }

        /// <summary>
        /// Dopełnienie macierzy o wiersze
        /// </summary>
        /// <param name="matrix">Macierz do dopełnienia</param>
        /// <param name="totalRows">Nowa liczba wierszy</param>
        /// <returns>Dopełniona macierz</returns>
        private static double[,] MatrixComplementRows(double[,] matrix, int totalRows)
        {
            double[,] complementedMatrix = new double[totalRows, matrix.GetLength(1)];
            for (int i = 0; i < matrix.GetLength(0); i++)
                for (int j = 0; j < matrix.GetLength(1); j++)
                    complementedMatrix[i, j] = matrix[i, j];
            return complementedMatrix;
        }

        /// <summary>
        /// Dopełnienie macierzy o kolumny
        /// </summary>
        /// <param name="matrix">Macierz do dopełnienia</param>
        /// <param name="totalColumns">Nowa liczba kolumn</param>
        /// <returns>Dopełniona macierz</returns>
        private static double[,] MatrixComplementColumns(double[,] matrix, int totalColumns)
        {
            double[,] complementedMatrix = new double[matrix.GetLength(0), totalColumns];
            for (int i = 0; i < matrix.GetLength(0); i++)
                for (int j = 0; j < matrix.GetLength(1); j++)
                    complementedMatrix[i, j] = matrix[i, j];
            return complementedMatrix;
        }

        /// <summary>
        /// Dopełnienie macierzy dla odejmowania
        /// </summary>
        /// <param name="firstMatrix">Macierz</param>
        /// <param name="secondMatrix">Macierz</param>
        private static void MatrixComplement(ref double[,] firstMatrix, ref double[,] secondMatrix)
        {
            int firstColumns = firstMatrix.GetLength(1);
            int firstRows = firstMatrix.GetLength(0);
            int secondColumns = secondMatrix.GetLength(1);
            int secondRows = secondMatrix.GetLength(0);

            if (firstColumns != secondColumns || firstRows != secondRows)
            {
                if (firstColumns < secondColumns)
                {
                    firstMatrix = MatrixComplementColumns(firstMatrix, secondColumns);
                    firstColumns = secondColumns;
                }
                else
                {
                    secondMatrix = MatrixComplementColumns(secondMatrix, firstColumns);
                    secondColumns = firstColumns;
                }

                if (firstRows < secondRows)
                {
                    firstMatrix = MatrixComplementRows(firstMatrix, secondRows);
                    firstRows = secondRows;
                }
                else
                {
                    secondMatrix = MatrixComplementRows(secondMatrix, firstRows);
                    secondRows = firstRows;
                }
            }
        }

        /// <summary>
        /// Odejmowanie macierzy
        /// </summary>
        /// <param name="firstMatrix">Macierz</param>
        /// <param name="secondMatrix">Macierz</param>
        /// <returns>Macierz</returns>
        public static double[,] MatrixSubstraction(double[,] firstMatrix, double[,] secondMatrix)
        {
            MatrixComplement(ref firstMatrix, ref secondMatrix);
            double[,] resultMatrix = new double[firstMatrix.GetLength(0), firstMatrix.GetLength(1)];
            for (int i = 0; i < firstMatrix.GetLength(0); i++)
            {
                for (int j = 0; j < firstMatrix.GetLength(1); j++)
                {
                    resultMatrix[i, j] = firstMatrix[i, j] - secondMatrix[i, j];
                }
            }
            return resultMatrix;
        }

        /// <summary>
        /// Transpozcyja macierzy
        /// </summary>
        /// <param name="matrix">Macierz</param>
        /// <returns>Macierz</returns>
        public static double[,] MatrixTransposition(double[,] matrix)
        {
            int matrixRow = matrix.GetLength(0);
            int matrixColumn = matrix.GetLength(1);
            double[,] newMatrix = new double[matrixColumn, matrixRow];
            for (int i = 0; i < matrixColumn; i++)
            {
                for (int j = 0; j < matrixRow; j++)
                {
                    newMatrix[i, j] = matrix[j, i];
                }
            }
            return newMatrix;
        }

        /// <summary>
        /// Wyświetlanie macierzy
        /// </summary>
        /// <param name="matrix">Macierz</param>
        public static void MatrixDisplay(double[,] matrix)
        {
            for (int first = 0; first < matrix.GetLength(0); first++)
            {
                for (int second = 0; second < matrix.GetLength(1); second++)
                {
                    Console.Write("{0} ", Math.Round(matrix[first, second], 2));
                }
                Console.WriteLine();
            }
        }

        /// <summary>
        /// Wektor na macierz
        /// </summary>
        /// <param name="input">Wektor</param>
        /// <returns>Macierz</returns>
        public static double[,] ConvertToMatrix(double[] input)
        {
            double[,] matrix = new double[1, input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                matrix[0, i] = input[i];
            }
            return matrix;
        }

        /// <summary>
        /// Mnożenie skalarne macierzy
        /// </summary>
        /// <param name="firstMatrix">Macierz</param>
        /// <param name="secondMatrix">Macierz</param>
        /// <returns>Macierz</returns>
        public static double[,] MatrixScalar(double[,] firstMatrix, double[,] secondMatrix)
        {
            int firstColumns = firstMatrix.GetLength(1);
            int firstRows = firstMatrix.GetLength(0);
            int secondColumns = secondMatrix.GetLength(1);
            int secondRows = secondMatrix.GetLength(0);

            if (firstRows == secondColumns)
            {
                var resultMatrix = new double[secondRows, firstColumns];
                for (int i = 0; i < secondRows; i++)
                {
                    if (secondColumns == 1)
                    {
                        for (int j = 0; j < firstColumns; j++)
                        {
                            resultMatrix[i, j] = firstMatrix[0, j] * secondMatrix[i, 0];
                        }
                    }
                    else
                    {
                        for (int j = 0; j < firstColumns; j++)
                        {
                            resultMatrix[i, j] = firstMatrix[0, j] * secondMatrix[i, j];
                        }
                    }
                }
                return resultMatrix;
            }
            else
            {
                throw new Exception("Liczba wierszy pierwszej macierzy różni się od liczby kolumn drugiej");
            }
        }

        /// <summary>
        /// Mnożenie skalarne wektorów/macierzy o jednym wierszu
        /// </summary>
        /// <param name="firstMatrix">Macierz</param>
        /// <param name="secondMatrix">Macierz</param>
        /// <returns>Macierz</returns>
        public static double[,] VectorScalar(double[,] firstMatrix, double[,] secondMatrix)
        {
            int firstColumns = firstMatrix.GetLength(1);
            int secondColumns = secondMatrix.GetLength(1);
            if (firstColumns == secondColumns)
            {
                var result = new double[1, secondColumns];
                for (int i = 0; i < firstColumns; i++)
                {
                    result[0, i] = firstMatrix[0, i] * secondMatrix[0, i];
                }

                return result;
            }
            else
            {
                throw new Exception("Wektory mają różną długość");
            }

        }

    }
}

