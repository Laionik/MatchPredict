using System;

namespace neural.Class
{
    class layer
    {
        public Perceptron perceptron;
        public int size;
        public double[,] values;
        public layer(int previousLayerSize, int numberOfNeurons, Perceptron.myDelegate activationFunction)
        {
            this.size = numberOfNeurons;
            values = new double[1, numberOfNeurons];
            perceptron = new Perceptron(previousLayerSize, numberOfNeurons,  activationFunction);
            perceptron.randomScalesGenerate();
        }
    }
}
