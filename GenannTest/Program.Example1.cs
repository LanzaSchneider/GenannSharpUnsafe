using System;

public partial class Program
{
    static int Example1()
    {
        Console.Out.Write("GENANN example 1.\n");
        Console.Out.Write("Train a small ANN to the XOR function using backpropagation.\n");

        /* This will make the neural network initialize differently each run. */
        /* If you don't get a good result, try again for a different result. */
        var random = new Random();

        /* Input and expected out data for the XOR function. */
        double[][] input = {new double[] {0, 0}, new double[] {0, 1}, new double[] {1, 0}, new double[] {1, 1}};
        double[] output = {0, 1, 1, 0};
        int i;
        
        /* New network with 2 inputs,
         * 1 hidden layer of 2 neurons,
         * and 1 output. */
        var ann = new Genann(2, 1, 2, 1, random);
        
        /* Train on the four labeled data points many times. */
        var iteration = 1000;
        for (i = 1; i < iteration; ++i)
        {
            var learningRate = 3.0;
            ann.Train(input[0], output, 0, learningRate);
            ann.Train(input[1], output, 1, learningRate);
            ann.Train(input[2], output, 2, learningRate);
            ann.Train(input[3], output, 3, learningRate);
        }

        /* Run the network and see what it predicts. */
        unsafe
        {
            var outputArray = new double[1];
            ann.Run(input[0], outputArray);
            Console.Out.Write("Output for [{0:N2}, {1:N2}] is {2:N2}.\n", input[0][0], input[0][1], outputArray[0]);
            ann.Run(input[1], outputArray);
            Console.Out.Write("Output for [{0:N2}, {1:N2}] is {2:N2}.\n", input[1][0], input[1][1], outputArray[0]);
            ann.Run(input[2], outputArray);
            Console.Out.Write("Output for [{0:N2}, {1:N2}] is {2:N2}.\n", input[2][0], input[2][1], outputArray[0]);
            ann.Run(input[3], outputArray);
            Console.Out.Write("Output for [{0:N2}, {1:N2}] is {2:N2}.\n", input[3][0], input[3][1], outputArray[0]);
        }

        return 0;
    }
}