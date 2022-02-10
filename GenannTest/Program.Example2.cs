using System;

public partial class Program
{
    static int Example2()
    {
        Console.Out.Write("GENANN example 2.\n");
        Console.Out.Write("Train a small ANN to the XOR function using random search.\n");
        
        var random = new Random();

        /* Input and expected out data for the XOR function. */
        double[][] input = {new double[] {0, 0}, new double[] {0, 1}, new double[] {1, 0}, new double[] {1, 1}};
        double[] output = {0, 1, 1, 0};
        int i;
        
        /* New network with 2 inputs,
         * 1 hidden layer of 2 neurons,
         * and 1 output. */
        var ann = new Genann(2, 1, 2, 1, random);
        
        var outputArray = new double[1];
        
        double err;
        double last_err = 1000;
        int count = 0;

        do {
            ++count;
            if (count % 1000 == 0) {
                /* We're stuck, start over. */
                ann.Randomize(random);
                last_err = 1000;
            }

            var save = new Genann(2, 1, 2, 1);
            Array.Copy(ann.Weights, save.Weights, ann.Weights.Length);

            /* Take a random guess at the ANN weights. */
            for (i = 0; i < ann.TotalWeights; ++i) {
                ann.Weights[i] += random.NextDouble() - 0.5;
            }

            /* See how we did. */
            err = 0;
            ann.Run(input[0], outputArray);
            err += Math.Pow(outputArray[0] - output[0], 2.0);
            ann.Run(input[1], outputArray);
            err += Math.Pow(outputArray[0] - output[1], 2.0);
            ann.Run(input[2], outputArray);
            err += Math.Pow(outputArray[0] - output[2], 2.0);
            ann.Run(input[3], outputArray);
            err += Math.Pow(outputArray[0] - output[3], 2.0);

            /* Keep these weights if they're an improvement. */
            if (err < last_err)
            {
                save = null;
                last_err = err;
            } else {
                ann = save;
            }
        } while (err > 0.01);
        
        Console.Out.Write("Finished in {0} loops.\n", count);

        /* Run the network and see what it predicts. */
        unsafe
        {
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