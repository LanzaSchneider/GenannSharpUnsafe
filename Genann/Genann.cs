using System;
using System.Diagnostics;

/// <summary>
/// Gen ann structure
/// </summary>
public class Genann
{
    /// <summary>
    /// Activate function delegate type
    /// </summary>
    public delegate double ActivateFunction(double a);

    /// <summary>
    /// How many inputs, outputs, and hidden neurons
    /// </summary>
    protected readonly int _Inputs, _HiddenLayers, _Hidden, _Outputs;

    /// <summary>
    /// Which activation function to use for hidden neurons. Default: <see cref="_ActivationSigmoidCached"/>
    /// </summary>
    protected readonly ActivateFunction _ActivationHidden;

    /// <summary>
    /// Which activation function to use for output. Default: <see cref="_ActivationSigmoidCached"/>
    /// </summary>
    protected readonly ActivateFunction _ActivationOutput;

    /// <summary>
    /// Total number of weights, and size of weights buffer
    /// </summary>
    protected readonly int _TotalWeights;

    /// <summary>
    /// Total number of neurons + inputs and size of output buffer
    /// </summary>
    protected readonly int _TotalNeurons;

    /// <summary>
    /// All weights (<see cref="_TotalWeights"/> long)
    /// </summary>
    protected double[] _Weight;

    /// <summary>
    /// Stores input array and output of each neuron (<see cref="_TotalNeurons"/> long)
    /// </summary>
    protected double[] _Output;

    /// <summary>
    /// Stores delta of each hidden and output neuron (<see cref="_TotalNeurons"/> - <see cref="_Inputs"/> long)
    /// </summary>
    protected double[] _Delta;

    /// <summary>
    /// Creates a new ann
    /// </summary>
    public Genann(int inputs, int hiddenLayers, int hidden, int outputs) : this(inputs, hiddenLayers, hidden, outputs,
        new Random())
    {
    }

    /// <summary>
    /// Creates a new ann with random
    /// </summary>
    public Genann(int inputs, int hiddenLayers, int hidden, int outputs, Random random)
    {
        Debug.Assert(hiddenLayers >= 0 && inputs >= 1 && outputs >= 1 && !(hiddenLayers > 0 && hidden < 1));
        var hiddenWeights = hiddenLayers != 0
            ? (inputs + 1) * hidden + (hiddenLayers - 1) * (hidden + 1) * hidden
            : 0;
        var outputWeights = (hiddenLayers != 0 ? hidden + 1 : inputs + 1) * outputs;
        var totalWeights = hiddenWeights + outputWeights;

        var totalNeurons = inputs + hidden * hiddenLayers + outputs;

        _Inputs = inputs;
        _HiddenLayers = hiddenLayers;
        _Hidden = hidden;
        _Outputs = outputs;

        _TotalWeights = totalWeights;
        _TotalNeurons = totalNeurons;

        // Allocate extra size for weights, outputs, and deltas.
        _Weight = new double[_TotalWeights];
        _Output = new double[_TotalNeurons];
        _Delta = new double[_TotalNeurons - _Inputs];

        Randomize(random);

        _ActivationHidden = _ActivationSigmoidCached;
        _ActivationOutput = _ActivationSigmoidCached;
    }

    /// <summary>
    /// Runs the feedforward algorithm to calculate the ann's output
    /// </summary>
    /// <param name="inputs">Array for inputs</param>
    /// <param name="outputs">Array for storage outputs, can be null</param>
    public unsafe void Run(double[] inputs, double[] outputs)
    {
        fixed ( double* output = _Output )
        {
            RunUnsafe(inputs, outputs, output);
        }
    }

    /// <summary>
    /// Runs the feedforward algorithm to calculate the ann's output
    /// </summary>
    /// <param name="inputs">Array for inputs</param>
    /// <param name="outputs">Array for storage outputs, can be null</param>
    /// <param name="outputPtr">The pointer to output array</param>
    /// <returns>The pointer to first output</returns>
    public unsafe double* RunUnsafe(double[] inputs, double[] outputs, double* outputPtr)
    {
        fixed (double* input = inputs, weight = _Weight, deltaWeight = _Delta)
        {
            var output = outputPtr;
            
            var w = weight;
            var o = output + _Inputs;
            var i = output;

            double* ret;

            // Copy the inputs to the scratch area, where we also store each neuron's output, for consistency. This way the first layer isn't a special case
            Array.Copy(inputs, _Output, _Inputs);

            int h, j, k;

            if (_HiddenLayers == 0)
            {
                ret = o;
                for (j = 0; j < _Outputs; ++j)
                {
                    double sum = *w++ * -1.0;
                    for (k = 0; k < _Inputs; ++k)
                    {
                        sum += *w++ * i[k];
                    }

                    *o++ = _ActivationOutput(sum);
                }

                // Output
                if (outputs != null)
                {
                    for (j = 0; j < _Outputs; j++)
                    {
                        outputs[j] = ret[j];
                    }
                }

                return ret;
            }

            // Figure input layer
            for (j = 0; j < _Hidden; ++j)
            {
                double sum = *w++ * -1.0;
                for (k = 0; k < _Inputs; ++k)
                {
                    sum += *w++ * i[k];
                }

                *o++ = _ActivationHidden(sum);
            }

            i += _Inputs;

            // Figure hidden layers, if any
            for (h = 1; h < _HiddenLayers; ++h)
            {
                for (j = 0; j < _Hidden; ++j)
                {
                    double sum = *w++ * -1.0;
                    for (k = 0; k < _Hidden; ++k)
                    {
                        sum += *w++ * i[k];
                    }

                    *o++ = _ActivationHidden(sum);
                }

                i += _Hidden;
            }

            ret = o;

            // Figure output layer
            for (j = 0; j < _Outputs; ++j)
            {
                double sum = *w++ * -1.0;
                for (k = 0; k < _Hidden; ++k)
                {
                    sum += *w++ * i[k];
                }

                *o++ = _ActivationOutput(sum);
            }

            // Sanity check that we used all weights and wrote all outputs.
            Debug.Assert(w - weight == _TotalWeights);
            Debug.Assert(o - output == _TotalNeurons);

            // Output
            if (outputs != null)
            {
                for (j = 0; j < _Outputs; j++)
                {
                    outputs[j] = ret[j];
                }
            }

            return ret;
        }
    }

    /// <summary>
    /// Does a single backprop update
    /// </summary>
    public void Train(double[] inputs, double[] desiredOutputs, double learningRate)
    {
        Train(inputs, desiredOutputs, 0, learningRate);
    }

    /// <summary>
    /// Does a single backprop update with desired outputs offset
    /// </summary>
    public unsafe void Train(double[] inputs, double[] desiredOutputs, int desiredOutputsOffset, double learningRate)
    {
        fixed (double* output = _Output, input = inputs, weight = _Weight, deltaWeight = _Delta, desired =
            desiredOutputs)
        {
            // To begin with, we must run the network forward
            RunUnsafe(inputs, null, output);
            
            // First set the output layer deltas
            int j;
            {
                var o = output + _Inputs + _Hidden * _HiddenLayers;
                var d = deltaWeight + _Hidden * _HiddenLayers;
                var t = desired + desiredOutputsOffset;

                // Set output layer deltas
                if (_ActivationOutput == _ActivationLinear)
                {
                    for (j = 0; j < _Outputs; j++)
                    {
                        *d++ = *t++ - *o++;
                    }
                }
                else
                {
                    for (j = 0; j < _Outputs; j++)
                    {
                        *d++ = (*t - *o) * *o * (1.0 - *o);
                        ++o;
                        ++t;
                    }
                }
            }

            // Set hidden layer deltas, start on last layer and 
                // Find first output deltawork backwards
            // Note that loop is skipped in the case of hidden_layers == 0
            int h, k;
            for (h = _HiddenLayers - 1; h >= 0; h--)
            {
                // Find first output and delta in this layer
                var o = output + _Inputs + h * _Hidden;
                var d = deltaWeight + h * _Hidden;

                // Find first delta in following layer (which may be hidden or output)
                var dd = deltaWeight + (h + 1) * _Hidden;

                // Find first weight in following layer (which may be hidden or output)
                var ww = weight + (_Inputs + 1) * _Hidden + (_Hidden + 1) * _Hidden * h;

                for (j = 0; j < _Hidden; ++j)
                {
                    double delta = 0;

                    for (k = 0; k < (h == _HiddenLayers - 1 ? _Outputs : _Hidden); ++k)
                    {
                        var forwardDelta = dd[k];
                        var windex = k * (_Hidden + 1) + j + 1;
                        var forwardWeight = ww[windex];
                        delta += forwardDelta * forwardWeight;
                    }

                    *d = *o * (1.0 - *o) * delta;
                    ++d;
                    ++o;
                }
            }

            // Train the outputs
            {
                var d = deltaWeight + _Hidden * _HiddenLayers; // First output delta.

                // Find first weight to first output delta
                var w = weight + (_HiddenLayers != 0
                    ? (_Inputs + 1) * _Hidden + (_Hidden + 1) * _Hidden * (_HiddenLayers - 1)
                    : 0);

                // Find first output in previous layer
                var i = output + (_HiddenLayers != 0
                    ? _Inputs + _Hidden * (_HiddenLayers - 1)
                    : 0);

                // Set output layer weights
                for (j = 0; j < _Outputs; ++j)
                {
                    *w++ += *d * learningRate * -1.0;
                    for (k = 1; k < (_HiddenLayers != 0 ? _Hidden : _Inputs) + 1; ++k)
                    {
                        *w++ += *d * learningRate * i[k - 1];
                    }

                    ++d;
                }

                Debug.Assert(w - weight == _TotalWeights);
            }

            // Train the hidden layers
            for (h = _HiddenLayers - 1; h >= 0; --h)
            {
                // Find first delta in this layer
                var d = deltaWeight + h * _Hidden;

                // Find first input to this layer
                var i = output + (h != 0
                    ? _Inputs + _Hidden * (h - 1)
                    : 0);

                // Find first weight to this layer
                double* w = weight + (h != 0
                    ? (_Inputs + 1) * _Hidden + (_Hidden + 1) * _Hidden * (h - 1)
                    : 0);
                
                for (j = 0; j < _Hidden; ++j)
                {
                    *w++ += *d * learningRate * -1.0;
                    for (k = 1; k < (h == 0 ? _Inputs : _Hidden) + 1; ++k)
                    {
                        *w++ += *d * learningRate * i[k - 1];
                    }

                    ++d;
                }
            }
        }
    }

    /// <summary>
    /// All weights data
    /// </summary>
    public double[] Weights => _Weight;

    /// <summary>
    /// Total number of weights, and size of weights buffer
    /// </summary>
    public int TotalWeights => _TotalWeights;

    /// <summary>
    /// Total number of neurons + inputs and size of output buffer
    /// </summary>
    public int TotalNeurons => _TotalNeurons;

    /// <summary>
    /// Sets weights randomly
    /// (Called by init)
    /// </summary>
    public void Randomize(Random random)
    {
        for (var i = 0; i < _TotalWeights; i++)
        {
            _Weight[i] = random.NextDouble() - 0.5;
        }
    }

    #region Activation

    private static double _ActivationLinear(double a) => a;

    private static double _ActivationThreshold(double a) => a > 0 ? 1 : 0;

    private static double _ActivationSigmoid(double a)
    {
        if (a < -45.0) return 0;
        if (a > 45.0) return 1;
        return 1.0 / (1 + Math.Exp(-a));
    }

    private static double[] _Lookup;
    private static double _Interval;
    
    private static double _ActivationSigmoidCached(double a)
    {
        Debug.Assert(!double.IsNaN(a));
        
        const double sigmoidDomMin = -15.0;
        const double sigmoidDomMax = 15.0;

        const int LOOKUP_SIZE = 4096;

        if (_Lookup == null)
        {
            const double f = (sigmoidDomMax - sigmoidDomMin) / LOOKUP_SIZE;
            int i;

            _Interval = LOOKUP_SIZE / (sigmoidDomMax - sigmoidDomMin);
            _Lookup = new double[LOOKUP_SIZE];
            for (i = 0; i < LOOKUP_SIZE; ++i) {
                _Lookup[i] = _ActivationSigmoid(sigmoidDomMin + f * i);
            }
        }

        if (a < sigmoidDomMin) return _Lookup[0];
        if (a >= sigmoidDomMax) return _Lookup[LOOKUP_SIZE - 1];
        
        var j = (uint)((a-sigmoidDomMin)*_Interval+0.5);

        if (j >= LOOKUP_SIZE)
            return _Lookup[LOOKUP_SIZE - 1];
        
        return _Lookup[j];
    }

    #endregion

    #region ActivationExtra

    private static double _ActivationRectifiedLinearUnit(double a) => Math.Max(0, a);

    private static double _ActivationRectifiedLinearUnitLeaky(double a) => Math.Max(0.01 * a, a);

    private static double _ActivationHyperbolicTangent(double a) => Math.Tanh(a);

    private static double _ActivationExponentialLinearUnit(double a)
    {
        if (a > 0)
            return a;
        return 0.01 * (Math.Exp(a) - 1);
    }

    private static double _ActivationGaussianErrorLinearUnit(double x)
    {
        return x / (1 + Math.Exp(-1.702 * x));
    }

    #endregion
}