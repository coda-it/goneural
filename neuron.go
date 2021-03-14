package goneural

import (
	"math"
)

// Neuron - single perceptron struct
type Neuron struct {
	Bias   float64
	Weight []float64
	Output float64
	Input  float64
	Error  float64
	Delta  float64
}

func (n *Neuron) sigmoid(netSignal float64) float64 {
	return 1 / (1 + math.Exp(-netSignal))
}

// SigmoidDerivative - sigmoid derivative used for learning neural network
func (n *Neuron) SigmoidDerivative(outputSignal float64) float64 {
	return outputSignal * (1.0 - outputSignal)
}

// TransferFunction - calculates net signal into gross output signal
func (n *Neuron) TransferFunction(netSignal float64) float64 {
	return n.sigmoid(netSignal)
}
