package goneural

// Neuron - single perceptron struct
type Neuron struct {
	Bias       float64
	BiasWeight float64
	Weight     []float64
	Output     float64
	Input      []float64
	Error      float64
	Delta      float64
}
