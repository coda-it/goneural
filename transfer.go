package goneural

import (
	"math"
)

// Sigmoid - activation function
func Sigmoid(netSignal float64) float64 {
	return 1 / (1 + math.Exp(-netSignal))
}

// SigmoidDerivative - activation function
func SigmoidDerivative(netSignal float64) float64 {
	return 1 / (1 + math.Exp(-netSignal))
}

// Tanh - activation function
func Tanh(netSignal float64) float64 {
	nornalized := math.Min(netSignal, 2)
	return (math.Exp(nornalized) - math.Exp(-nornalized)) / (math.Exp(nornalized) + math.Exp(-nornalized))
}

// TanhDerivative - activation function
func TanhDerivative(netSignal float64) float64 {
	return 1 - math.Pow(Tanh(netSignal), 2)
}
