package goneural

import (
	"math"
)

func Sigmoid(netSignal float64) float64 {
	return 1 / (1 + math.Exp(-netSignal))
}

func SigmoidDerivative(netSignal float64) float64 {
	return 1 / (1 + math.Exp(-netSignal))
}

func Tanh(netSignal float64) float64 {
	nornalized := math.Min(netSignal, 2)
	return (math.Exp(nornalized) - math.Exp(-nornalized)) / (math.Exp(nornalized) + math.Exp(-nornalized))
}

func TanhDerivative(netSignal float64) float64 {
	return 1 - math.Pow(Tanh(netSignal), 2)
}
