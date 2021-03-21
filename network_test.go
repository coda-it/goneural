package goneural

import (
	"testing"
)

func TestNetwork(t *testing.T) {
	t.Run("Should resolve XOR problem after training", func(t *testing.T) {
		network := Network{
			LearningRate:       0.1,
			TransferFunction:   Tanh,
			TransferDerivative: TanhDerivative,
			layers: []*Layer{
				&Layer{
					neurons: []*Neuron{
						&Neuron{
							Bias:   0,
							Weight: []float64{0.1, 0.5},
						},
						&Neuron{
							Bias:   0,
							Weight: []float64{1.3, 0.1},
						},
					},
				},
				&Layer{
					neurons: []*Neuron{
						&Neuron{
							Bias:   0,
							Weight: []float64{0.5, 0.5},
						},
					},
				},
			},
		}

		var outputs []float64

		for i := 0; i < 100; i++ {
			outputs = network.Think([]float64{0, 0})
			network.BackPropagate(outputs, []float64{0})
		}
		for i := 0; i < 100; i++ {
			outputs = network.Think([]float64{0, 1})
			network.BackPropagate(outputs, []float64{1})
		}
		for i := 0; i < 100; i++ {
			outputs = network.Think([]float64{1, 0})
			network.BackPropagate(outputs, []float64{1})
		}
		for i := 0; i < 100; i++ {
			outputs = network.Think([]float64{1, 1})
			network.BackPropagate(outputs, []float64{1})
		}

		outputs = network.Think([]float64{0, 0})
		if len(outputs) != 1 && outputs[0] <= 0.1 {
			t.Errorf("(0,0) XOR not resolved properly")
		}

		outputs = network.Think([]float64{1, 0})
		if len(outputs) != 1 && outputs[0] >= 0.9 {
			t.Errorf("(1,0) XOR not resolved properly")
		}

		outputs = network.Think([]float64{0, 1})
		if len(outputs) != 1 && outputs[0] >= 0.9 {
			t.Errorf("(0,1) XOR not resolved properly")
		}

		outputs = network.Think([]float64{1, 1})
		if len(outputs) != 1 && outputs[0] >= 0.9 {
			t.Errorf("(1,1) XOR not resolved properly")
		}
	})
}
