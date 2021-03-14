package goneural

import (
	"testing"
)

func TestNetwork(t *testing.T) {
	t.Run("Should adjust weights to get closer to the expected value", func(t *testing.T) {
		network := Network{
			LearningRate: 0.1,
			layers: []*Layer{
				&Layer{
					neurons: []*Neuron{
						&Neuron{
							Bias:   0,
							Weight: []float64{0.5, 0.5},
						},
						&Neuron{
							Bias:   0,
							Weight: []float64{0.5, 0.5},
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

		for i := 0; i < 1000; i++ {
			outputs = network.Think([]float64{0, 1})
			network.BackPropagate(outputs, []float64{1})
		}

		outputs = network.Think([]float64{0, 0})

		if len(outputs) != 1 && outputs[0] == 0.9684122982829187 {
			t.Errorf("Output signal is wrongly calculated")
		}
	})
}
