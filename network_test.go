package goneural

import (
	"math/rand"
	"testing"
)

func TestNetwork(t *testing.T) {
	t.Run("Should resolve XOR problem after training", func(t *testing.T) {
		network := Network{
			LearningRate:       0.05,
			TransferFunction:   Tanh,
			TransferDerivative: TanhDerivative,
			layers: []*Layer{
				&Layer{
					neurons: []*Neuron{
						&Neuron{
							Weight:     []float64{rand.Float64(), rand.Float64()},
							Bias:       1,
							BiasWeight: rand.Float64(),
						},
						&Neuron{
							Weight:     []float64{rand.Float64(), rand.Float64()},
							Bias:       1,
							BiasWeight: rand.Float64(),
						},
						&Neuron{
							Weight:     []float64{rand.Float64(), rand.Float64()},
							Bias:       1,
							BiasWeight: rand.Float64(),
						},
					},
				},
				&Layer{
					neurons: []*Neuron{
						&Neuron{
							Weight:     []float64{rand.Float64(), rand.Float64(), rand.Float64()},
							Bias:       1,
							BiasWeight: rand.Float64(),
						},
					},
				},
			},
		}

		for i := 0; i < 60000; i++ {
			o1 := network.Think([]float64{0, 0})
			network.BackPropagate([]float64{0})

			o2 := network.Think([]float64{0, 1})
			network.BackPropagate([]float64{1})

			o3 := network.Think([]float64{1, 0})
			network.BackPropagate([]float64{1})

			o4 := network.Think([]float64{1, 1})
			network.BackPropagate([]float64{0})

			DebugNetwork(network, i, o1, o2, o3, o4)
		}

		var outputs []float64

		outputs = network.Think([]float64{0, 0})
		if !(outputs[0] <= 0.1) {
			t.Errorf("(0 XOR 0) should be close to 0")
		}

		outputs = network.Think([]float64{1, 0})
		if !(outputs[0] >= 0.9) {
			t.Errorf("(1 XOR 0) should be close to 1")
		}

		outputs = network.Think([]float64{0, 1})
		if !(outputs[0] >= 0.9) {
			t.Errorf("(0 XOR 1) should be close to 1")
		}

		outputs = network.Think([]float64{1, 1})
		if !(outputs[0] < 0.01) {
			t.Errorf("(1 XOR 1) should be close to 0")
		}
	})
}
