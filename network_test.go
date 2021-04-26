package goneural

import (
	"fmt"
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

		for i := 0; i < 2000; i++ {
			network.Think([]float64{0, 0})
			network.BackPropagate([]float64{0})

			network.Think([]float64{0, 1})
			network.BackPropagate([]float64{1})

			network.Think([]float64{1, 0})
			network.BackPropagate([]float64{1})

			network.Think([]float64{1, 1})
			network.BackPropagate([]float64{0})
		}

		var outputs []float64

		outputs = network.Think([]float64{0, 0})
		fmt.Println("xor:", outputs)
		if !(outputs[0] <= 0.1) {
			t.Errorf("(0 XOR 0) should be close to 0")
		}

		outputs = network.Think([]float64{1, 0})
		fmt.Println("xor:", outputs)
		if !(outputs[0] >= 0.9) {
			t.Errorf("(1 XOR 0) should be close to 1")
		}

		outputs = network.Think([]float64{0, 1})
		fmt.Println("xor:", outputs)
		if !(outputs[0] >= 0.9) {
			t.Errorf("(0 XOR 1) should be close to 1")
		}

		outputs = network.Think([]float64{1, 1})
		fmt.Println("xor:", outputs)
		if !(outputs[0] < 0.1) {
			t.Errorf("(1 XOR 1) should be close to 0")
		}
	})
}
