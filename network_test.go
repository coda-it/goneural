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
							Weight:     []float64{-rand.Float64(), -rand.Float64()},
							Bias:       1,
							BiasWeight: rand.Float64(),
						},
					},
				},
				&Layer{
					neurons: []*Neuron{
						&Neuron{
							Weight:     []float64{rand.Float64(), -rand.Float64()},
							Bias:       1,
							BiasWeight: rand.Float64(),
						},
					},
				},
			},
		}

		for i := 0; i < 20000; i++ {
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
		if !(outputs[0] < 0.1) {
			t.Errorf("(1 XOR 1) should be close to 0")
		}
	})

	t.Run("Should recognize simple image / and \\", func(t *testing.T) {
		network := Network{
			LearningRate:       0.05,
			TransferFunction:   Tanh,
			TransferDerivative: TanhDerivative,
			layers: []*Layer{
				&Layer{
					neurons: []*Neuron{
						&Neuron{
							Weight:     []float64{-rand.Float64(), rand.Float64(), -rand.Float64(), rand.Float64()},
							Bias:       1,
							BiasWeight: rand.Float64(),
						},
						&Neuron{
							Weight:     []float64{rand.Float64(), -rand.Float64(), rand.Float64(), -rand.Float64()},
							Bias:       1,
							BiasWeight: rand.Float64(),
						},
					},
				},
				&Layer{
					neurons: []*Neuron{
						&Neuron{
							Weight:     []float64{-rand.Float64(), rand.Float64()},
							Bias:       1,
							BiasWeight: rand.Float64(),
						},
						&Neuron{
							Weight:     []float64{rand.Float64(), -rand.Float64()},
							Bias:       1,
							BiasWeight: rand.Float64(),
						},
					},
				},
			},
		}

		for i := 0; i < 2000; i++ {
			network.Think([]float64{0, 0, 0, 0})
			network.BackPropagate([]float64{0, 0})

			network.Think([]float64{0, 1, 1, 0})
			network.BackPropagate([]float64{1, 0})

			network.Think([]float64{1, 0, 0, 1})
			network.BackPropagate([]float64{0, 1})

			network.Think([]float64{1, 1, 1, 1})
			network.BackPropagate([]float64{0, 0})
		}

		var outputs []float64

		outputs = network.Think([]float64{0, 0, 0, 0})
		if !(outputs[0] <= 0.1 && outputs[1] <= 0.1) {
			t.Errorf("\n" +
				"00\n" +
				"00" + " should not be recognized as any shape",
			)
		}

		outputs = network.Think([]float64{0, 1, 1, 0})
		if !(outputs[0] >= 0.9 && outputs[1] <= 0.1) {
			t.Errorf("\n" +
				"01\n" +
				"10" + " should be recognized as /",
			)
		}

		outputs = network.Think([]float64{1, 0, 0, 1})
		if !(outputs[0] <= 0.1 && outputs[1] >= 0.9) {
			t.Errorf("\n" +
				"10\n" +
				"01" + " should be recognized as \\",
			)
		}

		outputs = network.Think([]float64{1, 1, 1, 1})
		if !(outputs[0] < 0.1 && outputs[1] <= 0.1) {
			t.Errorf("\n" +
				"11\n" +
				"11" + " should not be recognized as any shape",
			)
		}
	})
}
