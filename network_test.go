package goneural

import (
	"fmt"
	"math/rand"
	"os"
	"os/exec"
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
							Weight: []float64{rand.Float64(), rand.Float64()},
							Bias: 1,
							BiasWeight: rand.Float64(),
						},
						&Neuron{
							Weight: []float64{rand.Float64(), rand.Float64()},
							Bias: 1,
							BiasWeight: rand.Float64(),
						},
						&Neuron{
							Weight: []float64{rand.Float64(), rand.Float64()},
							Bias: 1,
							BiasWeight: rand.Float64(),
						},
					},
				},
				&Layer{
					neurons: []*Neuron{
						&Neuron{
							Weight: []float64{rand.Float64(), rand.Float64(), rand.Float64()},
							Bias: 1,
							BiasWeight: rand.Float64(),
						},
					},
				},
			},
		}

		var outputs []float64

		for i := 0; i < 60000; i++ {
			outputs1 := network.Think([]float64{0, 0})
			network.BackPropagate([]float64{0})

			outputs2 := network.Think([]float64{0, 1})
			network.BackPropagate([]float64{1})

			outputs3 := network.Think([]float64{1, 0})
			network.BackPropagate([]float64{1})

			outputs4 := network.Think([]float64{1, 1})
			network.BackPropagate([]float64{0})

			fmt.Printf("Epoch = %d\n", i)
			fmt.Printf("W[0,0] = %f, %f\n", network.layers[0].neurons[0].Weight, network.layers[0].neurons[0].BiasWeight)
			fmt.Printf("W[0,1] = %f, %f\n", network.layers[0].neurons[1].Weight, network.layers[0].neurons[1].BiasWeight)
			fmt.Printf("W[1,0] = %f, %f\n", network.layers[1].neurons[0].Weight, network.layers[1].neurons[0].BiasWeight)
			fmt.Printf("E[0,0] = %f, I=%f\n", network.layers[0].neurons[0].Error, network.layers[0].neurons[0].Input)
			fmt.Printf("E[0,1] = %f, I=%f\n", network.layers[0].neurons[1].Error, network.layers[0].neurons[1].Input)
			fmt.Printf("E[1,0] = %f, I=%f\n", network.layers[1].neurons[0].Error, network.layers[1].neurons[0].Input)
			fmt.Printf("O[0 XOR 0] = %f\n", outputs1[0])
			fmt.Printf("O[0 XOR 1] = %f\n", outputs2[0])
			fmt.Printf("O[1 XOR 0] = %f\n", outputs3[0])
			fmt.Printf("O[1 XOR 1] = %f\n", outputs4[0])
			c := exec.Command("clear")
			c.Stdout = os.Stdout
			c.Run()
		}

		outputs = network.Think([]float64{0, 0})
		fmt.Println("(0,0)", outputs)
		if !(outputs[0] <= 0.1) {
			t.Errorf("(0,0) XOR not resolved properly")
		}
		outputs = network.Think([]float64{1, 0})
		fmt.Println("(1,0)", outputs)
		if !(outputs[0] >= 0.9) {
			t.Errorf("(1,0) XOR not resolved properly")
		}
		fmt.Println("(0,1)", outputs)
		outputs = network.Think([]float64{0, 1})
		if !(outputs[0] >= 0.9) {
			t.Errorf("(0,1) XOR not resolved properly")
		}
		fmt.Println("(1,1)", outputs)
		outputs = network.Think([]float64{1, 1})
		if !(outputs[0] <= 0.1) {
			t.Errorf("(1,1) XOR not resolved properly")
		}
	})
}
