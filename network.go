package goneural

// Netwok - layerd feed forward neural network
type Network struct {
	layers       []*Layer
	LearningRate float64
}

// Think - feeds singals forward through whole network and produce output signals
func (n *Network) Think(inputs []float64) []float64 {
	var outputs []float64

	for lk, l := range n.layers {
		for _, p := range l.neurons {
			signal := 0.0

			if lk == 0 {
				for ik, i := range inputs {
					signal += i * p.Weight[ik]
				}
			} else {
				previousLayer := n.layers[lk-1]

				for ik, i := range previousLayer.neurons {
					signal += i.Output*p.Weight[ik] + p.Bias
				}
			}

			p.Input = signal
			p.Output = p.TransferFunction(signal)

			if lk == len(n.layers)-1 {
				outputs = append(outputs, p.Output)
			}
		}
	}

	return outputs
}

func (n *Network) errorCalculation(outputs []float64, expected []float64) {
	for lk := len(n.layers) - 1; lk >= 0; lk-- {
		l := n.layers[lk]

		for pk, p := range l.neurons {
			if lk == len(n.layers)-1 {
				p.Delta = (expected[pk] - p.Output) * p.SigmoidDerivative(p.Output)

			} else {
				nextLayer := n.layers[lk+1]

				for _, i := range nextLayer.neurons {
					p.Delta += i.Weight[pk] * i.Delta * p.SigmoidDerivative(p.Output)
				}
			}
		}
	}
}

func (n *Network) updateWeights() {
	for _, l := range n.layers {
		for _, p := range l.neurons {
			for wk, _ := range p.Weight {
				p.Weight[wk] += n.LearningRate * p.Delta * p.Input
			}
		}
	}
}

// BackPropagate - adjust neural network weights
func (n *Network) BackPropagate(outputs []float64, expected []float64) {
	n.errorCalculation(outputs, expected)
	n.updateWeights()
}
