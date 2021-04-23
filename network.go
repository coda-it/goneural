package goneural

// Network - layerd feed forward neural network
type Network struct {
	layers             []*Layer
	LearningRate       float64
	TransferFunction   func(netSignal float64) float64
	TransferDerivative func(netSignal float64) float64
	DebugMode          bool
	MeanError          float64
}

// Think - feeds input data aand forward through whole network and produce outputs
func (n *Network) Think(inputs []float64) []float64 {
	var outputs []float64

	for lk, l := range n.layers {
		for _, p := range l.neurons {
			activation := 0.0
			p.Input = []float64{}

			if lk == 0 {
				for ik, i := range inputs {
					activation += i * p.Weight[ik]
					p.Input = append(p.Input, i)
				}
			} else {
				previousLayer := n.layers[lk-1]

				for nk, n := range previousLayer.neurons {
					p.Input = append(p.Input, n.Output)
					activation += n.Output*p.Weight[nk]
				}
			}
			activation += p.Bias*p.BiasWeight
			p.Output = n.TransferFunction(activation)

			if lk == len(n.layers)-1 {
				outputs = append(outputs, p.Output)
			}
		}
	}

	return outputs
}

// BackPropagate - adjust neural network weights
func (n *Network) BackPropagate(expected []float64) {
	for lk := len(n.layers) - 1; lk >= 0; lk-- {
		l := n.layers[lk]

		for pk, p := range l.neurons {
			if lk == len(n.layers)-1 {
				p.Error = expected[pk] - p.Output
			} else {
				nextLayer := n.layers[lk+1]
				p.Error = 0.0

				for _, i := range nextLayer.neurons {
					p.Error += i.Weight[pk] * i.Delta
				}
			}

			p.Delta = p.Error * n.TransferDerivative(p.Output)

			for wk := range p.Weight {
				p.Weight[wk] = p.Weight[wk] + n.LearningRate*p.Delta*p.Input[wk]
			}
			p.BiasWeight = p.BiasWeight + n.LearningRate*p.Delta*p.Bias
		}
	}
}
