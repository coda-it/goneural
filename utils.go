package goneural

import (
	"fmt"
	"os"
	"os/exec"
)

// DebugNetwork - renders temporary dashboard for debugging purposes
func DebugNetwork(network Network, epoch int, outputs ...[]float64) {
	fmt.Printf("Epoch = %d\n", epoch)

	for lk, l := range network.layers {
		for nk, n := range l.neurons {
			fmt.Printf("N[%d,%d]\tI=%f\tW=%f\tE=%f\n", lk, nk, n.Input, n.Weight, n.Error)
		}
	}

	for ok, o := range outputs {
		fmt.Printf("OUTPUT[%d] = %f\n", ok, o)
	}

	c := exec.Command("clear")
	c.Stdout = os.Stdout
	c.Run()
}
