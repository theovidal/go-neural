package neural

import (
	"errors"
	"fmt"
	"github.com/theovidal/beta-project/matrix"
	"math/rand/v2"
)

func (n *Neural) Train(xSet, ySet [][]float64, epsilon float64, iterations int, verbose bool) error {
	if len(xSet) != len(ySet) {
		return errors.New("there should be an output for every input in training set")
	}

	for iteration := range iterations {
		if verbose {
			fmt.Printf("--- Iteration %d/%d ---\n", iteration, iterations)
		}

		rand.Shuffle(len(xSet), func(i, j int) {
			xSet[i], xSet[j] = xSet[j], xSet[i]
			ySet[i], ySet[j] = ySet[j], ySet[i]
		})

		for i := range len(xSet) {
			// -------------
			// STEP 0 - FORMAT THE DATA AND VERIFY USER INPUT
			// -------------

			xIn := matrix.FromVector(xSet[i], true)
			yOut := matrix.FromVector(ySet[i], true)

			if n.config.Inputs != xIn.GetN() {
				return fmt.Errorf("invalid sample %d: expected %d inputs, got %d", i, n.config.Inputs, xIn.GetN())
			}

			numLayers := len(n.config.Layout)
			if n.config.Layout[numLayers-1] != yOut.GetN() {
				return fmt.Errorf("invalid sample %d: expected %d outputs, got %d", i, n.config.Layout[numLayers-1], yOut.GetN())
			}

			// -------------
			// STEP 1 - PREDICT THE OUTPUT FOR THIS SAMPLE
			// -------------

			_, x, y, err := n.PredictColumn(xIn)
			if err != nil {
				return err
			}

			// -------------
			// STEP 2 - CALCULATE INITIAL ΔEy
			// -------------

			yOut.Scale(-1)
			deltaEY, err := matrix.Add(y[numLayers-1], yOut)
			if err != nil {
				return err
			}

			// -------------
			// STEP 3 - BACK-PROPAGATE
			// -------------

			for i = numLayers - 1; i >= 0; i-- {
				// -------------
				// STEP 3.1 - CALCULATE ΔEx FOR THIS LAYER
				// -------------

				x[i].ApplyFunction(n.config.Activator.Df)
				deltaEX, err := matrix.Hadamard(deltaEY, x[i])
				if err != nil {
					return err
				}

				// -------------
				// STEP 3.2 - CALCULATE ΔEw FOR THIS LAYER
				// -------------

				previousY := xIn
				if i != 0 {
					previousY = y[i-1]
				}
				deltaEW, err := matrix.Multiply(deltaEY, matrix.Transpose(previousY))
				if err != nil {
					return err
				}

				// -------------
				// STEP 3.3 - CALCULATE ΔEy FOR THIS LAYER
				// This will serve for the next iteration, to use the ΔEy of the previous layer
				// -------------

				deltaEY, err = matrix.Multiply(matrix.Transpose(n.weights[i]), deltaEX)
				if err != nil {
					return err
				}

				// -------------
				// STEP 3.4 - ADJUST THE WEIGHTS FOR THIS LAYER USING ΔEw
				// -------------

				deltaEW.Scale(-epsilon)
				err = n.weights[i].Add(deltaEW)
				if err != nil {
					return err
				}
			}
		}
	}

	return nil
}
