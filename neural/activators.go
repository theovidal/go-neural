package neural

import "math"

type Activator struct {
	f  func(float64) float64
	Df func(float64) float64
}

func SigmoidActivator() Activator {
	sigmoid := func(x float64) float64 {
		return 1 / (1 + math.Exp(-x))
	}

	return Activator{
		f: sigmoid,
		Df: func(x float64) float64 {
			return sigmoid(x) * (1 - sigmoid(x))
		},
	}
}
