package neural

import "github.com/theovidal/beta-project/matrix"

type Config struct {
	Inputs      int
	Layout      []int
	Activator   Activator
	Initializer Initializer
}

type Neural struct {
	weights []*matrix.Matrix[float64]
	biases  []*matrix.Matrix[float64]
	config  Config
}

func New(config Config) *Neural {
	var weights []*matrix.Matrix[float64]
	var biases []*matrix.Matrix[float64]

	for i := range len(config.Layout) {
		var m int
		if i == 0 {
			m = config.Inputs
		} else {
			m = config.Layout[i-1]
		}

		weights = append(weights, matrix.Initialize(config.Layout[i], m, config.Initializer))
		biases = append(biases, matrix.Initialize(config.Layout[i], 1, config.Initializer))
	}

	return &Neural{
		weights: weights,
		biases:  biases,
		config:  config,
	}
}
