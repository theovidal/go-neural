package neural

import "math/rand/v2"

type Initializer func(_, _ int) float64

func NormalInitializer(m, s float64) Initializer {
	return func(_, _ int) float64 {
		return rand.NormFloat64()*s + m
	}
}
