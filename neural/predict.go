package neural

import (
	"errors"
	"github.com/theovidal/beta-project/matrix"
)

func (n *Neural) Predict(input []float64) ([]float64, []*matrix.Matrix[float64], []*matrix.Matrix[float64], error) {
	result, x, y, err := n.PredictColumn(matrix.FromVector(input, true))
	return result.ToRow(), x, y, err
}

func (n *Neural) PredictColumn(input *matrix.Matrix[float64]) (result *matrix.Matrix[float64], x []*matrix.Matrix[float64], y []*matrix.Matrix[float64], err error) {
	if input.GetM() != 1 {
		err = errors.New("input should be a vector column")
		return
	}

	vec := input.Copy()
	for i, w := range n.weights {
		b := n.biases[i]

		if err = vec.MultiplyLeft(w); err != nil {
			return
		}

		if err = vec.Add(b); err != nil {
			return
		}

		x = append(x, vec.Copy())

		vec.ApplyFunction(n.config.Activator.f)
		y = append(y, vec.Copy())
	}

	result = y[len(y)-1]
	return
}
