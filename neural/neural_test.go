package neural

import (
	"fmt"
	"testing"
)

func TestNeural_Predict(t *testing.T) {
	neural := New(Config{
		Inputs:      4,
		Layout:      []int{5, 5, 3},
		Activator:   SigmoidActivator(),
		Initializer: NormalInitializer(0, 1),
	})

	_, x, y, err := neural.Predict([]float64{4, 8, 7, 1})
	if err != nil {
		t.Error(err)
	}

	for i := range len(x) {
		n, m := x[i].GetDimensions()
		fmt.Printf("%d %d\n", n, m)
		x[i].Print()
		y[i].Print()
	}
}

func TestNeural_Train(t *testing.T) {
	neural := New(Config{
		Inputs:      3,
		Layout:      []int{1},
		Activator:   SigmoidActivator(),
		Initializer: NormalInitializer(0, 1),
	})

	err := neural.Train(
		[][]float64{
			{1, 0, 0},
			{0, 1, 0},
			{0, 0, 1},
			{1, 1, 0},
		},
		[][]float64{
			{1},
			{0},
			{0},
			{1},
		},
		0.01,
		400,
		true,
	)
	if err != nil {
		t.Error(err)
	}

	result, _, _, err := neural.Predict([]float64{0, 0, 1})
	if err != nil {
		t.Error(err)
	}
	fmt.Printf("Prediction: %.4f\n", result[0])
}
