package matrix

func (A *Matrix[T]) GetDimensions() (int, int) {
	return A.n, A.m
}

func (A *Matrix[T]) GetN() int {
	return A.n
}

func (A *Matrix[T]) GetM() int {
	return A.m
}

func (A *Matrix[T]) ToTable() [][]T {
	result := make([][]T, A.n)
	for i := range A.n {
		result[i] = make([]T, A.m)
		for j := range A.m {
			result[i][j] = A.Get(i, j)
		}
	}
	return result
}

func (A *Matrix[T]) Copy() *Matrix[T] {
	t := make([]T, A.n*A.m)
	copy(t, A.t)
	return &Matrix[T]{
		A.n, A.m, t,
	}
}

func (A *Matrix[T]) ToRow() (result []T) {
	result = make([]T, A.n*A.m)
	copy(result, A.t)
	return
}
