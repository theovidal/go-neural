package matrix

import (
	"errors"
	"fmt"
)

// TODO: find a better approach
// 3 cases :
// - Square matrix: TODO
// - Vector column or row : transform to row or column
// - Other
func Transpose[T Number](A *Matrix[T]) *Matrix[T] {
	if A.n == 1 || A.m == 1 {
		// If A is a row (n == 1) then B should be a column
		B := FromVector(A.ToRow(), A.n == 1)
		return B
	}

	B := New[T](A.m, A.n)
	for i := range A.m {
		for j := range A.n {
			B.Set(i, j, A.Get(j, i))
		}
	}
	return B
}

func Add[T Number](A, B *Matrix[T]) (C *Matrix[T], err error) {
	if A.n != B.n || A.m != B.m {
		err = errors.New("dimensions are not equal")
		return
	}
	C = New[T](A.n, A.m)
	_ = C.Add(A)
	_ = C.Add(B)
	return
}

func Scale[T Number](A *Matrix[T], scale T) (C *Matrix[T]) {
	C = FromMatrix(A)
	C.Scale(scale)
	return
}

func Multiply[T Number](A, B *Matrix[T]) (C *Matrix[T], err error) {
	if A.m != B.n {
		err = errors.New(fmt.Sprintf("dimensions (%d, %d) and (%d, %d) don't match", A.n, A.m, B.n, B.m))
		return
	}

	C = New[T](A.n, B.m)

	for i := range A.n {
		for j := range B.m {
			var value T
			for k := range A.m {
				a := A.Get(i, k)
				b := B.Get(k, j)
				value += a * b
			}
			C.Set(i, j, value)
		}
	}
	return
}

func Hadamard[T Number](A, B *Matrix[T]) (C *Matrix[T], err error) {
	if A.n != B.n || A.m != B.m {
		err = errors.New("dimensions are not equal")
		return
	}
	C = New[T](A.n, A.m)
	_ = C.Add(A)
	_ = C.Hadamard(B)
	return
}

func ApplyFunction[T Number](A *Matrix[T], f func(_ T) T) (C *Matrix[T]) {
	C = FromMatrix(A)
	C.ApplyFunction(f)
	return
}

func AreEqual[T Number](A, B *Matrix[T]) bool {
	if A.n != B.n || A.m != B.m {
		return false
	}
	for i := range A.n * A.m {
		if A.t[i] != B.t[i] {
			return false
		}
	}
	return true
}
