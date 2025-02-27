package main

import (
	"gonum.org/v1/gonum/mat"
	"os"
	"syscall"
	"testing"
)

func Test_LoadAdjacency(t *testing.T) {
	A, err := LoadAdjacency("./karate.csv")
	if err != nil {
		t.Error("unexpected error", err)
		return
	}
	rows, cols := A.Dims()
	if rows != 34 || cols != 34 {
		t.Error("dimension error")
	}
	if w := A.At(0, 11); w != 1. {
		t.Error("expected 1 got", w)
	}
	if w := A.At(0, 29); w != 0. {
		t.Error("expected 0 got", w)
	}
}

func Test_LoadAdjacencyError(t *testing.T) {
	_, err := LoadAdjacency("./unknown.txt")
	if e, ok := err.(*os.PathError); ok && e.Err == syscall.ENOSPC {
		t.Error("an error was expected")
	}
}

func Test_GetDegrees(t *testing.T) {
	A, _ := LoadAdjacency("./karate.csv")
	D := GetDegrees(A)
	if d := D.At(0, 0); d != 16 {
		t.Error("expected 6 got", d)
	}
}

func Test_GetLaplacian(t *testing.T) {
	A, _ := LoadAdjacency("./karate.csv")
	L := GetLaplacian(A)

	if w := L.At(0, 11); w != 1. {
		t.Error("expected 1 got", w)
	}
	if d := L.At(0, 0); d != -16 {
		t.Error("expected 12 got", d)
	}
}

func Test_GetSmallEigenVectors(t *testing.T) {
	A, _ := LoadAdjacency("./karate.csv")
	L := GetLaplacian(A)
	E := GetSmallestEigenVectors(L, 4)
	rows, cols := E.Dims()
	if rows != 34 {
		t.Error("expected n rows got", rows)
	}
	if cols != 4 {
		t.Error("expected 2 rows got", cols)
	}
	for i := 0; i < 4; i++ {
		AssertEigenVectorNorm(E, i, t)
	}
}

func AssertEigenVectorNorm(E mat.Matrix, i int, t *testing.T) {
	eigen := mat.NewDense(34, 1, mat.Col(nil, i, E))
	norm := mat.Norm(eigen, 2)
	if norm < .99999 || norm > 1.00001 {
		t.Error("eigen vectors should have norm 1")
	}
}

func Test_KMeans(t *testing.T) {
	A, _ := LoadAdjacency("./karate.csv")
	L := GetLaplacian(A)
	E := GetSmallestEigenVectors(L, 2)
	labels, _ := KMeans(E, 2, 10)

	l00 := labels[0]
	l33 := labels[33]

	if l00 == l33 {
		t.Error("1 and 34 should not be in the same community")
	}
	if labels[1] != l00 {
		t.Error("1 and 2 should be in the same community")
	}
	if labels[32] != l33 {
		t.Error("33 and 34 should be in the same community")
	}
}

func Benchmark_KMeans(b *testing.B) {
	A, _ := LoadAdjacency("./karate.csv")
	for n := 0; n < b.N; n++ {
		L := GetLaplacian(A)
		E := GetSmallestEigenVectors(L, 2)
		_, _ = KMeans(E, 2, 10)
	}
}
