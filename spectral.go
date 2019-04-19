package main

import (
	"encoding/csv"
	"github.com/bugra/kmeans"
	"gonum.org/v1/gonum/mat"
	"os"
	"strconv"
)

func LoadAdjacency(path string) (mat.Symmetric, error) {
	file, err := os.Open(path)
	if err == nil {
		edges, err := csv.NewReader(file).ReadAll()
		return CreateSymmetricMatrix(edges), err
	}
	return nil, err
}

func CreateSymmetricMatrix(edges [][]string) mat.Symmetric {
	A := mat.NewSymDense(2, nil)
	for _, edge := range edges {
		start, _ := strconv.Atoi(edge[0])
		end, _ := strconv.Atoi(edge[1])
		A = growIfNecessary(A, start, end)
		A.SetSym(start-1, end-1, 1.)
	}
	return A
}

func growIfNecessary(A *mat.SymDense, start int, end int) *mat.SymDense {
	if n := A.Symmetric(); start > n || end > n {
		if start > end {
			A = A.GrowSym(start - n).(*mat.SymDense)
		} else {
			A = A.GrowSym(end - n).(*mat.SymDense)
		}
	}
	return A
}

func GetDegrees(A mat.Symmetric) mat.Diagonal {
	n := A.Symmetric()
	diag := make([]float64, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			diag[i] += A.At(i, j)
		}
	}
	return mat.NewDiagDense(n, diag)
}

func GetLaplacian(A mat.Symmetric) mat.Symmetric {
	n := A.Symmetric()
	L := mat.NewSymDense(n, nil)
	L.CopySym(A)
	D := GetDegrees(A)
	for i := 0; i < n; i++ {
		l := L.At(i, i) - D.At(i, i)
		L.SetSym(i, i, l)
	}
	return L
}

func GetSmallestEigenVectors(L mat.Symmetric, k int) mat.Matrix {
	n := L.Symmetric()
	decomposition := mat.EigenSym{}
	decomposition.Factorize(L, true)
	E := mat.NewDense(n, n, nil)
	decomposition.VectorsTo(E)
	return E.Slice(0, n, n-k-1, n-1)
}

func KMeans(E mat.Matrix, k int, iter int) ([]int, error) {
	rows, _ := E.Dims()
	rawData := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		rawData[i] = mat.Row(nil, i, E)
	}
	return kmeans.Kmeans(rawData, k, kmeans.EuclideanDistance, iter)
}
