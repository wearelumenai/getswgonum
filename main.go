package main

import "log"

func main() {
	const nbCommunities = 2
	A, _ := LoadAdjacency("./karate.csv")
	L := GetLaplacian(A)
	E := GetSmallestEigenVectors(L, nbCommunities)
	labels, _ := KMeans(E, nbCommunities, 10)
	log.Print(labels)
}
