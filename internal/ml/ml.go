// Package ml is used as interface on how a neural network should behave
package ml

import "gocv.io/x/gocv"

// NeuralNet is the interface representing the neural network
// used for calculating the object detections
type NeuralNet interface {
	SetPreferableBackend(backend gocv.NetBa