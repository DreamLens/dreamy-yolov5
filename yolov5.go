
// Package yolov5 provides a Go implementation of the YOLO V5 object detection system: https://pjreddie.com/darknet/yolo/.
//
// The yolov5 package leverages gocv(https://github.com/hybridgroup/gocv) for a neural net able to detect object.
//
// In order for the neural net to be able to detect objects, it needs the pre-trained network model
// consisting of a .cfg file and a .weights file. Using the Makefile provied by the library, these models
// can simply be downloaded by running 'make models'.
//
// In order to use the package, make sure you've checked the prerequisites in the README: https://github.com/wimspaargaren/yolov5#prerequisites
package yolov5

import (
	"fmt"
	"image"
	"image/color"
	"os"
	"strings"

	"gocv.io/x/gocv"

	"github.com/wimspaargaren/yolov5/internal/ml"
)

// Default constants for initialising the yolov5 net.
const (
	DefaultInputWidth  = 640
	DefaultInputHeight = 640

	DefaultConfThreshold float32 = 0.5
	DefaultNMSThreshold  float32 = 0.4
)

// Config can be used to customise the settings of the neural network used for object detection.
type Config struct {
	// InputWidth & InputHeight are used to determine the input size of the image for the network
	InputWidth  int
	InputHeight int
	// ConfidenceThreshold can be used to determine the minimum confidence before an object is considered to be "detected"
	ConfidenceThreshold float32
	// Non-maximum suppression threshold used for removing overlapping bounding boxes
	NMSThreshold float32

	// Type on which the network will be executed
	NetTargetType  gocv.NetTargetType
	NetBackendType gocv.NetBackendType

	// NewNet function can be used to inject a custom neural net
	NewNet func(modelPath string) ml.NeuralNet
}

// validate ensures that the basic fields of the config are set
func (c *Config) validate() {
	if c.NewNet == nil {
		c.NewNet = initializeNet
	}
	if c.InputWidth == 0 {
		c.InputWidth = DefaultInputWidth
	}
	if c.InputHeight == 0 {
		c.InputHeight = DefaultInputHeight
	}
}

// DefaultConfig used to create a working yolov5 net out of the box.
func DefaultConfig() Config {
	return Config{
		InputWidth:          DefaultInputWidth,
		InputHeight:         DefaultInputHeight,
		ConfidenceThreshold: DefaultConfThreshold,
		NMSThreshold:        DefaultNMSThreshold,
		NetTargetType:       gocv.NetTargetCPU,
		NetBackendType:      gocv.NetBackendDefault,
		NewNet:              initializeNet,
	}
}

// ObjectDetection represents information of an object detected by the neural net.
type ObjectDetection struct {
	ClassID     int
	ClassName   string
	BoundingBox image.Rectangle
	Confidence  float32
}
