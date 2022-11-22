
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

// Net the yolov5 net.
type Net interface {
	Close() error
	GetDetections(gocv.Mat) ([]ObjectDetection, error)
	GetDetectionsWithFilter(gocv.Mat, map[string]bool) ([]ObjectDetection, error)
}

// yoloNet the net implementation.
type yoloNet struct {
	net       ml.NeuralNet
	cocoNames []string

	DefaultInputWidth   int
	DefaultInputHeight  int
	confidenceThreshold float32
	DefaultNMSThreshold float32
}

// NewNet creates new yolo net for given weight path, config and coconames list.
func NewNet(modelPath, cocoNamePath string) (Net, error) {
	return NewNetWithConfig(modelPath, cocoNamePath, DefaultConfig())
}

// NewNetWithConfig creates new yolo net with given config.
func NewNetWithConfig(modelPath, cocoNamePath string, config Config) (Net, error) {
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("path to net model not found")
	}

	cocoNames, err := getCocoNames(cocoNamePath)
	if err != nil {
		return nil, err
	}

	config.validate()

	net := config.NewNet(modelPath)

	err = setNetTargetTypes(net, config)
	if err != nil {
		return nil, err
	}

	return &yoloNet{
		net:                 net,
		cocoNames:           cocoNames,
		DefaultInputWidth:   config.InputWidth,
		DefaultInputHeight:  config.InputHeight,
		confidenceThreshold: config.ConfidenceThreshold,
		DefaultNMSThreshold: config.NMSThreshold,
	}, nil
}

// initializeNet default method for creating neural network, leveraging gocv.
func initializeNet(modelPath string) ml.NeuralNet {
	net := gocv.ReadNetFromONNX(modelPath)
	return &net
}

func setNetTargetTypes(net ml.NeuralNet, config Config) error {
	err := net.SetPreferableBackend(config.NetBackendType)
	if err != nil {
		return err
	}

	err = net.SetPreferableTarget(config.NetTargetType)
	if err != nil {
		return err
	}
	return nil
}

// Close closes the net.
func (y *yoloNet) Close() error {
	return y.net.Close()
}

// GetDetections retrieve predicted detections from given matrix.
func (y *yoloNet) GetDetections(frame gocv.Mat) ([]ObjectDetection, error) {
	return y.GetDetectionsWithFilter(frame, make(map[string]bool))
}

// GetDetectionsWithFilter allows you to detect objects, but filter out a given list of coco name ids.
func (y *yoloNet) GetDetectionsWithFilter(frame gocv.Mat, classIDsFilter map[string]bool) ([]ObjectDetection, error) {
	blob := gocv.BlobFromImage(frame, 1.0/255.0, image.Pt(y.DefaultInputWidth, y.DefaultInputHeight), gocv.NewScalar(0, 0, 0, 0), true, false)
	// nolint: errcheck
	defer blob.Close()
	y.net.SetInput(blob, "")
	layerIDs := y.net.GetUnconnectedOutLayers()
	fl := []string{}

	for _, id := range layerIDs {
		layer := y.net.GetLayer(id)
		fl = append(fl, layer.GetName())
	}
	outputs := y.net.ForwardLayers(fl)
	for i := 0; i < len(outputs); i++ {
		// nolint: errcheck
		defer outputs[i].Close()
	}

	detections, err := y.processOutputs(frame, outputs, classIDsFilter)
	if err != nil {
		return nil, err
	}

	return detections, nil
}

// processOutputs process detected rows in the outputs.
func (y *yoloNet) processOutputs(frame gocv.Mat, outputs []gocv.Mat, filter map[string]bool) ([]ObjectDetection, error) {
	// FIXME add filter functionality
	_ = filter

	detections := []ObjectDetection{}
	bboxes := []image.Rectangle{}