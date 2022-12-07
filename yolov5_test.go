
package yolov5

import (
	"fmt"
	"image"
	"os"
	"path"
	"testing"

	"github.com/golang/mock/gomock"
	log "github.com/sirupsen/logrus"
	"github.com/stretchr/testify/suite"
	"gocv.io/x/gocv"

	"github.com/wimspaargaren/yolov5/internal/ml"
	"github.com/wimspaargaren/yolov5/internal/ml/mocks"
)

type YoloTestSuite struct {
	suite.Suite
}

func TestYoloTestSuite(t *testing.T) {
	suite.Run(t, new(YoloTestSuite))
}

func (s *YoloTestSuite) TestCorrectImplementation() {
	var _ Net = &yoloNet{}
}

func (s *YoloTestSuite) TestNewDefaultNetCorrectCreation() {
	net, err := NewNet("data/yolov5/yolov5s.onnx", "data/yolov5/coco.names")
	s.Require().NoError(err)
	yoloNet := net.(*yoloNet)

	s.NotNil(yoloNet.net)
	s.Equal(81, len(yoloNet.cocoNames))
	s.Equal(DefaultInputWidth, yoloNet.DefaultInputWidth)
	s.Equal(DefaultInputHeight, yoloNet.DefaultInputHeight)
	s.Equal(DefaultConfThreshold, yoloNet.confidenceThreshold)
	s.Equal(DefaultNMSThreshold, yoloNet.DefaultNMSThreshold)

	s.NoError(yoloNet.Close())
}

func (s *YoloTestSuite) TestNewCustomConfig_MissingNewNetFunc_CorrectCreation() {
	net, err := NewNetWithConfig("data/yolov5/yolov5s.onnx", "data/yolov5/coco.names", Config{})
	s.Require().NoError(err)
	yoloNet := net.(*yoloNet)

	s.NotNil(yoloNet.net)
	s.Equal(81, len(yoloNet.cocoNames))
	s.Equal(DefaultInputWidth, yoloNet.DefaultInputWidth)
	s.Equal(DefaultInputHeight, yoloNet.DefaultInputHeight)
	s.Equal(float32(0), yoloNet.confidenceThreshold)
	s.Equal(float32(0), yoloNet.DefaultNMSThreshold)

	s.NoError(yoloNet.Close())
}

func (s *YoloTestSuite) TestUnableTocCreateNewNet() {
	tests := []struct {
		Name               string
		ModelPath          string
		CocoNamePath       string
		Config             Config
		Error              error
		SetupNeuralNetMock func() *mocks.MockNeuralNet
	}{
		{
			Name:         "Non existent weights path",
			ModelPath:    "data/yolov5/notexistent",
			CocoNamePath: "data/yolov5/coco.names",
			Error:        fmt.Errorf("path to net model not found"),
		},
		{
			Name:         "Non existent coco names path",
			ModelPath:    "data/yolov5/yolov5s.onnx",
			CocoNamePath: "data/yolov5/notexistent",
		},
		{
			Name:         "Unable to set preferable backend",
			ModelPath:    "data/yolov5/yolov5s.onnx",
			CocoNamePath: "data/yolov5/coco.names",
			SetupNeuralNetMock: func() *mocks.MockNeuralNet {
				controller := gomock.NewController(s.T())
				neuralNetMock := mocks.NewMockNeuralNet(controller)
				neuralNetMock.EXPECT().SetPreferableBackend(gomock.Any()).Return(fmt.Errorf("very broken")).Times(1)
				return neuralNetMock
			},
			Error: fmt.Errorf("very broken"),
		},
		{
			Name:         "Unable to set preferable target type",
			ModelPath:    "data/yolov5/yolov5s.onnx",
			CocoNamePath: "data/yolov5/coco.names",
			SetupNeuralNetMock: func() *mocks.MockNeuralNet {
				controller := gomock.NewController(s.T())
				neuralNetMock := mocks.NewMockNeuralNet(controller)
				neuralNetMock.EXPECT().SetPreferableBackend(gomock.Any()).Return(nil).Times(1)
				neuralNetMock.EXPECT().SetPreferableTarget(gomock.Any()).Return(fmt.Errorf("very broken")).Times(1)
				return neuralNetMock
			},
			Error: fmt.Errorf("very broken"),
		},
	}

	for _, test := range tests {
		s.Run(test.Name, func() {
			test.Config.NewNet = func(string) ml.NeuralNet {
				return test.SetupNeuralNetMock()
			}
			_, err := NewNetWithConfig(test.ModelPath, test.CocoNamePath, test.Config)
			s.Error(err)
			if test.Error != nil {
				s.Equal(test.Error, err)
			}
		})
	}