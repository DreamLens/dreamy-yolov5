// Code generated by MockGen. DO NOT EDIT.
// Source: yolov5.go

// Package mock_yolov5 is a generated GoMock package.
package mock_yolov5

import (
        reflect "reflect"

        gomock "github.com/golang/mock/gomock"
        yolov5 "github.com/wimspaargaren/yolov5"
        gocv "gocv.io/x/gocv"
)

// MockNet is a mock of Net interface.
type MockNet struct {
        ctrl     *gomock.Controller
        recorder *MockNetMockRecorder
}

// MockNetMockRecorder is the mock recorder for MockNet.
type MockNetMockRecorder struct {
        mock *MockNet
}

// NewMockNet creates a new mock instance.
func NewMockNet(ctrl *gomock.Controller) *MockNet {
        mock := &MockNet{ctrl: ctrl}
        mock.recorder = &MockNetMockRecorder{mock}
        return mock
}

// EXPECT returns an object that allows the caller to indicate expected use.
func (m *MockNet) EXPECT() *MockNetMockRecorder {
        return m.recorder
}

// Close mocks base method.
func (m *MockNet) Close() error {
        m.ctrl.T.Helper()
        ret := m.ctrl.Call(m, "Close")
        ret0, _ := ret[0].(error)
        return ret0
}

// Close indicates an expected call of Close.
func (mr *MockNetMockRecorder) Close() *gomock.Call {
        mr.mock.ctrl.T.Helper()
        return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Close", reflect.TypeOf((*MockNet)(nil).Close))
}

// GetDetections mocks base method.
func (m *MockNet) GetDetections(arg0 gocv.Mat) ([]yolov5.ObjectDetection, error) {
        m.ctrl.T.Helper()
        ret := m.ctrl.Call(m, "GetDetections", arg0)
        ret0, _ := ret[0].([]yolov5.ObjectDetection)
        ret1, _ := ret[1].(error)
        return ret0, ret1
}

// GetDetections indicates an expected call of GetDetections.
func (mr *MockNetMockRecorder) GetDetections(arg0 interface{}) *gomock.Call {
        mr.mock.ctrl.T.Helper()
        return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetDetections", reflect.TypeOf((*MockNet)(nil).GetDetections), arg0)
}

// GetDetectionsWithFilter mocks bas