// Package main provides an example on how to run yolov5 using a camera.
package main

import (
	"os"
	"path"

	log "github.com/sirupsen/logrus"
	"gocv.io/x/gocv"

	"github.com/wimspaargaren/yolov5"
)

var (
	yolov5Model  