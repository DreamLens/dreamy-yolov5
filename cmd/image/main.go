// Package main provides an example on how to run yolov5 for a given image.
package main

import (
	"flag"
	"os"
	"path"

	log "github.com/sirupsen/logrus"
	"gocv.io/x/gocv"

	"github.com/wimspaargaren/yolov5"
)

var (
	yolov5Model   = path.Join(os.Getenv("GOPATH"), "src/github.com/wimspaargaren/yolov5/data/yolov5/yolov5s.onnx")
	cocoNamesPath = path.Join(os.Getenv("GOPATH"), "src/github.com/wimspaargaren/yolov5/data/yolov5/coco.names")
)

func main() {
	imagePath := flag.String("i", path.Join(os.Getenv("GOPATH"), "src/github.com/wimspaargaren/yolov5/data/example_images/street.jpg"), "specify the image path")
	flag.Parse()

	yolonet, err := yolov5.NewNet(yolov5Model, cocoNamesPath)
	if err != nil {
		log.WithErro