// Package main provides an example on how to run yolov5 for a given image.
package main

import (
	"flag"
	"os"
	"path"

	log "github.com/sirupsen/logrus"
	"gocv.io/x