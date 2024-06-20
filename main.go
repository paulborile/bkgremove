package main

import (
	"flag"
	"fmt"
	"image"

	// "image/color"

	"gocv.io/x/gocv"
)

func main() {
	// Load the image
	// use command line argument to pass image file
	inputFile := flag.String("i", "input.jpg", "image to remove background from")
	outputFile := flag.String("o", "output.jpg", "image to save the result to")
	flag.Parse()

	img := gocv.IMRead(*inputFile, gocv.IMReadColor)
	if img.Empty() {
		fmt.Println("Error reading image from file")
		return
	}
	defer img.Close()

	// Initialize the mask
	mask := gocv.NewMatWithSize(img.Rows(), img.Cols(), gocv.MatTypeCV8UC1)
	defer mask.Close()

	// Define the rectangle for GrabCut
	rect := image.Rect(10, 10, img.Cols()-10, img.Rows()-10)

	// Initialize background and foreground models
	bgdModel := gocv.NewMat()
	defer bgdModel.Close()
	fgdModel := gocv.NewMat()
	defer fgdModel.Close()

	// Apply GrabCut algorithm
	gocv.GrabCut(img, &mask, rect, &bgdModel, &fgdModel, 5, gocv.GCInitWithRect)

	// Create a mask where we mark sure foreground and possible foreground
	fgMask := gocv.NewMatWithSize(img.Rows(), img.Cols(), gocv.MatTypeCV8UC1)
	defer fgMask.Close()

	for y := 0; y < mask.Rows(); y++ {
		for x := 0; x < mask.Cols(); x++ {
			if mask.GetUCharAt(y, x) == 1 || mask.GetUCharAt(y, x) == 3 {
				fgMask.SetUCharAt(y, x, 255)
			} else {
				fgMask.SetUCharAt(y, x, 0)
			}
		}
	}

	// Create a new image with a white background
	whiteBackground := gocv.NewMatWithSize(img.Rows(), img.Cols(), gocv.MatTypeCV8UC3)
	defer whiteBackground.Close()
	whiteColor := gocv.NewScalar(255, 255, 255, 0)
	whiteBackground.SetTo(whiteColor)

	// Copy the foreground to the new image with white background
	img.CopyToWithMask(&whiteBackground, fgMask)

	// Save the result
	if ok := gocv.IMWrite(*outputFile, whiteBackground); !ok {
		fmt.Println("Error saving output image")
	} else {
		fmt.Println("Background removed successfully")
	}
}
