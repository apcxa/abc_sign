#!/usr/bin/python3

import jetson.inference
import jetson.utils

import argparse

#Set up our arguement parser
parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="this is the filename of the image to process")
parser.add_argument("--network", type=str, default="googlenet", help="the model to use, can change to:  googlenet, resnet-18, ect. (see --help for others)")

# Use arguement parser to get command line options:
opt = parser.parse_args()

#Load the image that we specified in options
img = jetson.utils.loadImage(opt.filename)

#Load the network for the ImageNet model:
net = jetson.inference.imageNet(opt.network)

class_idx, confidence = net.Classify(img)
class_desc = net.GetClassDesc(class_idx)

print("image is recognized as '{:s}' (class #{:d}) with {:f}% confidence".format(class_desc, class_idx, confidence * 100))