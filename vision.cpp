#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/tracking/tracker.hpp>
#include <opencv2/core/ocl.hpp>

#include <dlib/opencv.h>
#include <dlib/image_io.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

using namespace std;
using namespace dlib;
using namespace cv;

void imageSource() {
    namedWindow("imageSource");

    Mat img = imread("samples/card4.jpg");

    imshow("imageSource", img);

}

int main(int argc, char **argv) {

    imageSource();

    waitKey(0);
    return 0;
}

