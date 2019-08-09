#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/tracking/tracker.hpp>
#include <opencv2/core/ocl.hpp>

//#include <dlib/opencv.h>
//#include <dlib/image_io.h>
//#include <dlib/image_processing.h>
//#include <dlib/image_processing/frontal_face_detector.h>

using namespace std;
//using namespace dlib;
using namespace cv;
int thresh = 100;
int max_thresh = 255;
RNG rng(12345);

void imageSource() {
//    namedWindow("ImageSource");

    Mat img = imread("samples/card4.jpg");
    cv::resize(img, img, cv::Size((int) (img.cols * 0.75), (int) (img.rows * 0.75)), 0, 0, CV_INTER_LINEAR);
//    imshow("imageSource", img);

    // PROCESSING

    // STEP 1:
    // Greyscale
    cv::Mat greyMat;
    cv::cvtColor(img, greyMat, cv::COLOR_BGR2GRAY);
//    imshow("Gray Image", greyMat);
    // GaussianBlur
    cv::Mat gaussianBlurMat;
    cv::GaussianBlur(greyMat, gaussianBlurMat, Size(5, 5), 0);
//    imshow("GaussianBlur Image", gaussianBlurMat);
    // Canny
    cv::Mat cannyMat;
    cv::Canny(gaussianBlurMat, cannyMat, 75, 200);
//    imshow("Canny Image", cannyMat);

    // STEP 2:
    // Finding Contours
    // Find the contours in the edged image, keeping only the
    // largest ones, and initialize the screen contour
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(cannyMat, contours, hierarchy,
                 CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
    // iterate through all the top-level contours,
    // draw each connected component with its own random color
    Mat drawing = Mat::zeros(cannyMat.size(), CV_8UC3);
    for (int i = 0; i < contours.size(); i++) {
        Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
    }
    imshow("Components", drawing);


}

int main(int argc, char **argv) {

    imageSource();

    waitKey(0);
    return 0;
}

