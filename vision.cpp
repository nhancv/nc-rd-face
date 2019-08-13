#include <utility>

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

bool compareContourAreas(const std::vector<cv::Point> &contour1, const std::vector<cv::Point> &contour2) {
    double i = fabs(contourArea(cv::Mat(contour1)));
    double j = fabs(contourArea(cv::Mat(contour2)));
    return (i > j);
}

bool compareXCords(const Point &p1, const Point &p2) {
    return (p1.x < p2.x);
}

bool compareYCords(const Point &p1, const Point &p2) {
    return (p1.y < p2.y);
}

bool compareDistance(const pair<Point, Point> &p1, const pair<Point, Point> &p2) {
    return (norm(p1.first - p1.second) < norm(p2.first - p2.second));
}

float _distance(const Point &p1, const Point &p2) {
    return (float) sqrt(((p1.x - p2.x) * (p1.x - p2.x)) +
                        ((p1.y - p2.y) * (p1.y - p2.y)));
}

void resizeToHeight(const Mat &src, Mat &dst, int height) {
    Size s = Size((int) (src.cols * (height / float(src.rows))), height);
    resize(src, dst, s, CV_INTER_AREA);
}

void orderPoints(vector<Point> inpts, vector<Point> &ordered) {
    sort(inpts.begin(), inpts.end(), compareXCords);
    vector<Point> lm(inpts.begin(), inpts.begin() + 2);
    vector<Point> rm(inpts.end() - 2, inpts.end());

    sort(lm.begin(), lm.end(), compareYCords);
    Point tl(lm[0]);
    Point bl(lm[1]);
    vector<pair<Point, Point> > tmp;
    tmp.reserve(rm.size());
    for (auto &i : rm) {
        tmp.emplace_back(make_pair(tl, i));
    }

    sort(tmp.begin(), tmp.end(), compareDistance);
    Point tr(tmp[0].second);
    Point br(tmp[1].second);

    ordered.push_back(tl);
    ordered.push_back(tr);
    ordered.push_back(br);
    ordered.push_back(bl);
}

void fourPointTransform(const Mat &src, Mat &dst, vector<Point> pts) {
    vector<Point> ordered_pts;
    orderPoints(std::move(pts), ordered_pts);

    float wa = _distance(ordered_pts[2], ordered_pts[3]);
    float wb = _distance(ordered_pts[1], ordered_pts[0]);
    float mw = max(wa, wb);

    float ha = _distance(ordered_pts[1], ordered_pts[2]);
    float hb = _distance(ordered_pts[0], ordered_pts[3]);
    float mh = max(ha, hb);

    Point2f src_[] =
            {
                    Point2f((float) ordered_pts[0].x, (float) ordered_pts[0].y),
                    Point2f((float) ordered_pts[1].x, (float) ordered_pts[1].y),
                    Point2f((float) ordered_pts[2].x, (float) ordered_pts[2].y),
                    Point2f((float) ordered_pts[3].x, (float) ordered_pts[3].y),
            };
    Point2f dst_[] =
            {
                    Point2f(0, 0),
                    Point2f(mw - 1, 0),
                    Point2f(mw - 1, mh - 1),
                    Point2f(0, mh - 1)
            };
    Mat m = getPerspectiveTransform(src_, dst_);
    warpPerspective(src, dst, m, Size((int) mw, (int) mh));
}

void preProcess(const Mat &src, Mat &dst) {
    cv::Mat imageGrayed;
    cv::Mat imageOpen, imageClosed, imageBlurred;

    cvtColor(src, imageGrayed, CV_BGR2GRAY);
    imshow("Image grayed", imageGrayed);

    cv::Mat structuringElmt = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(4, 4));
    morphologyEx(imageGrayed, imageOpen, cv::MORPH_OPEN, structuringElmt);
    morphologyEx(imageOpen, imageClosed, cv::MORPH_CLOSE, structuringElmt);
    imshow("Image imageOpen", imageOpen);
    imshow("Image imageClosed", imageClosed);

    GaussianBlur(imageClosed, imageBlurred, Size(1, 1), 0);
    imshow("Image imageBlurred", imageBlurred);
    Canny(imageBlurred, dst, 75, 200);
    imshow("Image dst", dst);


}

void imageSource() {
    Mat image = imread("samples/business_cards/002.jpg");
    if (image.empty()) {
        printf("Cannot read image file");
        return;
    }

    double ratio = image.rows / 500.0;
    Mat orig = image.clone();
    resizeToHeight(image, image, 500);

    Mat gray, edged, warped;
    preProcess(image, edged);

    // find the contours in the edged image, keeping only the
    // largest ones, and initialize the screen contour
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    vector<vector<Point> > approx;
    findContours(edged, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
    approx.resize(contours.size());
    // loop over the contours
    int i;
    for (i = 0; i < contours.size(); i++) {
        // approximate the contour
        double peri = arcLength(contours[i], true);
        // if our approximated contour has four points, then we
        //  can assume that we have found our screen
        approxPolyDP(contours[i], approx[i], 0.02 * peri, true);
    }
    sort(approx.begin(), approx.end(), compareContourAreas);

    // show the contour (outline) of the piece of paper
    for (i = 0; i < approx.size(); i++) {
        drawContours(image, approx, i, Scalar(255, 255, 0), 2);
        if (approx[i].size() == 4) {
            break;
        }
    }
    imshow("drawContours", image);
    int j;
    if (i < approx.size()) {
        drawContours(image, approx, i, Scalar(0, 255, 0), 2);
        for (j = 0; j < approx[i].size(); j++) {
            approx[i][j] *= ratio;
        }

        // apply the four point transform to obtain a top-down
        // view of the original image

        fourPointTransform(orig, warped, approx[i]);
        // convert the warped image to grayscale, then threshold it
        // to give it that 'black and white' paper effect
        cvtColor(warped, warped, CV_BGR2GRAY, 1);
        adaptiveThreshold(warped, warped, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 9, 15);
        GaussianBlur(warped, warped, Size(3, 3), 0);

        resizeToHeight(warped, warped, 500);
        imshow("imageSource", warped);
    }


}

int main(int argc, char **argv) {

    imageSource();

    waitKey(0);
    return 0;
}

