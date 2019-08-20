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

using namespace std;
using namespace cv;

bool compareContourAreas(const vector<Point> &contour1, const vector<Point> &contour2) {
    double i = fabs(contourArea(Mat(contour1)));
    double j = fabs(contourArea(Mat(contour2)));
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
    orderPoints(move(pts), ordered_pts);

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
    Mat imageGrayed;
    Mat imageOpen, imageClosed, imageBlurred;
    if (src.channels() == 3) {
        cvtColor(src, imageGrayed, CV_BGR2GRAY);
    } else {
        imageGrayed = src;
    }
    imshow("Image grayed", imageGrayed);
    Mat structuringElmt = getStructuringElement(MORPH_ELLIPSE, Size(4, 4));
    morphologyEx(imageGrayed, imageOpen, MORPH_OPEN, structuringElmt);
    morphologyEx(imageOpen, imageClosed, MORPH_CLOSE, structuringElmt);
//    imshow("Image imageOpen", imageOpen);
//    imshow("Image imageClosed", imageClosed);

    GaussianBlur(imageClosed, imageBlurred, Size(1, 1), 0);
    Canny(imageBlurred, dst, 75, 200);
    imshow("Image imageBlurred", imageBlurred);
    imshow("Image dst", dst);


}

Mat linesDetection(const Mat &src) {
    // Transform source image to gray if it is not
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, CV_BGR2GRAY);
    } else {
        gray = src;
    }
    // Show gray image
//    imshow("gray", gray);
    // Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    Mat bw;
    adaptiveThreshold(~gray, bw, 255, CV_ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 9, -2);

    // Show binary image
//    imshow("binary", bw);
    // Create the images that will use to extract the horizontal and vertical lines
    Mat horizontal = bw.clone();
    Mat vertical = bw.clone();
    // Specify size on horizontal axis
    int horizontalsize = horizontal.cols / 50;
    // Create structure element for extracting horizontal lines through morphology operations
    Mat horizontalStructure = getStructuringElement(MORPH_RECT, Size(horizontalsize, 1));
    // Apply morphology operations
    erode(horizontal, horizontal, horizontalStructure, Point(-1, -1));
    dilate(horizontal, horizontal, horizontalStructure, Point(-1, -1));

    // Connect nearest broken line
    Mat horizontalStructureConnect = getStructuringElement(MORPH_RECT, Size(5, 1));
    dilate(horizontal, horizontal, horizontalStructureConnect, Point(-1, -1));
    // Remove small line
    Mat horizontalStructureShort = getStructuringElement(MORPH_RECT, Size(5, 1));
    erode(horizontal, horizontal, horizontalStructureShort, Point(-1, -1));
    // Connect nearest broken line
    dilate(horizontal, horizontal, horizontalStructureConnect, Point(-1, -1));
    dilate(horizontal, horizontal, horizontalStructureConnect, Point(-1, -1));
    dilate(horizontal, horizontal, horizontalStructureConnect, Point(-1, -1));
    // Show extracted horizontal lines
//    imshow("horizontal", horizontal);


    // Specify size on vertical axis
    int verticalsize = vertical.rows / 30;
    // Create structure element for extracting vertical lines through morphology operations
    Mat verticalStructure = getStructuringElement(MORPH_RECT, Size(1, verticalsize));
    // Apply morphology operations
    erode(vertical, vertical, verticalStructure, Point(-1, -1));
    dilate(vertical, vertical, verticalStructure, Point(-1, -1));

    // Connect nearest broken line
    Mat verticalStructureConnect = getStructuringElement(MORPH_RECT, Size(1, 5));
    dilate(vertical, vertical, verticalStructureConnect, Point(-1, -1));
    // Remove small line
    Mat verticalStructureShort = getStructuringElement(MORPH_RECT, Size(1, 5));
    erode(vertical, vertical, verticalStructureShort, Point(-1, -1));
    // Connect nearest broken line
    dilate(vertical, vertical, verticalStructureConnect, Point(-1, -1));
    dilate(vertical, vertical, verticalStructureConnect, Point(-1, -1));
    dilate(vertical, vertical, verticalStructureConnect, Point(-1, -1));

    // Show extracted vertical lines
//    imshow("vertical", vertical);

    Mat mask = vertical + horizontal;
//    imshow("mask", mask);
    vertical = mask;


    // Inverse vertical image
//    bitwise_not(vertical, vertical);
//    imshow("vertical_bit", vertical);
    // Extract edges and smooth image according to the logic
    // 1. extract edges
    // 2. dilate(edges)
    // 3. src.copyTo(smooth)
    // 4. blur smooth img
    // 5. smooth.copyTo(src, edges)
    // Step 1
    Mat edges;
    adaptiveThreshold(vertical, edges, 255, CV_ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3, -2);
//    imshow("edges", edges);
    // Step 2
    Mat kernel = Mat::ones(2, 2, CV_8UC1);
    dilate(edges, edges, kernel);
//    imshow("dilate", edges);
    // Step 3
    Mat smooth;
    vertical.copyTo(smooth);
    // Step 4
    blur(smooth, smooth, Size(2, 2));
    // Step 5
    smooth.copyTo(vertical, edges);
    // Show final result
//    imshow("smooth", vertical);

    /**
     * Use HoughLinesP to get all posible lines and find 4 limit points min-top-left, max-bottom-right
     * Depend on 4 limit points, get center of image, combine with valid lines to fine correct left, top, right, bottom.
     */
    vector<Vec4i> lines;
    Mat color = src.clone();
    Mat out = Mat::zeros(mask.size(), mask.type());
    HoughLinesP(vertical, lines, 1, CV_PI / 180, 80, 60, 10);
    int xMin = mask.cols;
    int yMin = mask.rows;
    int xMax = 0;
    int yMax = 0;
    for (size_t i = 0; i < lines.size(); i++) {
        Vec4i l = lines[i];
        Point a(l[0], l[1]);
        Point b(l[2], l[3]);
        double res = cv::norm(a - b);
        if (res > mask.cols / 2.0) {
            xMin = cv::min(xMin, a.x);
            xMax = cv::max(xMax, b.x);

            line(color, a, b, Scalar(255));
            circle(color, a, 5, Scalar(0, 0, 255) /*Red*/, 2);
            circle(color, b, 5, Scalar(0, 255, 255) /*Yellow*/, 2);
        }

        if (res > mask.rows / 2.0) {
            yMin = cv::min(yMin, a.y);
            yMax = cv::max(yMax, b.y);

            line(color, a, b, Scalar(255));
            circle(color, a, 5, Scalar(0, 0, 255) /*Red*/, 2);
            circle(color, b, 5, Scalar(0, 255, 255) /*Yellow*/, 2);
        }
    }

    int middleX = (xMax + xMin) / 2;
    int middleY = (yMax + yMin) / 2;
    Point leftTop(middleX, middleY);
    Point topRight(middleX, middleY);
    Point rightBottom(middleX, middleY);
    Point bottomLeft(middleX, middleY);
    for (size_t i = 0; i < lines.size(); i++) {
        Vec4i l = lines[i];
        Point a(l[0], l[1]);
        Point b(l[2], l[3]);
        // Find left, top
        if (a.x < middleX && a.y < middleY) {
            leftTop.x = cv::min(a.x, leftTop.x);
            leftTop.y = cv::min(a.y, leftTop.y);
        }
        if (b.x < middleX && b.y < middleY) {
            leftTop.x = cv::min(b.x, leftTop.x);
            leftTop.y = cv::min(b.y, leftTop.y);
        }
        // Find top, right
        if (a.x > middleX && a.y < middleY) {
            topRight.x = cv::max(a.x, topRight.x);
            topRight.y = cv::min(a.y, topRight.y);
        }
        if (b.x > middleX && b.y < middleY) {
            topRight.x = cv::max(b.x, topRight.x);
            topRight.y = cv::min(b.y, topRight.y);
        }
        // Find right, bottom
        if (a.x > middleX && a.y > middleY) {
            rightBottom.x = cv::max(a.x, rightBottom.x);
            rightBottom.y = cv::max(a.y, rightBottom.y);
        }
        if (b.x > middleX && b.y > middleY) {
            rightBottom.x = cv::max(b.x, rightBottom.x);
            rightBottom.y = cv::max(b.y, rightBottom.y);
        }
        // Find bottom, left
        if (a.x < middleX && a.y > middleY) {
            bottomLeft.x = cv::min(a.x, bottomLeft.x);
            bottomLeft.y = cv::max(a.y, bottomLeft.y);
        }
        if (b.x < middleX && b.y > middleY) {
            bottomLeft.x = cv::min(b.x, bottomLeft.x);
            bottomLeft.y = cv::max(b.y, bottomLeft.y);
        }
    }
//    imshow("color", color);

    line(out, leftTop, topRight, Scalar(255), 3, 8);
    line(out, topRight, Point(xMax, yMax), Scalar(255), 3, 8);
    line(out, Point(xMax, yMax), Point(xMin, yMax), Scalar(255), 3, 8);
    line(out, Point(xMin, yMax), leftTop, Scalar(255), 3, 8);

//    imshow("HoughLinesP", out);

    return out;
}

void imageSource(const string &path) {
    Mat image = imread(path);
    if (image.empty()) {
        printf("Cannot read image file");
        return;
    }

    double ratio = image.rows / 500.0;
    Mat orig = image.clone();
    resizeToHeight(image, image, 500);

    image = linesDetection(image);


    Mat edged = image;
//    preProcess(image, edged);

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
    Mat tmp = orig.clone();
    resizeToHeight(tmp, tmp, 500);
    for (i = 0; i < approx.size(); i++) {
        drawContours(tmp, approx, i, Scalar(255, 0, 255), 2);
        if (approx[i].size() == 4) {
            break;
        }
    }
    imshow(path + "_contours", tmp);
    int j;
    if (i < approx.size()) {
        drawContours(image, approx, i, Scalar(0, 255, 0), 2);
        for (j = 0; j < approx[i].size(); j++) {
            approx[i][j] *= ratio;
        }

        // apply the four point transform to obtain a top-down
        // view of the original image
        Mat warped;
        fourPointTransform(orig, warped, approx[i]);
        // convert the warped image to grayscale, then threshold it
        // to give it that 'black and white' paper effect
//        cvtColor(warped, warped, CV_BGR2GRAY, 1);
//        threshold(warped, warped, 127, 255, CV_THRESH_BINARY);
//        adaptiveThreshold(warped, warped, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 9, 15);
//        GaussianBlur(warped, warped, Size(3, 3), 0);

        resizeToHeight(warped, warped, 500);
//        imshow(path + "_res", warped);
    }

}

int main(int argc, char **argv) {

    char buffer[32];
    int i = 1;
    do {
        sprintf_s(buffer, "samples/business_cards/%03d.jpg", i);
        cout << i << " ";
        imageSource(buffer);
        i++;
    } while (i <= 10);
    waitKey(0);
    return 0;
}

