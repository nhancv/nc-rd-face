//
//  main.cpp
//  facemask
//
//  Created by iNhan Cao on 4/2/19.
//  Copyright © 2019 iNhan Cao. All rights reserved.
//

#include<opencv2/opencv.hpp>
#include<iostream>
using namespace std;
using namespace cv;
int main()
{
    Mat img = imread("headPose.jpg");
    namedWindow("image", WINDOW_NORMAL);
    imshow("image", img);
    waitKey(0);
    return 0;
}