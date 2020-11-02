#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

int main() {
    string video_path = "./out1_1.avi";

    VideoCapture cap;
    cap.open(video_path);

    if (!cap.isOpened()) {
        cerr << "Unable to load file." << endl;
        return -1;
    }

    int fps = cap.get(CAP_PROP_FPS);
    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    namedWindow("Left", WINDOW_NORMAL);
    namedWindow("Right", WINDOW_NORMAL);
    moveWindow("Left", 0, 0);
    moveWindow("Right", 640, 0);

    Mat frame, frame_l, frame_r;
    vector<Point2f> trace_l;
    vector<Point2f> trace_r;
    while (true) {
        // get one frame
        cap >> frame;
        if (frame.empty() == true) break;

        // convert to HSV
        Mat hsv_frame;
        cvtColor(frame, hsv_frame, COLOR_RGB2HSV);

        // search the ball
        Mat diff_frame = Mat::ones(height, width, CV_8U);
        for (int y = 0; y < hsv_frame.rows; y++) {
            for (int x = 0; x < hsv_frame.cols; x++) {
                int hue, sat, val;
                hue = hsv_frame.at<Vec3b>(y, x)[0];
                sat = hsv_frame.at<Vec3b>(y, x)[1];
                val = hsv_frame.at<Vec3b>(y, x)[2];
                if ((hue >= 70 && hue <= 120) && (sat >= 90 && sat <= 255) &&
                    (val >= 150 && val <= 255)) {
                    diff_frame.at<uchar>(y, x) = 255;
                } else {
                    diff_frame.at<uchar>(y, x) = 0;
                }
            }
        }
        dilate(diff_frame, diff_frame, Mat(), Point(-1, -1), 1);

        // split image
        Mat diff_frame_l, diff_frame_r;
        diff_frame_l =
            diff_frame(Rect(0, 0, diff_frame.cols / 2, diff_frame.rows));
        diff_frame_r = diff_frame(
            Rect(diff_frame.cols / 2, 0, diff_frame.cols / 2, diff_frame.rows));
        frame_l = frame(Rect(0, 0, frame.cols / 2, frame.rows));
        frame_r = frame(Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows));

        // calculate the coordinate of the ball
        Moments mu_l = moments(diff_frame_l, false);
        Point2f mc_l = Point2f(mu_l.m10 / mu_l.m00, mu_l.m01 / mu_l.m00);
        trace_l.push_back(mc_l);
        for (int i = 0; i < trace_l.size(); i++) {
            circle(frame_l, trace_l[i], 2, Scalar(0, 0, 255), -1);
        }
        Moments mu_r = moments(diff_frame_r, false);
        Point2f mc_r = Point2f(mu_r.m10 / mu_r.m00, mu_r.m01 / mu_r.m00);
        trace_r.push_back(mc_r);
        for (int i = 0; i < trace_r.size(); i++) {
            circle(frame_r, trace_r[i], 2, Scalar(0, 0, 255), -1);
        }

        resizeWindow("Left", 640, 360);
        resizeWindow("Right", 640, 360);
        imshow("Left", frame_l);
        imshow("Right", frame_r);

        waitKey(1000 / fps);
    }

    return 0;
}
