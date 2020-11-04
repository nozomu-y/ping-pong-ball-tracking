#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

using namespace std;
using namespace cv;

Mat frame_l, frame_r;
vector<Point2f> trace_l;
vector<Point2f> trace_r;
int height, width;

void trace_ball(Mat frame_half, vector<Point2f>& trace) {
    // convert to HSV
    Mat hsv_frame;
    cvtColor(frame_half, hsv_frame, COLOR_RGB2HSV);

    // search the ball
    Mat diff_frame = Mat::ones(height, width / 2, CV_8U);
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

    // calculate the coordinate of the ball
    Moments mu = moments(diff_frame, false);
    Point2f mc = Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);
    trace.push_back(mc);
    for (int i = 0; i < trace.size(); i++) {
        circle(frame_half, trace[i], 2, Scalar(0, 0, 255), -1);
    }
}

void Thread_l() { trace_ball(frame_l, trace_l); }

void Thread_r() { trace_ball(frame_r, trace_r); }

int main() {
    string video_path = "./out1_1.avi";

    VideoCapture cap;
    cap.open(video_path);

    if (!cap.isOpened()) {
        cerr << "Unable to load file." << endl;
        return -1;
    }

    int fps = cap.get(CAP_PROP_FPS);
    width = cap.get(CAP_PROP_FRAME_WIDTH);
    height = cap.get(CAP_PROP_FRAME_HEIGHT);

    namedWindow("Left", WINDOW_NORMAL);
    namedWindow("Right", WINDOW_NORMAL);
    moveWindow("Left", 0, 0);
    moveWindow("Right", 640, 0);

    Mat frame;
    while (true) {
        // get one frame
        cap >> frame;
        if (frame.empty() == true) break;

        frame_l = frame(Rect(0, 0, frame.cols / 2, frame.rows));
        frame_r = frame(Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows));

        thread th_l(Thread_l);
        thread th_r(Thread_r);

        th_l.join();
        th_r.join();

        resizeWindow("Left", 640, 360);
        resizeWindow("Right", 640, 360);
        imshow("Left", frame_l);
        imshow("Right", frame_r);

        waitKey(1000 / fps);
    }

    return 0;
}
