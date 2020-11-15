#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

using namespace std;
using namespace cv;

Mat projMatr1 = (Mat_<double>(3, 4) << 1.10082917e+03, 0.00000000e+00,
                 6.24244764e+02, 0.00000000e+00, 0.00000000e+00, 1.09806264e+03,
                 3.33867362e+02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                 1.00000000e+00, 0.00000000e+00);
Mat projMatr2 =
    (Mat_<double>(3, 4) << 1.09206116e+03, 7.00301809e+00, 6.39421445e+02,
     -8.03761654e+03, 1.75020071e+00, 1.10479669e+03, 3.10853998e+02,
     3.00618340e+02, -1.38470270e-02, 2.09410228e-02, 9.99684811e-01,
     2.32549042e-01);

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

        // triangulation
        if (!isnan(trace_l.back().x) && !isnan(trace_l.back().y) &&
            !isnan(trace_r.back().x) && !isnan(trace_r.back().y)) {
            Mat point4D, point3D;
            vector<Point2d> projPoints1{trace_l.back()};
            vector<Point2d> projPoints2{trace_r.back()};
            triangulatePoints(projMatr1, projMatr2, projPoints1, projPoints2,
                              point4D);
            convertPointsFromHomogeneous(point4D.reshape(4), point3D);
            point3D *= 0.035;
            cout << fixed << setprecision(4) << showpos
                 << "x:" << point3D.at<double>(0, 0)
                 << " y:" << point3D.at<double>(0, 1)
                 << " z:" << point3D.at<double>(0, 2) << endl;
        }

        resizeWindow("Left", 640, 360);
        resizeWindow("Right", 640, 360);
        imshow("Left", frame_l);
        imshow("Right", frame_r);

        waitKey(1000 / fps);
    }

    return 0;
}
