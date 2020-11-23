#include <chrono>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <ostream>
#include <ratio>
#include <string>
#include <thread>
#include <vector>

cv::Mat projMatr1 =
    (cv::Mat_<double>(3, 4) << 1.10082917e+03, 0.00000000e+00, 6.24244764e+02,
     0.00000000e+00, 0.00000000e+00, 1.09806264e+03, 3.33867362e+02,
     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00,
     0.00000000e+00);
cv::Mat projMatr2 =
    (cv::Mat_<double>(3, 4) << 1.09206116e+03, 7.00301809e+00, 6.39421445e+02,
     -8.03761654e+03, 1.75020071e+00, 1.10479669e+03, 3.10853998e+02,
     3.00618340e+02, -1.38470270e-02, 2.09410228e-02, 9.99684811e-01,
     2.32549042e-01);

std::mutex mtx;
std::condition_variable condition_variable;
cv::Mat frame, frame_l, frame_r;
std::vector<cv::Point2f> trace_l;
std::vector<cv::Point2f> trace_r;
int height, width, fps;
bool frame_ready, position_ready1, position_ready2;

void trace_ball(cv::Mat frame_half, std::vector<cv::Point2f>& trace) {
    // convert to HSV
    cv::Mat hsv_frame;
    cv::cvtColor(frame_half, hsv_frame, cv::COLOR_RGB2HSV);

    // search the ball
    cv::Mat diff_frame = cv::Mat::ones(height, width / 2, CV_8U);
    for (int y = 0; y < hsv_frame.rows; y++) {
        for (int x = 0; x < hsv_frame.cols; x++) {
            int hue, sat, val;
            hue = hsv_frame.at<cv::Vec3b>(y, x)[0];
            sat = hsv_frame.at<cv::Vec3b>(y, x)[1];
            val = hsv_frame.at<cv::Vec3b>(y, x)[2];
            if ((hue >= 70 && hue <= 120) && (sat >= 90 && sat <= 255) &&
                (val >= 150 && val <= 255)) {
                diff_frame.at<uchar>(y, x) = 255;
            } else {
                diff_frame.at<uchar>(y, x) = 0;
            }
        }
    }
    cv::dilate(diff_frame, diff_frame, cv::Mat(), cv::Point(-1, -1), 1);

    // calculate the coordinate of the ball
    cv::Moments mu = cv::moments(diff_frame, false);
    cv::Point2f mc = cv::Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);
    trace.push_back(mc);
    for (int i = 0; i < trace.size(); i++) {
        cv::circle(frame_half, trace[i], 2, cv::Scalar(0, 0, 255), -1);
    }
}

void Thread_l() {
    while (!frame_ready)
        ;
    {
        // std::cout << "trace_l:try" << std::endl;
        std::lock_guard<std::mutex> lock(mtx);
        // std::cout << "trace_l:enter" << std::endl;
        trace_ball(frame_l, trace_l);
    }
    position_ready1 = true;
    // std::cout << "trace_l:exit" << std::endl;
}

void Thread_r() {
    while (!frame_ready)
        ;
    {
        // std::cout << "trace_r:try" << std::endl;
        std::lock_guard<std::mutex> lock(mtx);
        // std::cout << "trace_r:enter" << std::endl;
        trace_ball(frame_r, trace_r);
    }
    position_ready2 = true;
    // std::cout << "trace_r:exit" << std::endl;
}

void Thread_split() {
    {
        // std::cout << "split:try" << std::endl;
        std::lock_guard<std::mutex> lock(mtx);
        // std::cout << "split:lock" << std::endl;
        frame_l = frame(cv::Rect(0, 0, frame.cols / 2, frame.rows));
        frame_r =
            frame(cv::Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows));
    }
    // std::cout << "split:exit" << std::endl;
    frame_ready = true;
}

void Thread_triangulation() {
    while (!position_ready1 || !position_ready2)
        ;
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (!isnan(trace_l.back().x) && !isnan(trace_l.back().y) &&
            !isnan(trace_r.back().x) && !isnan(trace_r.back().y)) {
            cv::Mat point4D, point3D;
            std::vector<cv::Point2d> projPoints1{trace_l.back()};
            std::vector<cv::Point2d> projPoints2{trace_r.back()};
            cv::triangulatePoints(projMatr1, projMatr2, projPoints1,
                                  projPoints2, point4D);
            cv::convertPointsFromHomogeneous(point4D.reshape(4), point3D);
            point3D *= 0.035;
            std::cout << std::fixed << std::setprecision(4) << std::showpos
                      << "x:" << point3D.at<double>(0, 0)
                      << " y:" << point3D.at<double>(0, 1)
                      << " z:" << point3D.at<double>(0, 2) << std::endl;
        }
    }
}

int main() {
    std::string video_path = "./out1_1.avi";

    cv::VideoCapture cap;
    cap.open(video_path);

    if (!cap.isOpened()) {
        std::cerr << "Unable to load file." << std::endl;
        return -1;
    }

    // fps = cap.get(cv::CAP_PROP_FPS);
    fps = 120;
    width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    cv::namedWindow("Left", cv::WINDOW_NORMAL);
    cv::namedWindow("Right", cv::WINDOW_NORMAL);
    cv::moveWindow("Left", 0, 0);
    cv::moveWindow("Right", 640, 0);

    while (true) {
        std::chrono::system_clock::time_point start =
            std::chrono::system_clock::now();
        cap >> frame;
        if (frame.empty() == true) break;

        frame_ready = false;
        position_ready1 = false;
        position_ready2 = false;

        std::thread th_split(Thread_split);
        std::thread th_l(Thread_l);
        std::thread th_r(Thread_r);
        std::thread th_traiangulation(Thread_triangulation);

        th_split.join();
        th_l.join();
        th_r.join();
        th_traiangulation.join();

        cv::resizeWindow("Left", 640, 360);
        cv::resizeWindow("Right", 640, 360);
        cv::imshow("Left", frame_l);
        cv::imshow("Right", frame_r);

        std::chrono::system_clock::time_point end =
            std::chrono::system_clock::now();
        double elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count();
        if (1000 / fps - elapsed > 0) {
            cv::waitKey(1000 / fps - elapsed);
        }
    }

    return 0;
}
