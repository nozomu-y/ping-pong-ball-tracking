#include <chrono>
#include <ctime>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <ostream>
#include <ratio>
#include <string>
#include <thread>
#include <vector>

int height, width, fps;

cv::VideoCapture cap;
cv::Mat frame, frame_l, frame_r;
cv::Point2f coord_l, coord_r;

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

std::mutex mtx_frame_r, mtx_frame_l;
std::mutex mtx_coord_l, mtx_coord_r;
std::condition_variable cv_frame_r, cv_frame_l;
std::condition_variable cv_coord_l, cv_coord_r;
bool frame_r_ready = false;
bool frame_l_ready = false;
bool coord_r_ready = false;
bool coord_l_ready = false;

int hueMin = 70;
int hueMax = 120;
int saturationMin = 90;
int saturationMax = 255;
int brightnessMin = 150;
int brightnessMax = 255;

void Thread_split() {
    while (1) {
        std::chrono::system_clock::time_point start =
            std::chrono::system_clock::now();
        cap >> frame;
        if (frame.empty() == true) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000 / fps));
            std::exit(0);
        }

        {
            std::lock_guard<std::mutex> lock_l(mtx_frame_l);
            std::lock_guard<std::mutex> lock_r(mtx_frame_r);

            frame_l = frame(cv::Rect(0, 0, frame.cols / 2, frame.rows));
            frame_r =
                frame(cv::Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows));

            frame_l_ready = true;
            frame_r_ready = true;
        }
        cv_frame_l.notify_one();
        cv_frame_r.notify_one();

        std::chrono::system_clock::time_point end =
            std::chrono::system_clock::now();
        double elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count();
        int sleep = 1000 / fps - elapsed;
        if (1000 / fps - elapsed > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep));
        }
    }
}

void Thread_l() {
    cv::Mat hsv_frame;
    while (1) {
        {
            std::unique_lock<std::mutex> uniq_lk(mtx_frame_l);
            cv_frame_l.wait(uniq_lk, [] { return frame_l_ready; });
            frame_l_ready = false;
            // convert to HSV
            cv::cvtColor(frame_l, hsv_frame, cv::COLOR_RGB2HSV);
        }

        // search the ball
        cv::Mat diff_frame = cv::Mat::ones(height, width / 2, CV_8U);
        cv::inRange(
            hsv_frame, cv::Scalar(hueMin, saturationMin, brightnessMin, 0),
            cv::Scalar(hueMax, saturationMax, brightnessMax, 0), diff_frame);
        cv::dilate(diff_frame, diff_frame, cv::Mat(), cv::Point(-1, -1), 1);

        // calculate the coordinate of the ball
        cv::Moments mu = cv::moments(diff_frame, false);
        cv::Point2f mc = cv::Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);
        {
            std::lock_guard<std::mutex> lock(mtx_coord_l);
            coord_l = mc;
            coord_l_ready = true;
        }
        cv_coord_l.notify_one();
    }
}

void Thread_r() {
    cv::Mat hsv_frame;
    while (1) {
        {
            std::unique_lock<std::mutex> uniq_lk(mtx_frame_r);
            cv_frame_r.wait(uniq_lk, [] { return frame_r_ready; });
            frame_r_ready = false;
            // convert to HSV
            cv::cvtColor(frame_r, hsv_frame, cv::COLOR_RGB2HSV);
        }

        // search the ball
        cv::Mat diff_frame = cv::Mat::ones(height, width / 2, CV_8U);
        cv::inRange(
            hsv_frame, cv::Scalar(hueMin, saturationMin, brightnessMin, 0),
            cv::Scalar(hueMax, saturationMax, brightnessMax, 0), diff_frame);
        cv::dilate(diff_frame, diff_frame, cv::Mat(), cv::Point(-1, -1), 1);

        // calculate the coordinate of the ball
        cv::Moments mu = cv::moments(diff_frame, false);
        cv::Point2f mc = cv::Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);
        {
            std::lock_guard<std::mutex> lock(mtx_coord_r);
            coord_r = mc;
            coord_r_ready = true;
        }
        cv_coord_r.notify_one();
    }
}

void Thread_triangulation() {
    cv::Mat point4D, point3D;
    while (1) {
        {
            std::unique_lock<std::mutex> uniq_lk_l(mtx_coord_l);
            std::unique_lock<std::mutex> uniq_lk_r(mtx_coord_r);
            cv_coord_l.wait(uniq_lk_l, [] { return coord_l_ready; });
            cv_coord_r.wait(uniq_lk_r, [] { return coord_r_ready; });
            coord_l_ready = false;
            coord_r_ready = false;

            if (isnan(coord_l.x) || isnan(coord_l.y) || isnan(coord_r.x) ||
                isnan(coord_r.y))
                continue;

            std::vector<cv::Point2d> projPoints1{coord_l};
            std::vector<cv::Point2d> projPoints2{coord_r};
            cv::triangulatePoints(projMatr1, projMatr2, projPoints1,
                                  projPoints2, point4D);
        }

        cv::convertPointsFromHomogeneous(point4D.reshape(4), point3D);
        point3D *= 0.035;
        std::cout << std::fixed << std::setprecision(4) << std::showpos
                  << "x:" << point3D.at<double>(0, 0)
                  << " y:" << point3D.at<double>(0, 1)
                  << " z:" << point3D.at<double>(0, 2) << std::endl;
    }
}

int main() {
    std::string video_path = "./out1_1.avi";

    cap.open(video_path);

    if (!cap.isOpened()) {
        std::cerr << "Unable to load file." << std::endl;
        return -1;
    }

    fps = 120;
    width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    std::thread th_split(Thread_split);
    std::thread th_l(Thread_l);
    std::thread th_r(Thread_r);
    std::thread th_traiangulation(Thread_triangulation);

    th_split.join();
    th_l.join();
    th_r.join();
    th_traiangulation.join();

    return 0;
}
