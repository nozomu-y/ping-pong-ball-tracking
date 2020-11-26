#include <chrono>
#include <ctime>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <ostream>
#include <ratio>
#include <string>
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

int main() {
    std::string video_path = "./out1_1.avi";

    cv::VideoCapture cap;
    cap.open(video_path);

    if (!cap.isOpened()) {
        std::cerr << "Unable to load file." << std::endl;
        return -1;
    }

    int fps = 120;
    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    cv::namedWindow("Left", cv::WINDOW_NORMAL);
    cv::namedWindow("Right", cv::WINDOW_NORMAL);
    cv::moveWindow("Left", 0, 0);
    cv::moveWindow("Right", 640, 0);

    std::vector<cv::Point2f> track_l;
    std::vector<cv::Point2f> track_r;

    int FOURCC = cv::VideoWriter::fourcc('M', 'P', '4', 'V');
    cv::VideoWriter vw("track.mp4", FOURCC, fps, cv::Size(width, height), 1);

    while (1) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty() == true) break;

        // split frame
        cv::Mat frame_l, frame_r;
        frame_l = frame(cv::Rect(0, 0, frame.cols / 2, frame.rows));
        frame_r =
            frame(cv::Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows));

        // convert to HSV
        cv::Mat hsv_frame_l, hsv_frame_r;
        cv::cvtColor(frame_l, hsv_frame_l, cv::COLOR_RGB2HSV);
        cv::cvtColor(frame_r, hsv_frame_r, cv::COLOR_RGB2HSV);

        // search the ball
        cv::Mat diff_frame_l = cv::Mat::ones(height, width / 2, CV_8U);
        cv::Mat diff_frame_r = cv::Mat::ones(height, width / 2, CV_8U);

        int hueMin = 70;
        int hueMax = 120;
        int saturationMin = 90;
        int saturationMax = 255;
        int brightnessMin = 150;
        int brightnessMax = 255;
        cv::inRange(
            hsv_frame_l, cv::Scalar(hueMin, saturationMin, brightnessMin, 0),
            cv::Scalar(hueMax, saturationMax, brightnessMax, 0), diff_frame_l);
        cv::inRange(
            hsv_frame_r, cv::Scalar(hueMin, saturationMin, brightnessMin, 0),
            cv::Scalar(hueMax, saturationMax, brightnessMax, 0), diff_frame_r);

        cv::dilate(diff_frame_l, diff_frame_l, cv::Mat(), cv::Point(-1, -1), 1);
        cv::dilate(diff_frame_r, diff_frame_r, cv::Mat(), cv::Point(-1, -1), 1);

        // calculate the coordinate of the ball
        cv::Moments mu_l = cv::moments(diff_frame_l, false);
        cv::Moments mu_r = cv::moments(diff_frame_r, false);
        cv::Point2f coord_l =
            cv::Point2f(mu_l.m10 / mu_l.m00, mu_l.m01 / mu_l.m00);
        cv::Point2f coord_r =
            cv::Point2f(mu_r.m10 / mu_r.m00, mu_r.m01 / mu_r.m00);
        track_l.push_back(coord_l);
        track_r.push_back(coord_r);

        for (int i = 0; i < track_l.size(); i++) {
            cv::circle(frame_l, track_l[i], 2, cv::Scalar(0, 0, 255), -1);
        }
        for (int i = 0; i < track_r.size(); i++) {
            cv::circle(frame_r, track_r[i], 2, cv::Scalar(0, 0, 255), -1);
        }

        // triangulation
        if (!isnan(coord_l.x) && !isnan(coord_l.y) && !isnan(coord_r.x) &&
            !isnan(coord_r.y)) {
            std::vector<cv::Point2d> projPoints1{coord_l};
            std::vector<cv::Point2d> projPoints2{coord_r};
            cv::Mat point4D, point3D;
            cv::triangulatePoints(projMatr1, projMatr2, projPoints1,
                                  projPoints2, point4D);
            cv::convertPointsFromHomogeneous(point4D.reshape(4), point3D);
            point3D *= 0.035;
            printf("x:%+.4f y:%+.4f z:%+.4f\n", point3D.at<double>(0, 0),
                   point3D.at<double>(0, 1), point3D.at<double>(0, 2));
        }
        cv::resizeWindow("Left", 640, 360);
        cv::resizeWindow("Right", 640, 360);
        cv::imshow("Left", frame_l);
        cv::imshow("Right", frame_r);

        cv::Mat frame_out;
        cv::Mat frames[] = {frame_l, frame_r};
        cv::hconcat(frames, 2, frame_out);
        vw << frame_out;

        cv::waitKey(1000 / fps);
    }

    return 0;
}
