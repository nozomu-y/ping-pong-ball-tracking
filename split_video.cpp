#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
using namespace std;
using namespace cv;

int main() {
    string video_path = "./dual.mp4";

    VideoCapture cap;
    cap.open(video_path);

    if (!cap.isOpened()) {
        cerr << "Unable to load file." << endl;
        return -1;
    }

    int fps = cap.get(CAP_PROP_FPS);

    Mat frame, frame_l, frame_r;
    namedWindow("Left", WINDOW_NORMAL);
    namedWindow("Right", WINDOW_NORMAL);
    moveWindow("Left", 0, 0);
    moveWindow("Right", 640, 0);

    while (true) {
        // get one frame
        cap >> frame;
        if (frame.empty() == true) break;

        // trim frame
        frame_l = frame(Rect(0, 0, frame.cols / 2, frame.rows));
        frame_r = frame(Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows));

        resizeWindow("Left", 640, 360);
        resizeWindow("Right", 640, 360);
        imshow("Left", frame_l);
        imshow("Right", frame_r);

        waitKey(1000 / fps);
    }

    return 0;
}
