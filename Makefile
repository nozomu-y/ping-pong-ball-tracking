CC = g++
CFLAGS = -std=c++14 `pkg-config --cflags opencv4` `pkg-config --libs opencv4`

split_video: split_video.cpp
	$(CC) $(CFLAGS) split_video.cpp -o split_video

trace_ball: trace_ball.cpp
	$(CC) $(CFLAGS) trace_ball.cpp -o trace_ball

clean:
	rm -rf left.mp4 right.mp4 split_video trace_ball
