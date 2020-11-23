CC = g++
CFLAGS = -std=c++14 -g `pkg-config --cflags opencv4` `pkg-config --libs opencv4`

split_video: split_video.cpp
	$(CC) $(CFLAGS) split_video.cpp -o split_video

track: track.cpp
	$(CC) $(CFLAGS) track.cpp -o track

track_ans: track_ans.cpp
	$(CC) $(CFLAGS) track_ans.cpp -o track_ans

clean:
	rm -rf left.mp4 right.mp4 split_video track track_ans
