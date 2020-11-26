CC = g++
CFLAGS = -std=c++14 -g `pkg-config --cflags opencv4` `pkg-config --libs opencv4` -lpthread

split_video: split_video.cpp
	$(CC) $(CFLAGS) $< -o $@

track: track.cpp
	$(CC) $(CFLAGS) $< -o $@

track_single: track_single.cpp
	$(CC) $(CFLAGS) $< -o $@

track_ans: track_ans.cpp
	$(CC) $(CFLAGS) $< -o $@

track_no_condition: track_no_condition.cpp
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf left.mp4 right.mp4 split_video track track_single track_ans track_no_condition *.dSYM
