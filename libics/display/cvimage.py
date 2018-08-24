# System Imports
import cv2
import numpy as np
import os


###############################################################################


class Video(object):

    def __init__(self):
        self.cap = cv2.VideoCapture()
        self.source_mode = None
        # Frame grab function
        self.frame_function = None
        # File playback
        self.origin_file_path = None
        # Device playback
        self.device_index = None
        # Continuous playback
        self.is_playing = False
        self.interval = None
        # Recording
        self.is_recording = False
        self.recording_file_path = None

    def _continuous_playback_mode(self):
        return self.interval is not None

    def set_playback_mode(self, continuous_playback, interval=1):
        """
        Sets whether playback is continuous or static.

        Parameters
        ----------
        continuous_playback : bool
            Whether to continuously play video.
        interval : positive int
            Interval between subsequent frames in ms.
        """
        if continuous_playback:
            if interval > 0:
                self.interval = interval
        else:
            self.interval = None

    def set_source_mode(self, source_mode,
                        frame_function=None, origin_file_path=None,
                        device_index=None):
        """
        Sets frame grab mode.

        Parameters
        ----------
        source_mode : str
            "frame_function", "file", "device"
        frane_function : callable
            Function that returns the next frame.
        file : str
            Path to video file.
        device : int
            Device index.

        Returns
        -------
        ret : bool
            `True` if mode setting succeeded.
        """
        ret = True
        if source_mode == "frame_function" and callable(frame_function):
            self.frame_function = frame_function
        elif source_mode == "file" and os.path.isfile(origin_file_path):
            self.origin_file_path = origin_file_path
            self.frame_function = lambda: self.cap.read()[1]
        elif source_mode == "device" and type(device_index) == int:
            self.device_index = device_index
            self.frame_function = lambda: self.cap.read()[1]
        else:
            ret = False
        if ret:
            self.source_mode = source_mode
        return ret

    def openCap(self):
        """
        Opens the video origin (if in file or device mode).

        Required before playback.
        """
        if self.source_mode == "file":
            self.cap.open(self.origin_file_path)
        elif self.source_mode == "device":
            self.cap.open(self.device_index)

    def closeCap(self):
        """
        Closes the video origin (if in file or device mode):
        """
        if self.source_mode == "file" or self.source_mode == "device":
            self.cap.release()

    def play(self, wnd_title="cvimage play", break_key="s"):
        """
        Plays the video if in continuous mode or displays the next frame.

        Parameters
        ----------
        wnd_title : str
            Window title of video player.
        break_key : char
            Keyboard character upon which playback stops.
        """
        if self._continuous_playback_mode():
            while True:
                cv2.imshow(wnd_title, self.frame_function())
                if cv2.waitKey(self.interval) == ord(break_key):
                    break
        else:
            cv2.imshow(wnd_title, self.frame_function())


###############################################################################


if __name__ == "__main__":

    # Create test images
    im_base = np.array([list(range(1000)) for _ in range(1000)],
                       dtype="float64")
    im_base = im_base / np.max(im_base)
    im = [im_base.copy() * i for i in np.arange(0, 1, 0.01)]
    counter = 0

    # Create test video stream
    def get_test_image():
        global im, counter
        counter = (counter + 1) % len(im)
        return im[counter]

    # Play video
    v = Video()
    v.set_playback_mode(True, interval=100)
    v.set_source_mode("frame_function", frame_function=get_test_image)
    v.open()
    v.play(wnd_title="Press s to stop playback")
    v.close()
