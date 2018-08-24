import libics.drv.itf.vimba as vimba
import libics.display.cvimage as cvimage


def get_camera(index=None):
    """
    Returns the Manta `Camera` object.
    """
    vimba.startup()
    cams = vimba.get_cameras()
    if len(cams) == 0:
        print("No Vimba cameras found.")
        return None
    print("Found Vimba cameras:")
    for i in range(len(cams)):
        print("  ({:d})".format(i), cams[i].get_id())
    if index is None:
        index = 0
    if index >= len(cams):
        print("Chosen camera index {:d} not available.".format(index))
        return None
    return cams[index]


def setup_camera(camera):
    """
    Sets up a camera object.
    """
    # Configure camera settings
    camera.open_cam(mode="full")
    camera.set_acquisition(mode="Continuous")
    camera.set_format(width="max", height="max", px_format="Mono8")
    camera.set_exposure(auto="Continuous")
    print("Acquisition:", camera.get_acquisition())
    print("Format:", camera.get_format())
    print("Exposure:", camera.get_exposure())
    # Configure Vimba API
    camera.set_frame_buffers(count=3)
    return camera


def setup_video(camera):
    """
    Sets up OpenCV video display of camera.
    """
    v = cvimage.Video()
    v.set_playback_mode(True, interval=20)
    v.set_source_mode("frame_function", frame_function=camera.get_image)
    return v


def play(camera, video):
    """
    Live-displays the camera image.
    """
    print("Open cap")
    video.open_cap()
    print("Start capture")
    camera.start_capture()
    print("Start acquisition")
    camera.start_acquisition()
    print("Play stream")
    video.play(wnd_title="Press s to stop stream", break_key="s")
    print("End acquisition")
    camera.end_acquisition()
    print("End capture")
    camera.end_capture()
    print("Close cap")
    video.close_cap()
    print("Finished playback")


def cleanup(camera):
    """
    Closes camera and Vimba API.
    """
    camera.close_cam()
    vimba.shutdown()


def main():
    camera = get_camera()
    setup_camera(camera)
    video = setup_video(camera)
    play(camera, video)
    cleanup(camera)


if __name__ == "__main__":
    main()
