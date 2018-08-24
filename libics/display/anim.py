# TODO: WIP

# System Imports
import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np

# Package Imports
try:
    from . import addpath   # noqa
except(ImportError):
    import addpath          # noqa
import util.misc


###############################################################################


class Animation(object):

    def __init__(self, axes=None):
        # Plotting canvas
        self.axes = axes
        # Animation settings
        self._interval = 100    # ms
        # Pre-loaded frames
        self._frames = None
        # Animation function
        self.func = None
        self.func_anim_args = None
        self.func_const_args = None
        # Repeat
        self._repeat = True
        self._repeat_delay = 0

    def reset_func(self):
        """
        Resets animation function settings and pre-loaded frames.
        """
        self._frames = None
        self.func = None
        self.func_anim_args = None
        self.func_const_args = None

    def set_framerate(self, framerate=None, interval=None):
        """
        Sets the animation framerate.

        Parameters
        ----------
        framerate : float
            Animation playback framerate in 1/s.
        interval : int
            Animation interval between frames in ms.
            Overwrites the framerate argument.
        """
        if interval is not None:
            self._interval = interval
        elif framerate is not None:
            self._interval = 1000.0 / framerate

    def set_repeat(self, repeat=True, repeat_delay=0):
        """
        Sets whether the animation is repeated.

        Parameters
        ----------
        repeat : bool
            If after animating the whole list the animation
            should be repeated.
        repeat_delay : number
            Delay time before repetition in ms.
        """
        self._repeat = repeat
        self._repeat_delay = repeat_delay

    def pre_load_frames(self):
        """
        Calculates the whole animation sequence and saves the frames.
        """
        self._frames = [self(anim_arg) for anim_arg in self.func_anim_args]

    def set_func(self, func):
        """
        Sets the animation function.
        """
        if callable(func):
            self.func = func

    def set_func_const_args(self, const_args):
        """
        Sets the constant (w.r.t. animation variable) parameters that
        are passed in each function call.

        Parameters
        ----------
        const_args : tuple() or None
            Function call unpacks (`*`) the tuple.
            With `None`, no further arguments are passed.
        """
        self.func_const_args = const_args

    def set_func_anim_args(self, anim_args):
        """
        Sets the animation independent variables.

        Parameters
        ----------
        anim_args : list(arg):
            List of variables corresponding to each frame.
            For animation, `self.func(anim_args[i])` is called.
        """
        self.func_anim_args = util.misc.assume_iter(anim_args)

    def __call__(self, *args):
        """
        Calls the saved function with the given argument.
        """
        if self.func_const_args is None:
            return self.func(*args)
        else:
            return self.func(*args, *self.func_const_args)

    def __getitem__(self, index):
        """
        Gets the function return. If pre-loaded frames exist, the data is
        retrieved there. Otherwise a full calculation of the frame is done.
        """
        if self._frames is not None:
            return self._frames[index % len(self._frames)]
        else:
            return self(self.func_anim_args[index % len(self.func_anim_args)])

    def _imshow_pre_loaded(self, index):
        print(index)
        self.axes.imshow(self._frames[index], interpolation="none")

    def _imshow_live(self, index):
        print(index)
        self.axes.imshow(self(self.func_anim_args[index]),
                         interpolation="none")

    def animate(self, **kwargs):
        """
        Runs the animation, i.e. `imshow`s the data on the `axes` canvas.

        If pre-loaded frames exist, these are played.
        Otherwise the frames are live-calculated.

        Parameters
        ----------
        **kwargs
            Keyword arguments that are passed to the animation
            constructor `matplotlib.animation.FuncAnimation`.
            This includes: `blit` (`bool`).
        """
        fig = self.axes.get_figure()
        animation = None
        # If pre-loaded frames available
        if self._frames is not None:
            animation = matplotlib.animation.FuncAnimation(
                fig, self._imshow_pre_loaded, frames=len(self._frames),
                interval=self._interval, repeat=self._repeat,
                repeat_delay=self._repeat_delay, **kwargs
            )
        # If live-animating frames
        else:
            animation = matplotlib.animation.FuncAnimation(
                fig, self._imshow_live, frames=len(self.func_anim_args),
                interval=self._interval, repeat=self._repeat,
                repeat_delay=self._repeat_delay, **kwargs
            )
        return animation


###############################################################################


class LiveAnimation(Animation, object):

    pass


###############################################################################


def ff(var):
    x, y = var
    return (
        np.cos(2 * np.pi * np.sqrt(x**2 + y**2))
        + np.sin(5 * np.pi * np.sqrt(x**2 + y**2))
    )


def f(i):
    return np.array(
        [[ff((x + i, y)) for y in range(100)] for x in range(100)]
    )


ax = plt.subplot(111)
a = Animation(axes=ax)
a.set_func(f)
a.set_func_anim_args(list(range(200)))
print("Pre-loading frames")
a.pre_load_frames()
print("Animating")
ani = a.animate()
plt.show(ani)

"""
def f(x, y):
    return (
        np.cos(2 * np.pi * np.sqrt(x**2 + y**2))
        + np.sin(5 * np.pi * np.sqrt(x**2 + y**2))
    )


def setup_frames():
    frames = []
    for i in range(200):
        array = np.full((100, 100), 0.0)
        for x in range(array.shape[0]):
            for y in range(array.shape[1]):
                array[x, y] = f(x + i, y)
        frames.append(array)
    return frames


print("Calculating frames")
_frames = setup_frames()


def grab_frame(i):
    print(i)
    global _frames
    return _frames[i % len(_frames)]


def update(i):
    global im1
    im1.set_data(grab_frame(i))


# create axes
ax1 = plt.subplot(111)
im1 = ax1.imshow(grab_frame(0))
# create animation
print("Animating frames")
ani = matplotlib.animation.FuncAnimation(
    plt.gcf(), update, frames=200, interval=20
)
plt.show()
"""
