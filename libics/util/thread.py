import threading
import time


###############################################################################
# Timers
###############################################################################


class StoppableThread(threading.Thread):

    """
    Thread that ends on a stop event.

    Parameters
    ----------
    stop_action : callable or None
        Function that is called upon `stop` method call.
        `None` is interpreted as a `pass` function.

    Notes
    -----
    This thread cannot be restarted.
    """

    def __init__(self, stop_action=None):
        super().__init__()
        self.stop_event = threading.Event()
        self.stop_action = stop_action

    def stop(self):
        """
        Stops the thread.
        """
        if self.isAlive():
            # set event to signal thread to terminate
            self.stop_event.set()
            # block calling thread until thread really has terminated
            self.join()
            if callable(self.stop_action):
                self.stop_action()


class PeriodicTimer():

    """
    Periodic timer that runs a worker function after each period timeout.

    Parameters
    ----------
    period : float
        Timeout period in seconds.

    Notes
    -----
    This timer can be restarted.
    """

    def __init__(self, period, worker_func, *args, **kwargs):
        self._thread = None
        self._stop_action = None
        self._period = period
        self._worker_func = worker_func
        self.set_args(*args, **kwargs)

    def run(self):
        """
        Timer thread method that periodically calls a function.

        For starting a thread running this `run` method, call the `start`
        method.

        Notes
        -----
        Can be stopped by running the `stop()` method.
        No dynamic arguments are allowed. Static (keyword) arguments can be set
        by calling the `set_args(*args, **kwargs)` method.
        Timer has feedback, i.e. corrects runtime delays if delays are smaller
        than the period.
        """
        target_time = time.time()
        while not self._thread.stop_event.is_set():
            self._worker_func(*self._args, **self._kwargs)
            diff_time = time.time() - target_time
            sleep_time = max(0, self._period - diff_time)
            time.sleep(sleep_time)
            target_time += self._period

    def start(self, *args, **kwargs):
        """
        Starts a thread which runs the `run` method.

        Arguments are passed on to the `Thread.start` method.
        """
        if self._thread is None:
            self._thread = StoppableThread(stop_action=self._stop_action)
            self._thread.run = self.run
            self._thread.start(*args, **kwargs)

    def stop(self):
        """
        Stops the timer.
        """
        if self._thread is not None and self._thread.isAlive():
            self._thread.stop()
            self._thread = None

    def set_stop_action(self, stop_action):
        """
        Sets the stop action.

        Parameters
        ----------
        stop_action : callable
            Function that is called upon `stop` method call.
        """
        if callable(stop_action):
            self._stop_action = stop_action

    def set_args(self, *args, **kwargs):
        """
        Sets the arguments with which the worker function is called.

        The worker function will be called as `worker_func(*args, **kwargs)`.
        """
        self._args = args
        self._kwargs = kwargs

    def get_period(self):
        return self._period

    def get_function(self):
        return self._worker_func

    def __call__(self):
        return self._worker_func(*self._args, **self._kwargs)


###############################################################################
# Tests
###############################################################################


if __name__ == "__main__":

    # Create test periodic timer
    def test_func():
        print("Timer timeout")

    timer = PeriodicTimer(2, test_func)

    # Run test
    timer.start()
    print("Start timer")
    for t in range(10):
        time.sleep(1)
        print("Clock: {:d}s".format(t + 1))
    print("Stop timer")
    timer.stop()
    time.sleep(2)
    print("Restart timer")
    timer.start()
    for t in range(5):
        time.sleep(1)
        print("Clock: {:d}s".format(t + 1))
    print("Stop timer")
    timer.stop()
