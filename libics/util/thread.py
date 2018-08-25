import threading
import time


###############################################################################
# Timers
###############################################################################


class StoppableThread(threading.Thread):

    """
    Thread that ends on a stop event.
    """

    def __init__(self):
        super().__init__()
        self.stop_event = threading.Event()

    def stop(self):
        """
        Stops the thread.
        """
        if self.isAlive():
            # set event to signal thread to terminate
            self.stop_event.set()
            # block calling thread until thread really has terminated
            self.join()


class PeriodicTimer(StoppableThread):

    """
    Periodic timer that runs a worker function after each period timeout.

    Parameters
    ----------
    period : int
        Timeout period in ms.
    """

    def __init__(self, period, worker_func, *args, **kwargs):
        super().__init__()
        self._period = period
        self._worker_func = worker_func
        self.set_args(*args, **kwargs)

    def start(self):
        """
        Starts a timer that periodically calls a function.

        Notes
        -----
        Can be stopped by running the `stop()` method.
        No dynamic arguments are allowed. Static (keyword) arguments can be set
        by calling the `set_args(*args, **kwargs)` method.
        """
        while not self.stop_event.is_set():
            self._worker_func(*self._args, **self._kwargs)
            time.sleep(self._interval)

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
