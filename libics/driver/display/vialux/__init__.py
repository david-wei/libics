import os

from libics.core import env
from . import alpV42 as alp42


###############################################################################
# Initialization
###############################################################################


# Global variable for Vialux ALP4.2 API object
_ALP42 = None


def startup_alp42():
    """
    Initializes the Vialux ALP4.2 C API.

    The Vialux ALP4.2 API requires a startup and a shutdown call.
    This function checks whether startup has already been called.

    Returns
    -------
    _ALP42 : alpV42.PY_ALP_API
        Vialux ALP4.2 API object.
    """
    global _ALP42
    if _ALP42 is None:
        _ALP42 = alp42.PY_ALP_API(
            dllPath=os.path.join(env.DIR_ITFAPI, "alpV42.dll")
        )
    return _ALP42




###############################################################################


class VialuxItf():

    def __init__(self, cfg):
        super().__init__(cfg)
        self._dmd = None
        self._alp = None
        self._seq = None
        self._seq_repetitions = None
        self.API = None

    def setup(self):
        self._alp = vialux.startup_alp42()
        self.API = vialux.alp42
        if self.cfg.device is None:
            self.cfg.device = 0
        elif isinstance(self.cfg.device, str):
            self.cfg.device = int(float(self.cfg.device))
        _, self._dmd = self._alp.AlpDevAlloc(self.cfg.device)

    def shutdown(self):
        self._alp.AlpDevHalt(self._dmd)
        self._alp.AlpDevFree(self._dmd)
        self._dmd = None
        self._alp = None
        self.API = None

    def connect(self):
        pass

    def close(self):
        pass

    def init(self, bitdepth, sequence_length):
        """
        Initializes a sequence by allocating memory for image display.

        Parameters
        ----------
        bitdepth : int
            Channel bitdepth of image.
        sequence_length : int or list
            Number of images in sequence or sequence itself.
        """
        bitdepth = 1    # only 1 possible for current implementation
        if hasattr(sequence_length, "__len__"):
            sequence_length = len(sequence_length)
        _, self._seq = self._alp.AlpSeqAlloc(
            self._dmd, bitplanes=bitdepth, picnum=sequence_length
        )

    def run(self, sequence, bitdepth, repetitions):
        """
        Loads the sequence memory and starts displaying the images.

        Parameters
        ----------
        sequence : list(np.ndarray(2))
            List of images.
        bitdepth : int
            Channel bitdepth of image.
        repetitions : int
            Number of repetitions of the sequence.
            0 (zero) is interpreted as continuos display.

        Notes
        -----
        As convenience function, one can directly run a sequence
        without init, but no sequence control modifications can
        be made.
        """
        if self._seq is None:
            self.init(bitdepth, len(sequence))
        self._seq_repetitions = repetitions
        for i, image in enumerate(sequence):
            image = image.T.copy()
            image = ctypes.create_string_buffer(image.flatten().tostring())
            self._alp.AlpSeqPut(
                self._dmd, self._seq, image, picoffset=i, picload=1
            )
        if self._seq_repetitions == 0:
            self._alp.AlpProjStartCont(self._dmd, self._seq)
        else:
            self._alp.AlpProjStart(self._dmd, self._seq)

    def stop(self):
        """
        Stops any playing sequence.
        """
        self._alp.AlpProjHalt(self._dmd)
        if self._seq is not None:
            self._alp.AlpSeqFree(self._dmd, self._seq)
        self._alp.AlpDevHalt(self._dmd)
        _, self._dmd = self._alp.AlpDevAlloc(self.cfg.device)
        self._seq = None
        self._seq_repetitions = None

    # ++++ Wrapper methods ++++++++++++++

    def dev_inq(self, key):
        """Device inquiry."""
        return self._alp.AlpDevInquire(self._dmd, key)[1]

    def dev_ctrl(self, key, val):
        """Device control."""
        return self._alp.AlpDevControl(self._dmd, key, val)

    def seq_inq(self, key):
        """Sequence inquiry."""
        if self._seq is None:
            raise err.RUNTM_DRV_DSP(
                err.RUNTM_DRV_DSP.str("no sequence allocated")
            )
        else:
            return self._alp.AlpSeqInquire(self._dmd, self._seq, key)[1]

    def seq_ctrl(self, key, val):
        """Sequence control."""
        if self._seq is not None:
            return self._alp.AlpSeqControl(self._dmd, self._seq, key, val)

    def seq_time(self, **kwargs):
        """
        Set sequence timing.

        Parameters
        ----------
        illuminatetime : int
            Illuminate time in microseconds (µs).
        picturetime : int
            Picture time in microseconds (µs).
        """
        self._alp.AlpSeqTiming(self._dmd, self._seq, **kwargs)

    def seq_wait(self):
        """
        Waits for a sequence to finish and returns.

        Returns
        -------
        wait_success : bool or None
            True: Sequence has finished.
            False: Sequence in continuos mode, cannot finish.
            None: No sequence is running.
        """
        if self._seq_repetitions is None:
            return None
        elif self._seq_repetitions == 0:
            return False
        else:
            self._alp.AlpProjHalt(self._dmd)
            return True
