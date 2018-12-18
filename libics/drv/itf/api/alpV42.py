"""
Author of original file: Konrad

[THIS FILE HAS BEEN MODIFIED FOR PYTHON 3.]

This file is the translation of the c header file alp.h to our Python API.
(The h file can be found as reference in this directory.)
All functions connecting with the DMD are stored as methods of the PY_ALP_API
class. The necessary library is alpV42.dll for 64bit (alpV42S.dll for 32bit).
You might have to install the driver software properly with the CD (ALP-4.2
Controller Suite VIALUX/DLP/Texas Instruments Installation CD).

Changelog
---------
* Various changes for Python 3
* Various changes in formatting for PEP8
* PY_ALP_API.AlpDevAlloc: Bugfix
  Switched initflag and devicenum
"""

import ctypes as ct
from ctypes import byref
import numpy as np
import matplotlib.pyplot as plt


ALP_ERRORS = {
    "ALP_OK":               0,      # successfull execution
    "ALP_NOT_ONLINE":       1001,   # The specified ALP has not been found or
                                    # is not ready.
    "ALP_NOT_IDLE":         1002,   # The ALP is not in idle state.
    "ALP_NOT_AVAILABLE":    1003,   # The specified ALP identifier is not
                                    # valid.
    "ALP_NOT_READY":        1004,   # The specified ALP is already allocated.
    "ALP_PARM_INVALID":     1005,   # One of the parameters is invalid.
    "ALP_ADDR_INVALID":     1006,   # Error accessing user data.
    "ALP_MEMORY_FULL":      1007,   # The requested memory is not available.
    "ALP_SEQ_IN_USE":       1008,   # The sequence specified is currently in
                                    # use.
    "ALP_HALTED":           1009,   # The ALP has been stopped while image
                                    # data transfer was active.
    "ALP_ERROR_INIT":       1010,   # Initialization error.
    "ALP_ERROR_COMM":       1011,   # Communication error.
    "ALP_DEVICE_REMOVED":   1012,   # The specified ALP has been removed.
    "ALP_NOT_CONFIGURED":   1013,   # The onboard FPGA is unconfigured.
    "ALP_LOADER_VERSION":   1014,   # The function is not supported by this
                                    # version of the driver file VlxUsbLd.sys.
    "ALP_ERROR_DONGLE":     1015,   # Please connect the USB Dongle
    "ALP_DEV_BUSY":         1100,   # the ALP is displaying a sequence or
                                    # image data download is active
    "ALP_DEV_READY":        1101,   # the ALP is ready for further requests
    "ALP_DEV_IDLE":         1102,   # the ALP is in wait state
    "ALP_PROJ_ACTIVE":      1200,   # ALP projection active
    "ALP_PROJ_IDLE":        1201    # no projection active
}


def flipDict(dictionary):
    flippedDict = {}
    for k, v in dictionary.iteritems():
        flippedDict[str(v)] = k
    return flippedDict


ErrorsFlipped = flipDict(ALP_ERRORS)


ALP_PARMS_DEV_CONTROL_TYPE = {
    "ALP_SYNCH_POLARITY":       2004,   # Select frame synch output signal
                                        # polarity
    "ALP_TRIGGER_EDGE":         2005,   # Select active input trigger edge
                                        # (slave mode)
    "ALP_USB_CONNECTION":       2016,   # Re-connect after a USB interruption
    "ALP_DEV_DMDTYPE":          2021,   # Select DMD type; only allowed for a
                                        # new allocated ALP-3 high-speed device
    "ALP_DEV_DISPLAY_HEIGHT":   2057,   # number of mirror rows on the DMD
    "ALP_DEV_DISPLAY_WIDTH":    2058    # number of mirror columns on the DMD
}

ALP_PARMS_DEV_CONTROL_TYPE_FLIP = flipDict(ALP_PARMS_DEV_CONTROL_TYPE)


ALP_PARMS_DEV_CONTROL_VALUE = {
    "ALP_LEVEL_HIGH":           2006,   # Active high synch output
    "ALP_LEVEL_LOW":            2007,   # Active low synch output
    "ALP_EDGE_FALLING":         2008,   # High to low signal transition
    "ALP_EDGE_RISING":          2009,   # Low to high signal transition
    "ALP_DMDTYPE_XGA":          1,      # 1024x768 px (0.7" Type A, D3000)
    "ALP_DMDTYPE_SXGA_PLUS":    2,      # 1400x1050 px (0.95" Type A, D3000)
    "ALP_DMDTYPE_1080P_095A":   3,      # 1920x1080 px (0.95" Type A, D4x00)
    "ALP_DMDTYPE_XGA_07A":      4,      # 1024x768 px (0.7" Type A, D4x00)
    "ALP_DMDTYPE_XGA_055A":     5,      # 1024x768 px (0.55" Type A, D4x00)
    "ALP_DMDTYPE_XGA_055X":     6,      # 1024x768 px (0.55" Type X, D4x00)
    "ALP_DMDTYPE_WUXGA_096A":   7,      # 1920x1200 px (0.96" Type A, D4100)
    "ALP_DMDTYPE_DISCONNECT":   255     # behaves like 1080p (D4100)
}

ALP_PARMS_DEV_CONTROL_VALUE_FLIP = flipDict(ALP_PARMS_DEV_CONTROL_VALUE)


ALP_PARMS_DEV_INQUIRE = {
    "ALP_DEVICE_NUMBER":            2000,   # Serial number of the ALP device
    "ALP_VERSION":                  2001,   # Version number of the ALP device
    "ALP_DEV_STATE":                2002,   # current ALP status, see above
    "ALP_AVAIL_MEMORY":             2003,   # ALP on-board sequence memory
                                            # available for further sequence
    "ALP_SYNCH_POLARITY":           2004,   # Select frame synch output signal
                                            # polarity
    "ALP_TRIGGER_EDGE":             2005,   # Select active input trigger edge
                                            # (slave mode)
    "ALP_USB_CONNECTION":           2016,   # Re-connect after a USB
                                            # interruption
    "ALP_DEV_DMDTYPE":              2021,   # Select DMD type
                                            # only allowed for a new allocated
                                            # ALP-3 high-speed device
                                            # allocation (AlpSeqAlloc)
                                            # number of binary pictures
    # Temperatures. Data format: signed long with 1 LSB=1/256 °C
    "ALP_DDC_FPGA_TEMPERATURE":     2050,   # V4100 Rev B: LM95231.
                                            # External channel:
                                            # DDC FPGAs Temperature Diode
    "ALP_APPS_FPGA_TEMPERATURE":    2051,   # V4100 Rev B: LM95231.
                                            # External channel:
                                            # Applic. FPGAs Temperature Diode
    "ALP_PCB_TEMPERATURE":          2052,   # V4100 Rev B: LM95231.
                                            # Internal channel:
                                            # "Board temperature"
    "ALP_DEV_DISPLAY_HEIGHT":       2057,   # number of DMD mirror rows
    "ALP_DEV_DISPLAY_WIDTH":        2058    # number of DMD mirror columns
}

ALP_PARMS_DEV_INQUIRE_FLIP = flipDict(ALP_PARMS_DEV_INQUIRE)


ALP_PARMS_SEQ_CONTROL_TYPE = {
    "ALP_SEQ_REPEAT":   2100,   # Non-continuous display of a sequence
                                # (AlpProjStart) allows  for configuring the
                                # number of sequence iterations.
    "ALP_FIRSTFRAME":   2101,   # First image of this sequence to be displayed.
    "ALP_LASTFRAME":    2102,   # Last image of this sequence to be displayed.
    "ALP_BITNUM  ":     2103,   # A sequence can be displayed with reduced bit
                                # depth for faster speed.
    "ALP_BIN_MODE":     2104,   # Binary mode: select from ALP_BIN_NORMAL and
                                # ALP_BIN_UNINTERRUPTED (AlpSeqControl)
    "ALP_DATA_FORMAT":  2110,   # Data format and alignment
}

ALP_PARMS_SEQ_CONTROL_TYPE_FLIP = flipDict(ALP_PARMS_SEQ_CONTROL_TYPE)


ALP_PARMS_SEQ_CONTROL_VALUE = {
    "ALP_BIN_NORMAL":           2105,   # Normal operation with progammable
                                        # dark phase
    "ALP_BIN_UNINTERRUPTED":    2106,   # Operation without dark phase
    "ALP_DATA_MSB_ALIGN":       0,      # Data is MSB aligned (default)
    "ALP_DATA_LSB_ALIGN":       1,      # Data is LSB aligned
    "ALP_DATA_BINARY_TOPDOWN":  2,      # Data is packed binary, top row first
                                        # bit7 of a byte = leftmost of 8 pixels
    "ALP_DATA_BINARY_BOTTOMUP": 3       # Data is packed binary, bot. row first
                                        # XGA:  one pixel row occupies 128 byte
                                        #       of binary data.
                                        #       Byte0.Bit7 = top left pixel
                                        #       (TOPDOWN format)
                                        # SXGA+: one pixel row occupies 176
                                        #        byte of binary data.
                                        #        First byte ignored.
                                        #        Byte1.Bit7 = top left pixel
                                        #        (TOPDOWN format)
}

ALP_PARMS_SEQ_CONTROL_VALUE_FLIP = flipDict(ALP_PARMS_SEQ_CONTROL_VALUE)


ALP_PARMS_SEQ_INQUIRE = {
    "ALP_BITPLANES":            2200,   # Bit depth of the pictures in the
                                        # sequence
    "ALP_BITNUM  ":             2103,   # A sequence can be displayed with
                                        # reduced bit depth for faster speed.
    "ALP_BIN_MODE":             2104,   # Binary mode: select from
                                        # ALP_BIN_NORMAL and
                                        # ALP_BIN_UNINTERRUPTED (AlpSeqControl)
    "ALP_PICNUM  ":             2201,   # Number of pictures in the sequence
    "ALP_FIRSTFRAME":           2101,   # First image of this sequence to be
                                        # displayed.
    "ALP_LASTFRAME":            2102,   # Last image of this sequence to be
                                        # displayed.
    "ALP_SEQ_REPEAT":           2100,   # Non-continuous display of a sequence
                                        # (AlpProjStart) allows for
                                        # configuring the number of sequence
                                        # iterations.
    "ALP_PICTURE_TIME":         2203,   # Time between the start of
                                        # consecutive pictures in the sequence
                                        # in microseconds, the corresponding
                                        # in frames per second is picture rate
                                        # [fps] = 1e6 / ALP_PICTURE_TIME [µs]
    "ALP_MIN_PICTURE_TIME":     2211,   # Minimum time between the start of
                                        # consecutive pictures in microseconds
    "ALP_MAX_PICTURE_TIME":     2213,   # Maximum value of ALP_PICTURE_TIME
                                        # = ALP_ON_TIME + ALP_OFF_TIME
                                        # ALP_ON_TIME may be smaller than
                                        # ALP_ILLUMINATE_TIME
    "ALP_ILLUMINATE_TIME":      2204,   # Duration of the display of one
                                        # picture in microseconds
    "ALP_MIN_ILLUMINATE_TIME":  2212,   # Minimum duration of the display of
                                        # one picture in microseconds depends
                                        # on ALP_BITNUM and ALP_BIN_MODE
    "ALP_ON_TIME":              2214,   # Total active projection time
    "ALP_OFF_TIME":             2215,   # Total inactive projection time
    "ALP_SYNCH_DELAY":          2205,   # Delay of the start of picture
                                        # display with respect to the frame
                                        # synch output (master mode) in µs
    "ALP_MAX_SYNCH_DELAY":      2209,   # Maximal duration of frame synch
                                        # output to projection delay in µs
    "ALP_SYNCH_PULSEWIDTH":     2206,   # Duration of the active frame synch
                                        # output pulse in microseconds
    "ALP_TRIGGER_IN_DELAY":     2207,   # Delay of the start of picture
                                        # display with respect to the active
                                        # trigger input edge in microseconds
    "ALP_MAX_TRIGGER_IN_DELAY": 2210,   # Maximal duration of trigger input to
                                        # projection delay in microseconds
    "ALP_DATA_FORMAT":          2110    # Data format and alignment
}

ALP_PARMS_SEQ_INQUIRE_FLIP = flipDict(ALP_PARMS_SEQ_INQUIRE)


ALP_PARMS_PROJ = {
    "ALP_DEFAULT":          0,
    "ALP_PROJ_MODE":        2300,   # Select from ALP_MASTER and ALP_SLAVE mode
    "ALP_MASTER  ":         2301,   # The ALP operation is controlled by
                                    # internal timing, a synch signal is sent
                                    # out for any picture displayed
    "ALP_SLAVE":            2302,   # The ALP operation is controlled by
                                    # external trigger, the next picture in a
                                    # sequence is displayed after the
                                    # detection of an external input trigger
                                    # signal.
    "ALP_PROJ_SYNC":        2303,   # Select from ALP_SYNCHRONOUS and
                                    # ALP_ASYNCHRONOUS mode
    "ALP_SYNCHRONOUS":      2304,   # The calling program gets control back
                                    # after completion of sequence display.
    "ALP_ASYNCHRONOUS":     2305,   # The calling program gets control back
                                    # immediately.
    "ALP_PROJ_INVERSION":   2306,   # Reverse dark into bright
    "ALP_PROJ_UPSIDE_DOWN": 2307,   # Turn the pictures upside down
    "ALP_PROJ_STATE":       2400,   # Inquire only
    "ALP_PROJ_ACTIVE":      1200,   # ALP projection active
    "ALP_PROJ_IDLE":        1201    # no projection active
}

ALP_PARMS_PROJ_FLIP = flipDict(ALP_PARMS_PROJ)


###############################################################################


class PY_ALP_API:
    """Functions for the ALP."""

    def __init__(self, dllPath="alpV42.dll"):
        self.dll = ct.windll.LoadLibrary(dllPath)

    # +++++++++++++++++++++

    def AlpDevAlloc(self, devicenum=0, initflag=0):
        InitFlag = ct.c_long(initflag)
        DeviceNum = ct.c_long(devicenum)
        DeviceHandle = ct.c_long(0)
        err = self.dll.AlpDevAlloc(DeviceNum, InitFlag, byref(DeviceHandle))
        print("AlpDevAlloc: {:s}".format(ErrorsFlipped[str(err)]))
        return err, DeviceHandle.value

    def AlpDevHalt(self, devhandle):
        DeviceHandle = ct.c_long(devhandle)
        err = self.dll.AlpDevFree(DeviceHandle)
        print("AlpDevHalt: {:s}".format(ErrorsFlipped[str(err)]))
        return err

    def AlpDevFree(self, devhandle):
        DeviceHandle = ct.c_long(devhandle)
        err = self.dll.AlpDevFree(DeviceHandle)
        print("AlpDevFree: {:s}".format(ErrorsFlipped[str(err)]))
        return err

    # +++++++++++++++++++++

    def AlpDevControl(self, devhandle, controltype, controlvalue):
        DeviceHandle = ct.c_long(devhandle)
        ControlType = ct.c_long(controltype)
        ControlValue = ct.c_long(controlvalue)
        err = self.dll.AlpDevControl(DeviceHandle, ControlType, ControlValue)
        print("AlpDevControl: {:s}, {:s} is {:s}".format(
            ErrorsFlipped[str(err)],
            ALP_PARMS_DEV_CONTROL_TYPE_FLIP[str(controltype)],
            ALP_PARMS_DEV_CONTROL_VALUE_FLIP[str(controlvalue)]
        ))
        return err

    def AlpDevInquire(self, devhandle, inquiretype):
        DeviceHandle = ct.c_long(devhandle)
        InquireType = ct.c_long(inquiretype)
        InquireValue = ct.c_long(0)
        err = self.dll.AlpDevInquire(
            DeviceHandle, InquireType, byref(InquireValue)
        )
        print("AlpDevInquire: {:s}, {:s} is {:d}".format(
            ErrorsFlipped[str(err)],
            ALP_PARMS_DEV_INQUIRE_FLIP[str(inquiretype)],
            InquireValue.value
        ))
        if inquiretype == (2050):
            temp = int(InquireValue.value / 256.)
            print("Temperature in Centigrades = {:2.1f}".format(temp))
        elif inquiretype == (2051):
            temp = InquireValue.value / 256.
            print("Temperature in Centigrades = {:2.1f}".format(temp))
        elif inquiretype == (2052):
            temp = InquireValue.value / 256.
            print("Temperature in Centigrades = {:2.1f}".format(temp))
        return err, InquireValue.value

    # +++++++++++++++++++++

    def AlpSeqAlloc(self, devhandle, bitplanes=1, picnum=1):
        DeviceHandle = ct.c_long(devhandle)
        BitPlanes = ct.c_long(bitplanes)
        PicNum = ct.c_long(picnum)
        SeqHandle = ct.c_ulong(0)
        err = self.dll.AlpSeqAlloc(
            DeviceHandle, BitPlanes, PicNum, byref(SeqHandle)
        )
        print("AlpSeqAlloc: {:s}".format(ErrorsFlipped[str(err)]))
        return err, SeqHandle.value

    def AlpSeqFree(self, devhandle, seqhandle):
        DeviceHandle = ct.c_long(devhandle)
        SeqHandle = ct.c_long(seqhandle)
        err = self.dll.AlpSeqFree(DeviceHandle, SeqHandle)
        print("AlpSeqFree: {:s}".format(ErrorsFlipped[str(err)]))
        return err

    def AlpSeqControl(self, devhandle, seqhandle, controltype, controlvalue):
        """
        ALP_DEFAULT doesn"t work. Use specified parameters only.
        """
        DeviceHandle = ct.c_long(devhandle)
        SeqHandle = ct.c_long(seqhandle)
        ControlType = ct.c_long(controltype)
        ControlValue = ct.c_long(controlvalue)
        err = self.dll.AlpSeqControl(
            DeviceHandle, SeqHandle, ControlType, ControlValue
        )
        if controltype == 2110:
            print("AlpSeqControl: {:s}, {:s} is {:s}".format(
                ErrorsFlipped[str(err)],
                ALP_PARMS_SEQ_CONTROL_TYPE_FLIP[str(controltype)],
                ALP_PARMS_SEQ_CONTROL_VALUE_FLIP[str(controlvalue)]
            ))
        elif controltype == 2104:
            print("AlpSeqControl: {:s}, {:s} is {:s}".format(
                ErrorsFlipped[str(err)],
                ALP_PARMS_SEQ_CONTROL_TYPE_FLIP[str(controltype)],
                ALP_PARMS_SEQ_CONTROL_VALUE_FLIP[str(controlvalue)]
            ))
        else:
            print("AlpSeqControl: {:s}, {:s} is {:d}".format(
                ErrorsFlipped[str(err)],
                ALP_PARMS_SEQ_CONTROL_TYPE_FLIP[str(controltype)],
                controlvalue
            ))
        return err

    def AlpSeqTiming(
        self,
        devhandle, seqhandle, illuminatetime=0, picturetime=0,
        synchdelay=0, synchpulsewidth=0, triggerindelay=0
    ):
        """Time values are given in microseconds (mus)."""
        DeviceHandle = ct.c_long(devhandle)
        SeqHandle = ct.c_long(seqhandle)
        IlluminateTime = ct.c_long(illuminatetime)
        PictureTime = ct.c_long(picturetime)
        SynchDelay = ct.c_long(synchdelay)
        SynchPulseWidth = ct.c_long(synchpulsewidth)
        TriggerInDelay = ct.c_long(triggerindelay)
        err = self.dll.AlpSeqTiming(
            DeviceHandle, SeqHandle, IlluminateTime, PictureTime,
            SynchDelay, SynchPulseWidth, TriggerInDelay
        )
        print("AlpSeqTiming: {:s}".format(ErrorsFlipped[str(err)]))
        return err

    def AlpSeqInquire(self, devhandle, seqhandle, inquiretype):
        """Time values are given in microseconds (mus)."""
        DeviceHandle = ct.c_long(devhandle)
        SeqHandle = ct.c_long(seqhandle)
        InquireType = ct.c_long(inquiretype)
        InquireValue = ct.c_long(0)
        err = self.dll.AlpSeqInquire(
            DeviceHandle, SeqHandle, InquireType, byref(InquireValue)
        )
        print("AlpSeqInquire: {:s}, {:s} is {:d}".format(
            ErrorsFlipped[str(err)],
            ALP_PARMS_SEQ_INQUIRE_FLIP[str(inquiretype)],
            InquireValue.value
        ))
        return err, InquireValue.value

    def AlpSeqPut(self, devhandle, seqhandle, data, picoffset=0, picload=1):
        """
        Loads the sequence to the ALP.
        Data needs to be an array of Bytes (e.g. char"s) in C,
        values of 0 to 255.

        Example
        -------
        >>> import numpy as np
        >>> import ctypes as ct
        >>> height = 768
        >>> width = 1024
        >>> nBytes = int(width * height)
        >>> C_ByteArray = nBytes * ct.c_uint
        >>> myArray = C_ByteArray()
        >>> testpicture = np.zeros((height, width)).astype(np.uint8)
        >>> testpicture[300:400, 300:400] = (
        ...     np.ones((100, 100)) * 255
        ... ).astype(np.uint8)
        >>> myArray = ct.create_string_buffer(
        ...     testpicture.flatten()[:-1].tostring()
        ... )
        Values > 127 are interpreted as 1 ("on"),
        values < 128 are interpreted as 0 ("off").
        """
        DeviceHandle = ct.c_long(devhandle)
        SeqHandle = ct.c_long(seqhandle)
        PicOffset = ct.c_long(picoffset)
        PicLoad = ct.c_long(picload)
        err = self.dll.AlpSeqPut(
            DeviceHandle, SeqHandle, PicOffset, PicLoad, ct.pointer(data)
        )
        print("AlpSeqPut: {:s}".format(ErrorsFlipped[str(err)]))
        return err

    # +++++++++++++++++++++

    def AlpProjControl(self, devhandle, controltype, controlvalue=0):
        DeviceHandle = ct.c_long(devhandle)
        ControlType = ct.c_long(controltype)
        ControlValue = ct.c_long(controlvalue)
        err = self.dll.AlpProjControl(DeviceHandle, ControlType, ControlValue)
        print("AlpProjControl: {:s}, {:s} is {:s}".format(
            ErrorsFlipped[str(err)],
            ALP_PARMS_PROJ_FLIP[str(controltype)],
            ALP_PARMS_PROJ_FLIP[str(controlvalue)]
        ))
        return err

    def AlpProjInquire(self, devhandle, inquiretype):
        DeviceHandle = ct.c_long(devhandle)
        InquireType = ct.c_long(inquiretype)
        InquireValue = ct.c_long(0)
        err = self.dll.AlpProjInquire(
            DeviceHandle, InquireType, byref(InquireValue)
        )
        print("AlpProjInquire: {:s}, {:s} is {:s}".format(
            ErrorsFlipped[str(err)],
            ALP_PARMS_PROJ_FLIP[str(inquiretype)],
            ALP_PARMS_PROJ_FLIP[str(InquireValue.value)]
        ))
        return err, InquireValue.value

    def AlpProjStart(self, devhandle, seqhandle):
        DeviceHandle = ct.c_long(devhandle)
        SeqHandle = ct.c_long(seqhandle)
        err = self.dll.AlpProjStart(DeviceHandle, SeqHandle)
        print("AlpProjStart: {:s}".format(ErrorsFlipped[str(err)]))
        return err

    def AlpProjStartCont(self, devhandle, seqhandle):
        """Starts project in a continuous loop."""
        DeviceHandle = ct.c_long(devhandle)
        SeqHandle = ct.c_long(seqhandle)
        err = self.dll.AlpProjStartCont(DeviceHandle, SeqHandle)
        print("AlpProjStartCont: {:s}".format(ErrorsFlipped[str(err)]))
        return err

    def AlpProjHalt(self, devhandle):
        """Ends current projection."""
        DeviceHandle = ct.c_long(devhandle)
        err = self.dll.AlpProjHalt(DeviceHandle)
        print("AlpProjHalt: {:s}".format(ErrorsFlipped[str(err)]))
        return err

    def AlpProjWait(self, devhandle):
        """insert help text"""
        DeviceHandle = ct.c_long(devhandle)
        err = self.dll.AlpProjWait(DeviceHandle)
        print("AlpProjWait: {:s}".format(ErrorsFlipped[str(err)]))
        return err


###############################################################################


if __name__ == "__main__":

    height = 768
    width = 1024
    nBytes = int(width * height)
    C_ByteArray = nBytes * ct.c_uint
    myArray = C_ByteArray()

    target_profile = np.zeros((height, width), dtype=np.uint8)
    target_profile[300:450, 450:550] = np.ones((150, 100)).astype(np.uint8)

    myArray = ct.create_string_buffer(target_profile.flatten()[:-1].tostring())

    alp = PY_ALP_API()
    err1, devhandle = alp.AlpDevAlloc()
    err2, seqhandle = alp.AlpSeqAlloc(devhandle, 1, 1)
    print("[DeviceHandle = {:d}, SequenceHandle = {:d}]".format(
        devhandle, seqhandle
    ))

    alp.AlpDevInquire(devhandle, 2051)

    alp.AlpSeqPut(devhandle, seqhandle, data=myArray)

    alp.AlpSeqTiming(devhandle, seqhandle, picturetime=int(5e6))

    alp.AlpProjStart(devhandle, seqhandle)

    print("DMD is currently displaying a pattern.")

    alp.AlpProjWait(devhandle)

    alp.AlpSeqFree(devhandle, seqhandle)
    alp.AlpDevHalt(devhandle)

    # show the test picture
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(np.frombuffer(myArray, dtype="|S1").view(np.uint8)
              .reshape((height, width)), interpolation="nearest")
    plt.show()
