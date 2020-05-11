# Bugfix
# Weird bug where importing vrmusbcamapi before usb prevents proper
# functioning of libusb.
import usb.core
usb.core.find(find_all=True)
# END Bugfix

from . import alpV42        # noqa
from . import vrmusbcamapi  # noqa
