# Import libics_driver as libics.driver for backward compatibility
try:
    from libics_driver import *
    import re
    import sys
    
    _modules = {}
    for k, v in sys.modules.items():
        match = re.match(r"libics_driver\.(\S+)", k)
        if match:
            _modules[f"libics.driver.{match.group(1)}"] = v
    sys.modules.update(_modules)

except ImportError:
    pass
