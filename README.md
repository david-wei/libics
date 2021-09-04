# LibICS

LibICS consists of three sub-libraries:
* `core` provides general purpose data structures and I/O,
* `driver` allows accessing drivers to communicate with external devices and
* `tools` contains a set of analysis functions.

The full documentation can be found [here](https://david-wei.github.io/libics).

# Dependencies

Python dependencies required for the library functionality are specified
in the [setup](./setup.py) file, so package managers can automatically
install them.

Libraries providing extensions (or actual implementations) that not
necessarily all users need have to be installed separately. This chapter
gives an overview of these additional external dependencies.

## Drivers

### Interfaces

| Developer | Library | Type | Description |
| --------- | ------- | ---- | ----------- |
| [libusb](https://libusb.info) | [libusb 1.0](https://github.com/libusb/libusb/releases) | C API | Windows USB library, DLL needs to be on system path |


### Cameras

| Developer | Library | Type | Description |
| --------- | ------- | ---- | ----------- |
| [AlliedVision](https://www.alliedvision.com) | [Vimba](https://www.alliedvision.com/en/products/software.html) | C API | communication with AlliedVision Vimba cameras, i.a. Manta G-145B-NIR |
| [Github: morefigs](https://github.com/morefigs) | [pymba](https://github.com/morefigs/pymba) | Python wrapper | communication with AlliedVision Vimba cameras, i.a. Manta G-145B-NIR |


## Files

| Developer | Library | Type | Description |
| --------- | ------- | ---- | ----------- |
| [Github: fujii-team](https://github.com/fujii-team) | [sif_reader](https://github.com/fujii-team/sif_reader) | Python wrapper | R/W singularity image files (.sif) |
