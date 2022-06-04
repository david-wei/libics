# LibICS

LibICS consists of two subpackages:
* `core` provides general purpose data structures and I/O,
* `tools` contains a set of analysis functions.

The full documentation can be found [here](https://david-wei.github.io/libics).

## Installation

Clone this repository:

```
git clone https://www.github.com/david-wei/libics.git
```

Install the library:

```
pip install ./libics
```

## Dependencies

Libraries providing extensions (or actual implementations) that not
necessarily all users need have to be installed separately. This chapter
gives an overview of these additional external dependencies.

## Files

| Library | Description | File types [read/write] | Installation |
| ------- | ----------- | ----------------------- | ------------ |
| [pymongo](https://github.com/mongodb/mongo-python-driver) | MongoDB database | [r/w] Binary JSON files (.bson) | `pip install pymongo` |
| [sif_reader](https://github.com/fujii-team/sif_reader) | SIF file reader | [r] Singularity image files (.sif) | `pip install git+https://github.com/fujii-team/sif_reader.git` |
