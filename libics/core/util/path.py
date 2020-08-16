import re
import os


###############################################################################


class FolderContentsResult(object):

    """
    Results container for :py:func:`get_folder_contents`.

    Attributes
    ----------
    regex : `str`
        Matching regex.
    path : `str`
        Parent directory path used for search.
    folders : `list(str)`
        Full list of subfolders.
    files : `list(str)`
        Full list of files within folder.
    folders_matched : `list(str)`
        List of matched folders.
    files_matched : `list(str)`
        List of matched files.
    folders_unmatched : `list(str)`
        List of unmatched folders.
    files_unmatched : `list(str)`
        List of unmatched files.
    """

    def __init__(self, **kwargs):
        self.regex = ""
        self.path = ""
        self.folders_matched = []
        self.files_matched = []
        self.folders_unmatched = []
        self.files_unmatched = []
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def folders(self):
        return self.folders_matched + self.folders_unmatched

    @property
    def files(self):
        return self.files_matched + self.files_unmatched

    def __iter__(self):
        for item in [
            self.folders, self.files, self.folders_matched, self.files_matched,
            self.folders_unmatched, self.files_unmatched
        ]:
            yield item

    def __getitem__(self, key):
        return list(self)[key]


def get_folder_contents(folder, regex=None):
    """
    Gets the files/subfolders in a folder.

    Parameters
    ----------
    folder : str
        Folder in which to search for files/subfolders.
    regex : str
        Regular expression that is checked.

    Returns
    -------
    res : `FolderContentsResult`
        Result container.
    """
    if regex == "":
        regex = None
    folders_matched, folders_unmatched = [], []
    files_matched, files_unmatched = [], []
    for item in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, item)):
            try:
                if regex is not None and re.search(regex, item):
                    files_matched.append(item)
                else:
                    files_unmatched.append(item)
            except re.error:
                files_unmatched.append(item)
        else:
            try:
                if regex is not None and re.search(regex, item):
                    folders_matched.append(item)
                else:
                    folders_unmatched.append(item)
            except re.error:
                folders_unmatched.append(item)
    return FolderContentsResult(
        regex=regex, path=folder,
        folders_matched=folders_matched, files_matched=files_matched,
        folders_unmatched=folders_unmatched, files_unmatched=files_unmatched
    )


def assume_file_exists(file_path):
    """
    Checks if a file exists. Creates an empty file if not.

    Parameters
    ----------
    file_path : str
        Path to the file to be checked.

    Returns
    -------
    existed : bool
        True if file has existed before.
    """
    if os.path.exists(file_path):
        return True
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    open(file_path, "a").close()
    return False
