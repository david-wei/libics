import re
import os


###############################################################################


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
    folder : list(str)
        Full list of subfolders.
    files : list(str)
        Full list of files within folder.
    folders_match : list(str)
        List of subfolders matching regex.
    files_match : list(str)
        List of files matching regex.
    folders_rest : list(str)
        List of subfolders not matching regex.
    files_rest : list(str)
        List of files not matching regex.
    """
    if regex == "":
        regex = None
    folders, folders_match, folders_rest = [], [], []
    files, files_match, files_rest = [], [], []
    for item in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, item)):
            files.append(item)
            try:
                if regex is not None and re.search(regex, item):
                    files_match.append(item)
                else:
                    files_rest.append(item)
            except re.error:
                files_rest.append(item)
        else:
            folders.append(item)
            try:
                if regex is not None and re.search(regex, item):
                    folders_match.append(item)
                else:
                    folders_rest.append(item)
            except re.error:
                folders_rest.append(item)
    return folders, files, folders_match, files_match, folders_rest, files_rest
