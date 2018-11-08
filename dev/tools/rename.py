"""
Tool to rename files and folders.
"""
import copy
import os
import re
import sys

from PyQt5.Qt import Qt
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QAbstractScrollArea,
    QPushButton, QLineEdit, QLabel, QFileDialog,
    QTableWidget, QTableWidgetItem
)

from libics.util import misc


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


def replace_names(folder, regex, name_ls, format_ls, replace_ls, rename=True):
    """
    Parameters
    ----------
    folder : str
        Folder where files/subfolders are located.
    regex : str
        Regular expression matching the part of the name
        that should be replaced.
        Regex brackets are automatically appended `(`, `)`.
    name_ls : list(str)
        File/subfolder names which should be replaced.
    format_ls : list(str) or str
        Formatter strings for replacement. If only one
        string is given, it will be applied to all items.
    replace_ls : list or obj
        Formatter string insertions.
        Tuples are flattened and passed to the formatter.
        If only one item is given, it will be applied to all
        items.
    rename : bool
        Flag whether to apply the renaming procedure.

    Returns
    -------
    new_name_ls : list(str)
        List of replaced file/subfolder names.

    Raises
    -----
    AssertionError
        If name_ls, format_ls, replace_ls do not have the same lengths.

    Notes
    -----
    If the replacement regex could not be matched, the name is not changed.
    """
    if not isinstance(format_ls, list):
        format_ls = len(name_ls) * [format_ls]
    if not isinstance(replace_ls, list):
        replace_ls = len(name_ls) * [replace_ls]
    assert(len(name_ls) == len(format_ls) == len(replace_ls))
    new_name_ls = copy.deepcopy(name_ls)
    for i, name in enumerate(name_ls):
        try:
            match = re.search("(^.*)(" + regex + ")(.*)", name)
            if match:
                new_name_ls[i] = ("{:s}" + format_ls[i] + "{:s}").format(
                    match.group(1),
                    *misc.assume_tuple(replace_ls[i]),
                    match.group(3)
                )
        except re.error:
            pass
    if rename:
        for i, name in enumerate(name_ls):
            os.rename(
                os.path.join(folder, name),
                os.path.join(folder, new_name_ls[i])
            )
    return new_name_ls


def _rename(folder, old_name_ls, new_name_ls):
    """
    Renames the file names specified in `old_name_ls` into the corresponding
    `new_name_ls` items. Both paths will be applied in the same `folder`.

    Parameters
    ----------
    folder : str
        Folder where files/subfolders are located.
    old_name_ls : list(str)
        File/subfolder names which should be replaced.
    new_name_ls : list(str)
        Renamed file/subfolder names (corresponding to item number in
        `old_name_ls`).
    """
    assert(len(old_name_ls) == len(new_name_ls))
    for i, name in enumerate(old_name_ls):
        os.rename(
            os.path.join(folder, name),
            os.path.join(folder, new_name_ls[i])
        )


###############################################################################


class RenameWidget(QWidget):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.setup_logic()
        self.setup_ui()
        self.setup_connection()

    def setup_logic(self):
        self.regex_filter = ""          # Regex: filter folder items
        self.regex_replace = ""         # Regex: replacement
        self.format_replace = "{:s}"    # Format: replacement
        self.value_replace = ""         # Value: for replacement formatter
        self.match_file_ls = []         # List: matched file names
        self.match_folder_ls = []       # List: matched subfolder names
        self.rest_file_ls = []          # List: unmatched file names
        self.rest_folder_ls = []        # List: unmatched subfolder names
        self.new_file_ls = []           # List: replaced file names
        self.new_folder_ls = []         # List: replaced folder names

    def setup_ui(self):
        # Window
        self.setWindowTitle("Rename files and subfolders")

        # Set folder button
        self.qbutton_set_folder = QPushButton("Set folder")
        # Apply renaming button
        self.qbutton_apply_rename = QPushButton("Apply renaming")
        # Controls layout
        self.q_control = QHBoxLayout()
        self.q_control.addWidget(self.qbutton_set_folder)
        self.q_control.addWidget(self.qbutton_apply_rename)

        # Regex filter
        self.q_regex_filter = QVBoxLayout()
        self.qlabel_regex_filter = QLabel("Filter regex:")
        self.qline_regex_filter = QLineEdit()
        self.q_regex_filter.addWidget(self.qlabel_regex_filter)
        self.q_regex_filter.addWidget(self.qline_regex_filter)
        # Regex replace
        self.q_regex_replace = QVBoxLayout()
        self.qlabel_regex_replace = QLabel("Replace regex:")
        self.qline_regex_replace = QLineEdit()
        self.q_regex_replace.addWidget(self.qlabel_regex_replace)
        self.q_regex_replace.addWidget(self.qline_regex_replace)
        # Value replace
        self.q_value_replace = QVBoxLayout()
        self.qlabel_value_replace = QLabel("Replace value:")
        self.qline_value_replace = QLineEdit()
        self.q_value_replace.addWidget(self.qlabel_value_replace)
        self.q_value_replace.addWidget(self.qline_value_replace)
        # Lineedit layout
        self.q_lineedit = QHBoxLayout()
        self.q_lineedit.addLayout(self.q_regex_filter)
        self.q_lineedit.addLayout(self.q_regex_replace)
        self.q_lineedit.addLayout(self.q_value_replace)

        # Matched and new folder
        self.qlist_folder = QTableWidget()
        self.qlist_folder.setColumnCount(2)
        self.qlist_folder.setHorizontalHeaderLabels(
            ("Old folder name", "New folder name")
        )
        self.qlist_folder.setSizeAdjustPolicy(
            QAbstractScrollArea.AdjustToContents
        )
        self.qlist_folder.resizeColumnsToContents()
        # Matched and new file
        self.qlist_file = QTableWidget()
        self.qlist_file.setColumnCount(2)
        self.qlist_file.setHorizontalHeaderLabels(
            ("Old file name", "New file name")
        )
        self.qlist_file.setSizeAdjustPolicy(
            QAbstractScrollArea.AdjustToContents
        )
        self.qlist_file.resizeColumnsToContents()
        # Unmatched folder
        self.qlist_rest_folder = QTableWidget()
        self.qlist_rest_folder.setColumnCount(1)
        self.qlist_rest_folder.setHorizontalHeaderLabels(
            ("Unmatched folder", )
        )
        self.qlist_rest_folder.setSizeAdjustPolicy(
            QAbstractScrollArea.AdjustToContents
        )
        self.qlist_rest_folder.resizeColumnsToContents()
        # Unmatched file
        self.qlist_rest_file = QTableWidget()
        self.qlist_rest_file.setColumnCount(1)
        self.qlist_rest_file.setHorizontalHeaderLabels(
            ("Unmatched file", )
        )
        self.qlist_rest_file.setSizeAdjustPolicy(
            QAbstractScrollArea.AdjustToContents
        )
        self.qlist_rest_file.resizeColumnsToContents()
        # Folder/file list
        self.q_list = QHBoxLayout()
        self.q_list.addWidget(self.qlist_rest_folder)
        self.q_list.addWidget(self.qlist_folder)
        self.q_list.addWidget(self.qlist_rest_file)
        self.q_list.addWidget(self.qlist_file)

        # Main layout
        self.q_main = QVBoxLayout()
        self.q_main.addLayout(self.q_control)
        self.q_main.addLayout(self.q_lineedit)
        self.q_main.addLayout(self.q_list)
        self.setLayout(self.q_main)

    def setup_connection(self):
        self.qbutton_set_folder.clicked.connect(
            self.on_qbutton_set_folder_clicked
        )
        self.qbutton_apply_rename.clicked.connect(
            self.on_qbutton_apply_rename_clicked
        )
        self.qline_regex_filter.textChanged.connect(
            self.on_qline_regex_filter_textchanged
        )
        self.qline_regex_replace.textChanged.connect(
            self.on_qline_regex_replace_textchanged
        )
        self.qline_value_replace.textChanged.connect(
            self.on_qline_value_replace_textchanged
        )

    # +++++++++++++++++++++++++++++++++

    def __set_table_contents(self):
        self.qlist_folder.setRowCount(len(self.match_folder_ls))
        self.qlist_file.setRowCount(len(self.match_file_ls))
        self.qlist_rest_folder.setRowCount(len(self.rest_folder_ls))
        self.qlist_rest_file.setRowCount(len(self.rest_file_ls))
        for i, item in enumerate(self.rest_folder_ls):
            qtwi = QTableWidgetItem(item)
            qtwi.setFlags(Qt.NoItemFlags)
            self.qlist_rest_folder.setItem(i, 0, qtwi)
        for i, item in enumerate(self.rest_file_ls):
            qtwi = QTableWidgetItem(item)
            qtwi.setFlags(Qt.ItemIsSelectable)
            self.qlist_rest_file.setItem(i, 0, qtwi)
        for i, item in enumerate(self.match_folder_ls):
            qtwi = QTableWidgetItem(item)
            qtwi.setFlags(Qt.ItemIsEnabled)
            self.qlist_folder.setItem(i, 0, qtwi)
            qtwi = QTableWidgetItem(self.new_folder_ls[i])
            qtwi.setFlags(Qt.ItemIsSelectable)
            self.qlist_folder.setItem(i, 1, qtwi)
        for i, item in enumerate(self.match_file_ls):
            qtwi = QTableWidgetItem(item)
            qtwi.setFlags(Qt.ItemIsSelectable)
            self.qlist_file.setItem(i, 0, qtwi)
            qtwi = QTableWidgetItem(self.new_file_ls[i])
            qtwi.setFlags(Qt.ItemIsSelectable)
            self.qlist_file.setItem(i, 1, qtwi)
        self.qlist_rest_folder.resizeColumnsToContents()
        self.qlist_rest_file.resizeColumnsToContents()
        self.qlist_folder.resizeColumnsToContents()
        self.qlist_file.resizeColumnsToContents()

    @pyqtSlot()
    def on_qbutton_set_folder_clicked(self):
        # UI
        dialog = QFileDialog()
        folder = dialog.getExistingDirectory(
            caption="Choose folder whose items should be renamed"
        )
        if folder == "":
            return
        self.folder = folder
        # Logic
        self.on_qline_regex_filter_textchanged("")

    @pyqtSlot()
    def on_qbutton_apply_rename_clicked(self):
        _rename(self.folder, self.match_folder_ls, self.new_folder_ls)
        _rename(self.folder, self.match_file_ls, self.new_file_ls)
        self.on_qline_regex_filter_textchanged(self.regex_filter)

    @pyqtSlot(str)
    def on_qline_regex_filter_textchanged(self, s):
        self.regex_filter = s
        _fc = get_folder_contents(self.folder, regex=self.regex_filter)
        self.match_folder_ls, self.match_file_ls = _fc[2], _fc[3]
        self.rest_folder_ls, self.rest_file_ls = _fc[4], _fc[5]
        self.on_qline_textchanged()

    @pyqtSlot(str)
    def on_qline_regex_replace_textchanged(self, s):
        self.regex_replace = s
        self.on_qline_textchanged()

    @pyqtSlot(str)
    def on_qline_value_replace_textchanged(self, s):
        self.value_replace = s
        self.on_qline_textchanged()

    def on_qline_textchanged(self):
        self.new_folder_ls = replace_names(
            self.folder, self.regex_replace, self.match_folder_ls,
            self.format_replace, self.value_replace, rename=False
        )
        self.new_file_ls = replace_names(
            self.folder, self.regex_replace, self.match_file_ls,
            self.format_replace, self.value_replace, rename=False
        )
        self.__set_table_contents()


###############################################################################


def main():
    app = QApplication(sys.argv)
    rename_widget = RenameWidget()
    rename_widget.show()
    app_ret = app.exec_()
    return app_ret


if __name__ == "__main__":
    main()
