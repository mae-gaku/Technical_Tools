import os
import zipfile


def unzip(file):
    filename = file.filename
    file.save(filename)
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall()
        os.remove(filename)
        files = os.listdir()

    return file
    