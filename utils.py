from google_drive_downloader import GoogleDriveDownloader as google_drive
from multiprocessing import Pool
from rich.progress import *
import numpy as np
import argparse
import zipfile
import cv2
import os

def pbar(spinner=True, description=True, bar=True, percentage=True, time=True, filesize=False, total_filesize=False, count=False, units='items'):
    progress = []

    if spinner:
        progress.append(SpinnerColumn())

    if description:
        progress.append("[progress.description]{task.description}")

    if bar:
        progress.append(BarColumn())

    if percentage:
        progress.append("[progress.percentage]{task.percentage:>3.0f}%")

    if time:
        progress.append(TimeRemainingColumn())

    if filesize:
        progress.append(FileSizeColumn())

    if total_filesize:
        progress.extend(['of', TotalFileSizeColumn()])

    if count:
        progress.append("[progress.completed]{task.completed} %s" % units)

    return progress

def unzip(src, dst):
    progress = pbar(filesize = True, total_filesize = True)

    with zipfile.ZipFile(src) as zf:
        size = sum([zinfo.file_size for zinfo in zf.filelist])

        with Progress(*progress) as progress:
            task = progress.add_task(f'Extracting to {dst}', total = size)
            for member in zf.infolist():
                try:
                    zf.extract(member, dst)
                except zipfile.error:
                    pass
                progress.advance(task, member.file_size)

dataset2fileID = {
    'MontezumaRevenge': '10g_7NGs43eX_PyeeqgvZQGrkYNDU1ldB',
    'Breakout':         '1KHJ3i_SDQVICT4nyuLtqkELjUo7G9zFl',
    'Pitfall':          '187sSHqOSoLRNFE4_UvdsqjBL6uPpJ57k',
    'Qbert':            '1039o1Z40_8nZbf_XM_cxZWpbSjKZRk0Y',
}

def download(dataset, destination):
    if dataset in dataset2fileID:
        fileID = dataset2fileID[dataset]
        google_drive.download_file_from_google_drive(file_id = fileID, dest_path = destination, unzip = False)
    else:
        expected = '/'.join(list(dataset2fileID.keys()))
        raise ValueError(f'Expected `dataset` to be one of {expected}, but found {dataset}')

def _work(args):
    root, dest, name, size = args

    if name.endswith('.jpeg'):
        src = os.path.join(root, name)
        dst = os.path.join(dest, name)

        img = cv2.imread(src)

        if not img is None:
            res = cv2.resize(img, (size, size), interpolation = cv2.INTER_AREA)
            cv2.imwrite(dst, res)

def resize(root, mode='move', workers=64, size=160):
    if mode == 'move':
        dest = os.path.join(os.path.dirname(root), 'resized')
        os.mkdir(dest)
        dest = os.path.join(dest, 'cells')
        os.mkdir(dest)
    elif mode == 'replace':
        dest = root
    else:
        raise ValueError(f'Expected write mode `mode` to be one of move/replace, but found {mode}')

    progress = pbar()

    filenames = os.listdir(root)
    filecount = len(filenames)

    with progress:
        with Pool(workers) as pool:
            list(progress.track(pool.imap(_work, ((root, dest, name, size) for name in filenames)), total = filecount, description = f'Resizing to {dest}'))
