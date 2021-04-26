from google_drive_downloader import GoogleDriveDownloader as google_drive
from multiprocessing import Pool
from rich.progress import *
import numpy as np
import argparse
import zipfile
import cv2
import os

def unzip(src, dst, desc='Extracting'):
    progress = Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        FileSizeColumn(),
        'of',
        TotalFileSizeColumn()
    )

    with zipfile.ZipFile(src) as zf:
        size = sum([zinfo.file_size for zinfo in zf.filelist])

        with progress:
            task = progress.add_task(desc, total = size)
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

def work(args):
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

    progress = Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
    )

    filenames = os.listdir(root)
    filecount = len(filenames)

    with progress:
        with Pool(workers) as pool:
            list(progress.track(pool.imap(work, ((root, dest, name, size) for name in filenames)), total = filecount, description = f'Writing to {dest}'))
