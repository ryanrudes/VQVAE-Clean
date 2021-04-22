from multiprocessing import Pool
from rich.progress import *
import numpy as np
import argparse
import cv2
import os

def work(args):
    root, dest, name, size = args

    if name.endswith('.jpeg'):
        src = os.path.join(root, name)
        dst = os.path.join(dest, name)

        img = cv2.imread(src)

        if not img is None:
            res = cv2.resize(img, (size, size), interpolation = cv2.INTER_AREA)
            cv2.imwrite(dst, res)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--directory', '-d', type=str)
    parser.add_argument('--writemode', '-w', type=str, choices=['move', 'replace'], default='move')
    parser.add_argument('--workers', type=int, default=64)
    parser.add_argument('--size', '-s', type=int, default=160)
    args = parser.parse_args()

    root = args.directory
    mode = args.writemode
    size = args.size
    if mode == 'move':
        dest = os.path.join(os.path.dirname(args.directory), 'resized')
        os.mkdir(dest)
        dest = os.path.join(dest, 'cells')
        os.mkdir(dest)
    else:
        dest = args.directory

    progress = Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
    )

    filenames = os.listdir(root)
    filecount = len(filenames)

    print ('Writing to:', dest)

    with progress:
        with Pool(args.workers) as pool:
            list(progress.track(pool.imap(work, ((root, dest, name, size) for name in filenames)), total = filecount))
