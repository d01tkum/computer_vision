# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:12:17 2019

@author: Takuma Doi
"""

from matplotlib import pyplot as plt
import numpy as np
from optparse import OptionParser
import os
from PIL import Image, ImageOps
import sys


def im_equalize(f_path, fname):
    file_path = os.path.join(f_path, fname)

    im = Image.open(file_path)
    im = im.convert("YCbCr")
    yy, cb, cr = im.split()

    yy = ImageOps.equalize(yy)
    im = Image.merge("YCbCr", (yy, cb, cr))

    return np.array(im.convert("RGB"))


def _gamma_correction(im, gamma):
    arr = np.asarray(im)
    arr = 255.0 * (arr / 255.0)**(gamma)
    return Image.fromarray(np.uint8(arr))


def im_gamma_equalize(f_path, fname):
    file_path = os.path.join(f_path, fname)

    # 画像ファイルを取り込んで RGB を確保。
    im = Image.open(file_path)
    im = im.convert("RGB")

    # ガンマ補正を解除して RGB=>YCbCr 変換
    # Y だけガンマ補正をかけ直す。
    im = _gamma_correction(im, 1 / 2.2)
    im = im.convert("YCbCr")
    yy, cb, cr = im.split()  # linear yy,cc,cr
    yy = _gamma_correction(yy, 2.2)

    # 輝度に対してヒストグラム平坦化
    yy = ImageOps.equalize(yy)

    # Y のガンマ補正を解除して YCbCR=>RGB 変換
    # RGB にガンマ補正を掛け直す
    yy = _gamma_correction(yy, 1 / 2.2)
    im = Image.merge("YCbCr", (yy, cb, cr))
    im = im.convert("RGB")

    return np.array(_gamma_correction(im, 2.2))


if __name__ == "__main__":

    optparser = OptionParser()
    optparser.add_option('-inf', '--inputFile',
                         dest='input',
                         help='filename',
                         default=None)
    optparser.add_option('-outf', '--outputFile',
                         dest='output',
                         help='filename',
                         default=None)

    (options, args) = optparser.parse_args()
    inFile = None
    outFile = None
    if options.input is None:
        inFile = sys.argv[1]
        assert inFile is None, 'please set input file'
    if options.output is None:
        outFile = sys.argv[2]
        assert outFile is None, 'please set outout file'

    for index in [round(0.1 * x, 2) for x in range(1, 11)]:
        # Define input directory's path
        data_dir_path = os.path.join(
            "D:",
            "backup20190318",
            "workspace",
            "leaf_number=16",
            "normal_images_envlight="
        )
        data_dir_path = data_dir_path + str(index)
        print("\n--------------------\n", data_dir_path)

        # Load images in the directory
        im_list = [data for data in os.listdir(
            data_dir_path) if ".png" in data]

        # Define output directory's path
        out_dir_path = data_dir_path + "-eq"
        if not os.path.exists(out_dir_path):
            os.makedirs(out_dir_path)

        # Perform histogram equalization
        for im_name in im_list:
            im_eq = im_equalize(data_dir_path, im_name)
#            im_eq = im_gamma_equalize(data_dir_path, im_name)

            out_name = im_name.rstrip(".png") + "-eq" + ".png"
            out_path = os.path.join(
                out_dir_path, out_name
            )

            print(out_path)
            plt.imsave(out_path, im_eq, format='png')
