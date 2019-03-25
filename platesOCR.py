import cv2
import tesserocr
import numpy as np
from matplotlib import pyplot as plt
import glob
import os
import pytesseract
import string

plates = glob.glob("plates\\*.png")
processed = glob.glob("processed\\*.png")
resized = glob.glob("resized\\*.png")
bordered = glob.glob("borders\\*.png")


def adaptiveThreshold(plates):
    for i, plate in enumerate(plates):
        img = cv2.imread(plate)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cv2.imshow('gray', gray)

        ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        # cv2.imshow('thresh', thresh)

        threshMean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 10)
        # cv2.imshow('threshMean', threshMean)

        threshGauss = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 27)
        cv2.imshow('threshGauss', threshGauss)
        cv2.imwrite("processed\\plate{}.png".format(i), threshGauss)

        cv2.waitKey(0)


def resize(processed):
    for i, image in enumerate(processed):
        image = cv2.imread(image)

        ratio = 200.0 / image.shape[1]
        dim = (200, int(image.shape[0] * ratio))

        resizedLinear = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)

        cv2.imwrite("resized\\plate{}.png".format(i), resizedLinear)


def addBorder(resized):
    for i, image in enumerate(resized):
        image = cv2.imread(image)

        bordersize = 10
        border = cv2.copyMakeBorder(image, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize,
                                    borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])

        cv2.imwrite("borders\\plate{}.png".format(i), border)


def cleanOCR(borders):
    detectedOCR = []

    for i, image in enumerate(borders):
        image = cv2.imread(image)

        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 180, threshold=100, lines=np.array([]),
                                minLineLength=100, maxLineGap=80)

        a, b, c = lines.shape
        for i in range(a):
            x = lines[i][0][0] - lines[i][0][2]
            y = lines[i][0][1] - lines[i][0][3]
            if x != 0:
                if abs(y / x) < 1:
                    cv2.line(image, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (255, 255, 255),
                             1, cv2.LINE_AA)

        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        gray = cv2.morphologyEx(image, cv2.MORPH_CLOSE, se)

        # OCR
        config = '-l eng --oem 1 --psm 3'
        text = pytesseract.image_to_string(gray, config=config)

        validChars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                      'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        cleanText = []

        for char in text:
            if char in validChars:
                cleanText.append(char)

        plate = ''.join(cleanText)
        # print(plate)

        detectedOCR.append(plate)

        # cv2.imshow('img', gray)
        # cv2.waitKey(0)

    return detectedOCR


# adaptiveThreshold(plates)
# resize(processed)
# addBorder(resized)
platesList = cleanOCR(bordered)

print(platesList)
