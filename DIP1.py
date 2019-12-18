import os
from os import path
import scipy.io as scio
from scipy import fftpack
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d as conv2d
import cv2


class Censoring:
    def __init__(self, motionPath):
        self.PSFS = []
        self.zippedVals = lambda: zip(scio.loadmat(motionPath)['X'], scio.loadmat(motionPath)['Y'])


    def plotTrajectories(self):
        if not path.exists('trajectories'):
            os.mkdir('trajectories')
        for fileNumber, coordinate in enumerate(self.zippedVals()):
            plt.figure()
            plt.plot(coordinate[0], coordinate[1])
            plt.savefig(path.join('trajectories', str(fileNumber) + '.png'))
            plt.close()

    def PSFsCreate(self):
        for x, y in self.zippedVals():
            psfSize = 11
            psf = np.zeros((psfSize, psfSize))
            center = (psfSize - 1) / 2
            for Xi, Yi in zip(x, y):
                psfRow = round(center + Xi)
                psfCol = round(center - Yi)
                if psfCol < psfSize and psfRow < psfSize:
                    psf[int(psfCol), int(psfRow)] += 1

            self.PSFS.append(psf)


class blurred:
    def __init__(self, originalImage, psfs):
        self.psfs = psfs
        self.imgs = [conv2d(originalImage, mat, 'same') for mat in psfs]
        self.plot()

    def plot(self):
        if not path.exists('PSF'):
            os.mkdir('PSF')
        if not path.exists('blurred_images'):
            os.mkdir('blurred_images')
        for i, mat in enumerate(self.psfs):
            fig, axs = plt.subplots(1, 1, constrained_layout=True)
            name = str(i) + '.png'
            axs.imshow(self.imgs[i], cmap='gray')
            plt.savefig(path.join('blurred_images', name))
            axs.imshow(mat, cmap='gray')
            plt.savefig(path.join('PSF', name))
            plt.close()


class deblurred:
    def __init__(self, blurredImages):
        self.fft_box = [fftpack.fftn(blurredImages[i]) for i in range(100)]
        self.imgU = np.zeros((256, 256)).astype(complex)
        self.W = np.zeros((256, 256)).astype(float)

    def calcAndPlot(self,i, p=13):
        amp = np.abs(self.fft_box[i])
        Wi = np.mean(amp)
        Wi = Wi ** p
        self.imgU += self.fft_box[i] * Wi
        self.W += Wi
        res = np.abs(fftpack.ifftn(self.imgU / self.W))
        fig, axs = plt.subplots(1, 1, constrained_layout=True)
        axs.imshow(res, cmap='gray')
        plt.savefig(path.join('reconstructed', str(i) + '.png'))
        plt.close()
        return res

    def reconstruct(self):
        if not path.exists('reconstructed'):
            os.mkdir('reconstructed')
        return [self.calcAndPlot(index) for index in range(100)]


def plotPSNRS(images, originalImage):
    psnrs = []
    for fixed in images:
        MSE = np.mean(np.power(np.abs(np.subtract(np.divide(np.array(fixed), 1000), originalImage)), 2))
        psnrs.append(20 * np.log10(255) - 10 * np.log10(MSE))
    plt.figure()
    plt.title('PSNR vs number of frames')
    plt.xlabel("Number of Images")
    plt.ylabel("PSNR value")
    plt.plot(psnrs)
    plt.savefig('psnr_graph.png')
    plt.close()

def main():
    blur = Censoring('100_motion_paths.mat')
    blur.plotTrajectories()
    blur.PSFsCreate()
    originalImage = cv2.imread('DIPSourceHW1.jpg', 0)
    plotPSNRS(deblurred(blurred(originalImage, blur.PSFS).imgs).reconstruct(), originalImage)

if __name__ == '__main__':
    main()