import cv2
import numpy as np
import matplotlib.pyplot as plt

def BGR2GRAY(img):
    gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
    return gray

def Gabor_filter(K_size=111, Sigma=7, Gamma=1.2, Lambda=10, Psi=0, angle=0):
    d = K_size // 2
    gabor = np.zeros((K_size, K_size), dtype=np.float32)

    for y in range(K_size):
        for x in range(K_size):
            px = x - d
            py = y - d
            theta = angle / 180. * np.pi
            _x = np.cos(theta) * px + np.sin(theta) * py
            _y = -np.sin(theta) * px + np.cos(theta) * py
            gabor[y, x] = np.exp(-(_x ** 2 + Gamma ** 2 * _y ** 2) / (2 * Sigma ** 2)) * np.cos(
                2 * np.pi * _x / Lambda + Psi)
    gabor /= np.sum(np.abs(gabor))
    return gabor

def Gabor_filtering(gray, K_size=111, Sigma=7, Gamma=1.2, Lambda=10, Psi=0, angle=0):
    H, W = gray.shape
    out = np.zeros((H, W), dtype=np.float32)
    gabor = Gabor_filter(K_size=K_size, Sigma=Sigma, Gamma=Gamma, Lambda=Lambda, Psi=0, angle=angle)
    out = cv2.filter2D(gray, -1, gabor)
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)
    return out

def Gabor_process(img):
    H, W, _ = img.shape
    gray = BGR2GRAY(img).astype(np.float32)
    As = [0, 30, 60, 90, 120, 150]
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0.2)
    out = np.zeros([H, W], dtype=np.float32)
    for i, A in enumerate(As):
        _out = Gabor_filtering(gray, K_size=11, Sigma=1.5, Gamma=1.2, Lambda=3, angle=A)
        out += _out
        plt.imshow(_out, cmap='gray')
        plt.show()

    out = out / out.max() * 255
    out = out.astype(np.uint8)
    return out

img = cv2.imread(r'D:\hm\otp.PNG').astype(np.float32)
img1 = cv2.imread(r'D:\hm\otp.PNG')
def main():
    cv2.imshow("orig", img1)

    out = Gabor_process(img)
    cv2.imshow("result",cv2.bitwise_not( out))

    thresh = 25
    img_binary = cv2.threshold(out, thresh, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("result_b-w", img_binary)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()