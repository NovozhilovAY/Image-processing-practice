import cv2
import numpy as np

img = cv2.imread("D:\\hm\\1\\00_75.png", 0)
rows, cols = img.shape[:2]
d0 = 70
nrows = cv2.getOptimalDFTSize(rows)
ncols = cv2.getOptimalDFTSize(cols)
nimg = np.zeros((nrows, ncols))
nimg[:rows,:cols] = img
fft_mat = cv2.dft(np.float32(nimg), flags = cv2.DFT_COMPLEX_OUTPUT)
fft_mat = np.fft.fftshift(fft_mat)

def fft_distances(m, n):
    u = np.array([i - m/2 for i in range(m)], dtype=np.float32)
    v = np.array([i - n/2 for i in range(n)], dtype=np.float32)
    ret = np.ones((m, n))
    for i in range(m):
        for j in range(n):
            ret[i][j] = np.sqrt(u[i]*u[i] + v[j]*v[j])
    u = np.array([i if i<=m/2 else m-i for i in range(m)], dtype=np.float32)
    v = np.array([i if i<=m/2 else m-i for i in range(m)], dtype=np.float32)
    return ret

def change_filter(flag):
    if flag == 1:
        filter_mat = np.zeros((nrows, ncols ,2), np.float32)
        cv2.circle(filter_mat, (np.int(ncols/2), np.int(nrows/2)) , d0, (1,1,1), -1)
    elif flag == 2:
        n = 2
        filter_mat = None
        duv = fft_distances(*fft_mat.shape[:2])
        filter_mat = 1 / (1+ np.power(duv/d0, 2*n))
        filter_mat = cv2.merge((filter_mat, filter_mat))

    else:
        filter_mat = None
        duv = fft_distances(*fft_mat.shape[:2])
        filter_mat = np.exp(-(duv*duv) / (2*d0*d0))
        filter_mat = cv2.merge((filter_mat, filter_mat))
    return filter_mat

def ifft(fft_mat):
    f_ishift_mat = np.fft.ifftshift(fft_mat)
    img_back = cv2.idft(f_ishift_mat)
    img_back = cv2.magnitude(*cv2.split(img_back))
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(np.around(img_back))[:rows,:cols]

img1 = ifft(change_filter(1) * fft_mat)
img2 = ifft(change_filter(2) * fft_mat)
img3 = ifft(change_filter(3) * fft_mat)

cv2.imshow("orig", img)
cv2.imshow("img1", img1)
cv2.imshow("img2", img2)
cv2.imshow("img3", img3)
cv2.waitKey()



