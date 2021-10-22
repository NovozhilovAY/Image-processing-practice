import cv2 as cv
import numpy as np
def calc_of_damage_and_non_damage(image):
    hsv_img = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    markers = np.zeros((image.shape[0], image.shape[1]), "int32")
    markers[90:140, 90:140] = 255
    markers[236:255, 0:20] = 1
    markers[0:20, 0:20] = 1
    markers[0:20, 236:255] = 1
    markers[236:255, 236:255] = 1

    leafs_area_BGR = cv.watershed(image, markers)
    healthy_part = cv.inRange(hsv_img, (33, 25, 25), (86, 255, 255))
    ill_part = leafs_area_BGR - healthy_part
    mask = np.zeros_like(image, np.uint8)
    mask[leafs_area_BGR > 1] = (0, 255, 0)
    mask[ill_part > 1] = (0, 0, 255)
    return mask

def bilateral_filter(image):
    return cv.bilateralFilter(image, 15, 75, 75)

def gaussian_filter(image):
    return cv.GaussianBlur(image, (7, 7), cv.BORDER_DEFAULT)

def erode_filter(image):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    return cv.erode(image, kernel)

def get_result_image(image, mask):
    return cv.add(image, mask)


path = "C:\\Users\\sasha\\PycharmProjects\\ImageProcessingHomeWork2\\test\\"
img_list = []
for i in range(1, 13):
    filename = ""
    filename += path + str(i) + ".jpg"
    img_list.append(cv.imread(filename))
for i in range(8, 12):
    cv.imshow("orig img "+str(i), img_list[i])
    bilateral = bilateral_filter(img_list[i])
    erode = erode_filter(img_list[i])
    gauss = gaussian_filter(img_list[i])
    result_image = get_result_image(img_list[i], calc_of_damage_and_non_damage(bilateral))
    cv.imshow("bilateral  " + str(i), result_image)
    result_image = get_result_image(img_list[i], calc_of_damage_and_non_damage(erode))
    cv.imshow("erode  " + str(i), result_image)
    result_image = get_result_image(img_list[i], calc_of_damage_and_non_damage(gauss))
    cv.imshow("gauss  " + str(i), result_image)

k = cv.waitKey(0)

