import cv2
import numpy as np
import sys

class Image_Stitching():
    def __init__(self) :
        self.ratio=0.85
        self.min_match=10
        self.sift=cv2.xfeatures2d.SIFT_create()
        self.smoothing_window_size=800

    def registration(self,img1,img2):
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        good_points = []
        good_matches=[]
        for m1, m2 in raw_matches:
            if m1.distance < self.ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
        cv2.imwrite('matching.jpg', img3)
        if len(good_points) > self.min_match:
            image1_kp = np.float32(
                [kp1[i].pt for (_, i) in good_points])
            image2_kp = np.float32(
                [kp2[i].pt for (i, _) in good_points])
            H, status = cv2.findHomography(image1_kp, image2_kp, cv2.RANSAC,5.0)
            print(H)
        
            height, width, channels = img2.shape
            im1Reg = cv2.warpPerspective(img1, H, (width, height))
            cv2.imwrite("im1Reg.jpeg", im1Reg)
            return im1Reg




def main(argv1,argv2):
    img1 = cv2.imread(argv1)
    img2 = cv2.imread(argv2)
    final=Image_Stitching().registration(img1,img2)
    cv2.imwrite("final.jpeg", final)
    dim = (img2.shape[1], img2.shape [0])
    resized_img = cv2.resize(final, dim, interpolation = cv2.INTER_AREA)
    dst = cv2.addWeighted(img2,0.3,resized_img,0.5,0)
    cv2.imwrite("overlapped.jpeg", dst)
    #cv2.imshow("overlapped", dst)
    #cv2.waitKey(0)

if __name__ == '__main__':
    try: 
        main(sys.argv[1],sys.argv[2])
    except IndexError:
        print ("Please input two source images: ")
        print ("For example: python Image_Stitching.py '/Users/linrl3/Desktop/picture/p1.jpg' '/Users/linrl3/Desktop/picture/p2.jpg'")