import cv2
import numpy as np
import operator
import os

# module level variables ##########################################################################
MIN_CONTOUR_AREA = 50
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 20


# SZ=20
# bin_n = 16 # Number of bins
# affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
# svm_params = dict( kernel_type = cv2.SVM_LINEAR,svm_type = cv2.SVM_C_SVC,C=2.67, gamma=5.383 )
class ContourWithData():

    npaContour = None           # contour
    boundingRect = None         # bounding rect for contour
    intRectX = 0                # bounding rect top left corner x location
    intRectY = 0                # bounding rect top left corner y location
    intRectWidth = 0            # bounding rect width
    intRectHeight = 0           # bounding rect height
    fltArea = 0.0               # area of contour

    def calculateRectTopLeftPointAndWidthAndHeight(self):               # calculate bounding rect info
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):                            # this is oversimplified, for a production grade program
        if self.fltArea < MIN_CONTOUR_AREA: return False        # much better validity checking would be necessary
        return True

# def deskew(img):
#     cv2.imshow('img_start',img)
#     m = cv2.moments(img)
#     #print m['mu11']
#     if abs(m['mu02']) < 1e-2:
#         return img.copy()
#     skew = m['mu11']/m['mu02']
#     print skew
#     M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
#     img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
#     cv2.imshow('img_end',img)
#     return img

def main():
    allContoursWithData = []
    validContoursWithData = []

##################################################################################
##
##    #npaClassifications = np.array([9], np.float32)
#####for training for one sample
##    '''
##    npaClassifications= np.loadtxt("classifications.txt", np.float32)
##    npaClassifications.sort()
##    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))
##    '''
##
#####fopr training for multiple samples
##    npaClassifications = []
##    for i in range(0,10):
##        for j in range(1,501):
##            npaClassifications.append([i])
##    npaClassifications = np.array(npaClassifications,np.float32)
##    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))
##    
##    
##    print npaClassifications
##    npaFlattenedImages = np.empty((0,400))
##
#####for training with only one sample
##    '''
##    for i in range(0,10):
##        npaFlattenedImage= np.loadtxt("flat_text/text("+str(i)+")/flattened_images"+str(i+1)+".txt",np.float32)
##
##        npaFlattenedImages = np.append(npaFlattenedImages, [npaFlattenedImage],0)
##        npaFlattenedImages = np.array(npaFlattenedImages,np.float32)
##    '''
#####for training with only multiple samples
##    for i in range(0,10):
##        for j in range(1,501):+
##            npaFlattenedImage= np.loadtxt("flat_text/text("+str(i)+")/flattened_images"+str(j)+".txt",np.float32)
##            npaFlattenedImages = np.append(npaFlattenedImages, [npaFlattenedImage],0)
##
##    npaFlattenedImages = np.array(npaFlattenedImages,np.float32)
##    
##    print len(npaFlattenedImages),npaFlattenedImages
##    np.savetxt("saved_data/classifications.txt", npaClassifications)           # write flattened images to file
##    np.savetxt("saved_data/flattened_images.txt", npaFlattenedImages)          #
#####################################################################################
#####################################################################################

###################################################################
    
    npaFlattenedImages= np.loadtxt("npaFlattenedImages.txt",np.float32) 
    npaClassifications= np.loadtxt("npaClassifications", np.float32)
    kNearest = cv2.KNearest()
    kNearest.train(npaFlattenedImages, npaClassifications)
    #svm = cv2.SVM()
    #svm.train(npaFlattenedImages, npaClassifications,params = svm_params)
    #print z

    #imgTestingNumbers = cv2.imread('samples/sample(9)/node'+str(6)+'.jpg')
    #imgTestingNumbers = cv2.imread('self_samples/5/sample1.png')
    imgTestingNumbers = cv2.imread("test.jpg")
    RP = imgTestingNumbers.shape[0] / 800 if imgTestingNumbers.shape[0] <= imgTestingNumbers.shape[1] else imgTestingNumbers.shape[1] / 800
    imgTestingNumbers = cv2.resize (imgTestingNumbers, (imgTestingNumbers.shape[1] / RP, imgTestingNumbers.shape[0] / RP))
    #cv2.imshow('imgTestingNumber1',imgTestingNumbers)
    


    imgGray = cv2.cvtColor(imgTestingNumbers, cv2.COLOR_BGR2GRAY)
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)
    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)
    # cv2.imshow('imgThresh',imgThresh)
    imgThreshCopy = imgThresh.copy()



    npaContours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(imgTestingNumbers,npaContours,-1,(255,255,0),2)
    for npaContour in npaContours:
        # print cv2.contourArea(npaContour)
        contourWithData = ContourWithData()                                             # instantiate a contour with data object
        contourWithData.npaContour = npaContour                                         # assign contour to contour with data
        contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)     # get the bounding rect
        contourWithData.calculateRectTopLeftPointAndWidthAndHeight()                    # get bounding rect info
        contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)           # calculate the contour area
        allContoursWithData.append(contourWithData)

    for contourWithData in allContoursWithData:
        if contourWithData.checkIfContourIsValid():
            validContoursWithData.append(contourWithData)

    validContoursWithData.sort(key = operator.attrgetter("intRectX"))
    strFinalString = ""
    i = 0
    a = 0.0
    b = 0
    for contourWithData in validContoursWithData:
        i += 1
        cv2.rectangle(imgTestingNumbers,(contourWithData.intRectX, contourWithData.intRectY),(contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),(0, 255, 0),2)
        imgROI = imgThresh[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]
        imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
        npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
        npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
        #npaROIResized = deskew(npaROIResized)
        npaROIResized = np.float32(npaROIResized)


        retval, npaResults, neigh_resp, dists = kNearest.find_nearest(npaROIResized, k = 1)
        #npaResults = svm.predict_all(npaROIResized)
        strCurrentChar = str(int(npaResults[0][0]))
        strFinalString = strFinalString + strCurrentChar

        cv2.namedWindow('Fuck '+str(i),cv2.WINDOW_NORMAL)
        cv2.imshow('Fuck '+str(i),imgROI)
        print strCurrentChar

        b += 1

        if (cv2.waitKey(0) & 255) == 121:  ### For Windows Os remove this 255 ###
            a = a + 1
        cv2.destroyAllWindows()

    print 'Accuracy:',a / b

    #print "\n" + strFinalString + "\n"

    cv2.imshow("imgTestingNumbers", imgTestingNumbers)
    # cv2.imwrite("photo_ocr_example.jpg",imgTestingNumbers)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return
            
main()