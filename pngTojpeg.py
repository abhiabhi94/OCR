import os, cv2, sys

Dir = '/home/abhyudai/Desktop/OCR/DataSet'
for dir in os.listdir(Dir):
	imageDir = os.path.join(Dir, dir)
	# print imageDir
	os.mkdir(imageDir.replace('DataSet', 'JPEG_DataSet'))
	imageFolder = os.listdir(imageDir)
	for images in imageFolder:
		# print imageDir
		# print images, os.path.join(imageDir, images)
		imgPath = os.path.join(imageDir, images)
		# print imgPath
		# sys.exit(0)
		imgNewPath = imgPath[:-4] + '.jpeg'
		imgNewPath = imgNewPath.replace('DataSet', 'JPEG_DataSet')
		# print imgNewPath
		cv2.imwrite(imgNewPath, cv2.imread(imgPath, 1))
		# img = cv2.imread(imgNewPath, 1)
		# cv2.imshow('', img)
		# cv2.waitKey(0)

