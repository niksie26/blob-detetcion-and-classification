import cv2
from math import sqrt, pi
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def detect_blobs(img_file, debug=False):
    """
    Function to detect blobs in an image. The technique used here is to binarize the image using Otsu threshold and use
    cv2.findContours().
    References:
    1. https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
    2. https://medium.com/analytics-vidhya/opencv-findcontours-detailed-guide-692ee19eeb18
    3. https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71
    4. https://learnopencv.com/blob-detection-using-opencv-python-c/
    :param img_file: (str) Input image filename - In this case blue_blobs.png
    :param debug: (bool) Flag to turn on plots for debugging, default is False
    :return img: () Original image read using imread
    :return contours: () List of lists containing the blobs detected in the image
    """
    # Read image
    img = cv2.imread(img_file)
    # Convert to grayscale
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Binarize the image. Threshold value = 29 based on inspecting the image values.
    # TODO: Instead of hard-coding value of 29, determine using thresholding technique.
    ret, thresh = cv2.threshold(image_gray, 29, 255, cv2.THRESH_OTSU)
    # Find contours in the image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if debug:
        # Copy the image and draw contours on this to avoid overwriting original image
        img_cpy = img
        # Uncomment only to capture video of the images with individual contours being drawn around the detected blobs'
        # height, width, layers = img.shape
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # video = cv2.VideoWriter('video.avi', fourcc, 1, (width, height))
        # Loop through each contour and draw it around the corresponding blob in the image
        for i in range(len(contours)):
            img_rep = img_cpy
            cnt = contours[i]
            cv2.drawContours(img_rep, [cnt], 0, (0, 255, 255), 5)
            # Add a text indicating the number of the blob corresponding to the contour list
            cv2.putText(img_rep, str(i), tuple(map(tuple, cnt[0]))[0], cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
            # video.write(img_rep)
            cv2.imwrite("out_" + str(i) + ".png", img_rep)
        cv2.destroyAllWindows()
        # video.release()
        plt.imshow(image_gray)
        plt.show()
        plt.imshow(thresh)
        plt.show()
        plt.imshow(img)
        plt.show()

    return img, contours


def extract_features(contours):
    """
    Extract features from each blob From the blobs found in the image, these are the features that are
    extracted:
    1. Area
    2. Perimeter
    3. Circularity
    4. Aspect ratio
    Area: The area of the blob is calculated using contourArea from OpenCV, This is used to calculate the area using
    the extracted contours from the image.
    Perimeter: The perimeter of the blob is calculated using arcLength from OpenCV, This is used to calculate the
    perimeter using the extracted contours from the image. Here, ‘True’ is used as the parameter to signify that the
    blob is a closed contour.
    Circularity: Circularity describes how close to a circle the observed blob is. The
    ‘fuller’ the circle, the closer the value is to 1. The Circularity is calculated as :
    circularity=4(pi)(area)/(perimeter)^2
    Aspect ratio: For aspect ratio, I fit the least possible bounding rectangle to the blob and calculated the ratio
    of length to breadth. This is an important feature as it defines how symmetric the shape of the blob is.
    Reference:
     1. https://github.com/NJNischal/Blob-Detection-and-Classification/blob/master/README.md
    :param contours: ()List of lists containing the blobs detected in the image
    :return feature_vec: List of lists containing feature vector comprised of area, perimeter,
    circularity and aspect ratio
    """
    # TODO: Are there other features that describe the blobs?
    # area = []
    # perimeter = []
    # circularity = []
    # aspect_ratio = []
    feature_vec = []
    n = len(contours)
    # Loop through the list of contours (blobs) and calculate area, perimeter, circularity and aspect ratio for each.
    for i in range(n):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        # area.append(A)
        perimeter = cv2.arcLength(cnt, True)
        # perimeter.append(P)
        circularity = 4 * pi * area / (perimeter ** 2)
        # circularity.append(Circ)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        aspect_ratio = sqrt(((box[0][0] - box[1][0]) ** 2) + ((box[0][1] - box[1][1]) ** 2)) / (
            sqrt(((box[2][0] - box[1][0]) ** 2) + ((box[2][1] - box[1][1]) ** 2)))
        # aspect_ratio.append(aspectR)
        values = [area, perimeter, circularity, aspect_ratio]
        feature_vec.append(values)
    return feature_vec


def find_clusters(features, debug=False):
    """
    Unsupervised Learning via KMeans Clustering has been used to cluster all the detected blobs in the image to 4
    unique clusters based on the features extracted in extract_features(). The number of clusters = 4 as per problem
    statement. Feature vector has been scaled/normalized. References: 1.
    https://realpython.com/k-means-clustering-python/#overview-of-clustering-techniques 2.
    https://scikit-learn.org/stable/modules/clustering.html#k-means 3.
    https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_assumptions.html#sphx-glr-auto-examples-cluster
    -plot-kmeans-assumptions-py
    :param features: () List of list consisting of a feature vector for each detected blob
    :param debug: (bool) Flag to turn on plots for debugging, default is False
    :return kmeans.labels_: (int) List of integers assigning a class label = {0, 1, 2, 3}
    to each of the detected blobs based on the features
    """
    # TODO: Implement evaluation routine to validate KMeans results
    # TODO: Try different clustering techniques and compare evaluation metrics to use most optimum clustering technique
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    kmeans = KMeans(init="random", n_clusters=4, n_init=10, max_iter=300, random_state=42)
    kmeans.fit(scaled_features)
    if debug:
        print("Kmeans Inertia:", kmeans.inertia_)
        print("Kmeans Cluster Centers:", kmeans.cluster_centers_)
        print("Number of iterations to converge", kmeans.n_iter_)
        print("Kmeans labels", kmeans.labels_)
    return kmeans.labels_


def apply_labels_to_blobs(image, contours, class_labels, debug=False):
    """
    Function to assign each detected blob to a color corresponding to the cluster ID/class label.
    cv2.fillPoly is used to fill in the blobs with the color.
    Final image showing blobs in 4 colors is written out to a png file.
    :param image: () Original Image read by cv2.imread
    :param contours: () List of lists containing the blobs detected in the image
    :param class_labels: (int) List of integers containing a class label = {0, 1, 2, 3} for each of the detected blobs
    :param debug: (bool) Flag to turn on plots for debugging, default is False
    :return: None
    """
    n = len(contours)
    # Dictionary mapping class label to a unique RGB color vector
    clr_map = {0: (255, 0, 255), 1: (0, 255, 255), 2: (0, 0, 255), 3: (255, 255, 0)}
    # Loop through all the detected blobs and fill them with a color according to their class label
    for i in range(n):
        cnt = contours[i]
        cv2.fillPoly(image, [cnt], color=clr_map[class_labels[i]])
        if debug:
            plt.imshow(image)
            plt.show()
    if debug:
        plt.imshow(image)
        plt.show()
    # Write out the image with filled and clustered blobs
    # TODO: Output file name can be made a parameter
    cv2.imwrite("Output_with_Clusters.png", image)


if __name__ == "__main__":
    """
    Problem Statement: Download the attached image file. Write a python script that clusters all the blue blobs in 
    the image into four distinct clusters based on their size and shape. The script should generate a new image with 
    identical dimensions and blobs, but each blue blob should be given a new color based on its cluster ID (4 unique 
    clusters and 4 unique colors). You are welcome to use third party packages (e.g. numpy, scikit-learn, tensorflow, 
    etc.) though the design and implementation of custom features that you believe to have meaningful value for 
    describing blob morphology is a plus. We are expecting a solution that could be developed in 45-90 minutes. 
    """

    # Fixed inputs/ parameters
    im_file = "blue_blobs.png"
    debug_flag = False

    # Detect blobs in the input image
    input_image, contour_list = detect_blobs(im_file, debug_flag)
    # Extract features from each of the detected blobs
    feature_vector = extract_features(contour_list)
    # Cluster detected blobs into 4 classes/clusters and get labels for each of the detected blobs
    labels = find_clusters(feature_vector, debug_flag)
    # Assign each blob to a new color based on its cluster ID/class label
    apply_labels_to_blobs(input_image, contour_list, labels, debug_flag)
    print("Done!!")
