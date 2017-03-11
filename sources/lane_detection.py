import numpy as np
import cv2
import glob, os
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # current iteration index w.r.t iteration size
        self.currentIteration = 0

        self.MAX_ITERATION_SIZE = 4

        self.frameCounter = 0
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.left_nfitx = np.zeros((self.MAX_ITERATION_SIZE, 720))
        self.right_nfitx = np.zeros((self.MAX_ITERATION_SIZE, 720))

        self.left_fit = None
        self.right_fit = None

        self.leftx_base = None
        self.rightx_base = None

        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None


def camera_caliberation(nx = 9, ny = 6):
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('../camera_cal/calibration*.jpg')
    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)


    #cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return (mtx, dist)


def cal_undistort(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)

def color_gradient_binary(img):
    image = np.copy(img)

    LUV = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    white1 = cv2.inRange(LUV, np.array([225, 0, 0]), np.array([255, 255, 255]))

    HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    white2 = cv2.inRange(HLS, np.array([0, 215, 0]), np.array([255, 255, 255]))
    yellow1 = cv2.inRange(HLS, np.array([10, 80, 100]), np.array([100, 210, 155]))

    LAB = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    yellow2 = cv2.inRange(LAB, np.array([0, 0, 155]), np.array([255, 255, 200]))

    bit_layer = yellow1 | white2 | white1 | yellow2

    binary = np.zeros_like(image[:, :, 0])
    binary[bit_layer.astype(bool)] = 1

    return binary


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def save_result(original, result, name, folder):
    basename = os.path.basename(name)
    fname = "../output_images/" + folder + os.path.splitext(basename)[0] + ".png"
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(original)
    ax1.set_title('Original Image', fontsize=40)

    ax2.imshow(result, cmap="gray")
    ax2.set_title('Pipeline Result', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    print(fname)
    plt.savefig(fname)

def warp_image(img, inverse= False):
    src = np.float32([[250, 680],
        [1050, 680],
        [590, 455],
        [695, 455]
    ])

    dst = np.float32([[320, 680],
        [950, 680],
        [320, 0],
        [950, 0]
    ])

    # Given src and dst points, calculate the perspective transform matrix
    if (inverse == False) :
        M = cv2.getPerspectiveTransform(src, dst)
    else:
        M = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size)
    return warped

def set_histogram(binary_warped):
    global g_line

    bottom_half_y = int(binary_warped.shape[0] / 2)
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[bottom_half_y:, :], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right liness
    midpoint = np.int(histogram.shape[0] / 2)
    g_line.leftx_base = np.argmax(histogram[:midpoint])
    g_line.rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    #print ("Histogram peak set")
    #print((g_line.leftx_base, g_line.rightx_base))

def find_lane_lines(binary_warped):
    global g_line

    set_histogram(binary_warped)
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = g_line.leftx_base
    rightx_current = g_line.rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    g_line.left_fit = left_fit
    g_line.right_fit = right_fit
    g_line.detected = True

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    return (left_fitx, right_fitx, ploty)

def find_lane_lines_efficient(binary_warped):
    global g_line

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (g_line.left_fit[0] * (nonzeroy ** 2) + g_line.left_fit[1] * nonzeroy + g_line.left_fit[2] - margin)) & (
    nonzerox < (g_line.left_fit[0] * (nonzeroy ** 2) + g_line.left_fit[1] * nonzeroy + g_line.left_fit[2] + margin)))
    right_lane_inds = (
    (nonzerox > (g_line.right_fit[0] * (nonzeroy ** 2) + g_line.right_fit[1] * nonzeroy + g_line.right_fit[2] - margin)) & (
    nonzerox < (g_line.right_fit[0] * (nonzeroy ** 2) + g_line.right_fit[1] * nonzeroy + g_line.right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Calculate line width
    left_intercept = g_line.left_fit[0] * binary_warped.shape[0] ** 2 + g_line.left_fit[1] * binary_warped.shape[0] + g_line.left_fit[2]
    right_intercept = g_line.right_fit[0] * binary_warped.shape[0] ** 2 + g_line.right_fit[1] * binary_warped.shape[0] + g_line.right_fit[2]
    width_pixels = right_intercept - left_intercept

    if width_pixels < 400:
        g_line.detected = False
        print("Lines are too close. Resetting ...")
        print(width_pixels)
        return find_lane_lines(binary_warped)
    else:
        g_line.left_fit = left_fit
        g_line.right_fit = right_fit

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    return (left_fitx, right_fitx, ploty)

def calculate_vehicle_position(binary_warped):
    global  g_line
    xm_per_pix = 3.7 / 650
    left_line_x = g_line.left_fit[0] * binary_warped.shape[0] ** 2 + g_line.left_fit[1] * binary_warped.shape[0] + \
                     g_line.left_fit[2]
    right_line_x = g_line.right_fit[0] * binary_warped.shape[0] ** 2 + g_line.right_fit[1] * binary_warped.shape[0] + \
                      g_line.right_fit[2]

    vehicle_deviation = ((left_line_x + right_line_x) / 2.0) - (binary_warped.shape[1] / 2.0)
    return vehicle_deviation * xm_per_pix

def calculate_radius(binary_warped, left_fitx, right_fitx):
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 650  # meters per pixel in x dimension

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    y_eval = 720
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    return (left_curverad, right_curverad   )

def process_image(img):
    global g_line

    g_line.frameCounter += 1
    undist_img = cal_undistort(img, mtx, dist)
    binary_img = color_gradient_binary(undist_img)
    imgWidth = binary_img.shape[1]
    imgHeight = binary_img.shape[0]

    vertices = np.array([[(0, imgHeight),  # bottom left
                          (imgWidth * 0.47, imgHeight * 0.60),  # top left
                          (imgWidth * 0.53, imgHeight * 0.60),  # top right
                          (imgWidth, imgHeight)]], dtype=np.int32)  # bottom right

    binary_masked_img = region_of_interest(binary_img, vertices)

    warped_img = warp_image(binary_masked_img)

    if g_line.detected:
        left_fitx, right_fitx, ploty = find_lane_lines_efficient(warped_img)
    else:
        left_fitx, right_fitx, ploty = find_lane_lines(warped_img)

    # Line information
    g_line.left_nfitx[g_line.currentIteration] = left_fitx
    g_line.right_nfitx[g_line.currentIteration] = right_fitx

    g_line.currentIteration += 1
    g_line.currentIteration %= g_line.MAX_ITERATION_SIZE

    left_fitx_average = left_fitx
    right_fitx_average = right_fitx

    # Average the lines
    if (g_line.frameCounter > g_line.MAX_ITERATION_SIZE):
        left_fitx_average = np.average(g_line.left_nfitx, axis = 0)
        right_fitx_average = np.average(g_line.right_nfitx, axis = 0)



    left_line = np.array([np.transpose(np.vstack([left_fitx_average, ploty]))])
    right_line = np.array([np.flipud(np.transpose(np.vstack([right_fitx_average, ploty])))])
    binary_warped_zero = np.zeros_like(warped_img).astype(np.uint8)

    lanes_warped = np.dstack((binary_warped_zero, binary_warped_zero, binary_warped_zero))
    lane_pts = np.hstack((left_line, right_line))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(lanes_warped, np.int_([lane_pts]), (0, 255, 0))
    unwarped_result = cv2.addWeighted(undist_img, 1, warp_image(lanes_warped, True), 0.3, 0)

    vehicle_deviation = calculate_vehicle_position(warped_img)
    left_curvature, right_curvature  = calculate_radius(warped_img, left_fitx_average, right_fitx_average)

    cv2.putText(unwarped_result, 'Vehicle Lane Deviation in m: {:06.3f}'.format(vehicle_deviation), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,255,155), 2, cv2.LINE_8)

    cv2.putText(unwarped_result, 'Curvature in m: Left {:06.2f}'.format(left_curvature), (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 255, 155), 2, cv2.LINE_8)

    cv2.putText(unwarped_result,
                'Right {:06.2f}'.format(right_curvature), (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 255, 155), 2, cv2.LINE_8)

    return unwarped_result



def test_images():
    # Load test_images
    test_images = glob.glob('../test_images/test*.jpg')
    for fname in test_images:
        img = cv2.imread(fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Undistort the image
        undist_img = cal_undistort(img, mtx, dist)

        save_result(img, undist_img, fname, "undistorted/")

        # Get the binary thresholded image
        binary_img = color_gradient_binary(undist_img)
        save_result(img, binary_img, fname, "threshold/")

        imgWidth = binary_img.shape[1]
        imgHeight = binary_img.shape[0]

        vertices = np.array([[(0, imgHeight),  # bottom left
                              (imgWidth * 0.47, imgHeight * 0.60),  # top left
                              (imgWidth * 0.53, imgHeight * 0.60),  # top right
                              (imgWidth, imgHeight)]], dtype=np.int32)  # bottom right

        binary_masked_img = region_of_interest(binary_img, vertices)

        warped_img = warp_image(binary_masked_img)
        save_result(img, warped_img, fname, "warped/")

        left_fitx, right_fitx, ploty = find_lane_lines(warped_img)

        left_line = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        right_line = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        binary_warped_zero = np.zeros_like(warped_img).astype(np.uint8)

        lanes_warped = np.dstack((binary_warped_zero, binary_warped_zero, binary_warped_zero))
        lane_pts = np.hstack((left_line, right_line))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(lanes_warped, np.int_([lane_pts]), (0, 255, 0))
        unwarped_result = cv2.addWeighted(undist_img, 1, warp_image(lanes_warped, True), 0.3, 0)
        save_result(img, unwarped_result, fname, "final/")

######### Main  #############

#Get camera caliberation
mtx, dist = camera_caliberation()

g_line = Line()
#test the pipleline on images
test_images()

#Test on video
g_line = Line()
project_output = '../project_video_output.mp4'
clip1 = VideoFileClip("../project_video.mp4")
out_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
out_clip.write_videofile(project_output, audio=False)