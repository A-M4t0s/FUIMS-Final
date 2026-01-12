#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <sensor_msgs/NavSatFix.h>
#include <geometry_msgs/QuaternionStamped.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <cv_bridge/cv_bridge.h>

#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/PoseRotationPrior.h>
#include <gtsam/slam/PoseTranslationPrior.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/navigation/GPSFactor.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video.hpp>

#include <signal.h>
#include <thread>
#include <mutex>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <Eigen/Dense>
#include <unordered_map>
#include <fstream>
#include <cstdlib>
#include <iomanip>
#include <deque>

// =========================================================
//  ROSBAG Defines
// =========================================================
#define BAG_PATH "/home/tony/catkin_ws/bags/bags_fuims/agucadoura.bag"
#define CAMERA_TOPIC "/dji_m350/cameras/main/compressed"
#define QUATERNION_TOPIC "/dji_m350/quaternion"
#define VELOCITY_TOPIC "/dji_m350/velocity"
#define GPS_TOPIC "/dji_m350/gps"

#define GREEN "\033[1;32m"
#define RED "\033[1;31m"
#define YELLOW "\033[1;33m"
#define CYAN "\033[1;36m"
#define RESET "\033[0m"

#define INFO(msg) ROS_INFO_STREAM(CYAN << msg << RESET)
#define OK(msg) ROS_INFO_STREAM(GREEN << msg << RESET)
#define WARN(msg) ROS_WARN_STREAM(YELLOW << msg << RESET)
#define ERROR(msg) ROS_ERROR_STREAM(RED << msg << RESET)

// =========================================================
//  Data Structures
// =========================================================
/**
 * @brief Image Frame Structure
 * @param image Undistorted Image as cv::Mat
 * @param timestamp ROS Time Stamp
 * @return ImageFrame structure
 */
typedef struct
{
  cv::Mat image;
  ros::Time timestamp;
} ImageFrame;

/**
 * @brief Points Structure
 * @param ids Vector of Point IDs
 * @param pts Vector of 2D Points as cv::Point2f
 * @param isTracked Vector of booleans indicating if points are tracked
 * @param age Vector of integers indicating the age of each point
 * @return Points structure
 */
struct Points
{
  std::vector<int> ids;         
  std::vector<cv::Point2f> pts;
  std::vector<bool> isTracked; 
  std::vector<int> age;        
};

// =========================================================
// Signal Handling
// =========================================================
volatile sig_atomic_t g_requestShutdown = 0;
volatile sig_atomic_t terminate_requested = 0;

void signalHandler(int sig)
{
  if (sig == SIGINT)
  {
    g_requestShutdown = 1;
    ROS_WARN("SIGINT received. Shutting down VIO.");
  }
}

// =========================================================
// VIO Manager Class
// =========================================================
class vioManager
{
public:
  vioManager()
  {
    // =========================================================
    // Setup
    // =========================================================
    // Setting signal 'SIGINT' reception
    signal(SIGINT, signalHandler);

    // Initializing ROS Publishers [TODO]

    // Initializing ROS Subscribers
    imgSub = nh.subscribe<sensor_msgs::CompressedImage>(CAMERA_TOPIC, 1, &vioManager::imgCallback, this);
    resetSub = nh.subscribe<std_msgs::Bool>("vio/reset", 1, &vioManager::resetCallback, this);

    // =========================================================
    // Processing Loop
    // =========================================================
    ros::Rate rate(200);
    while (ros::ok() && !g_requestShutdown)
    {
      // Image Buffer Processing
      if (!imgMsgs.empty())
      {
        // Pop the first image frame from the buffer
        ImageFrame frame = imgMsgs.front();
        imgMsgs.pop_front();

        if (isFirstFrame) // First Frame Processing
        {
          OK("First frame received with timestamp: " << frame.timestamp.toSec());
          isFirstFrame = false;
        }
        else // Subsequent Frame Processing
        {
          INFO("Processing frame with timestamp: " << frame.timestamp.toSec());
        }
      }

      ros::spinOnce();
      rate.sleep();
    }
  }

private:
  // =========================================================
  // Variables
  // =========================================================
  // ROS Related
  ros::NodeHandle nh;
  ros::Publisher matchingPub, featurePub, posePub, pathPub, gtPathPub;
  ros::Subscriber resetSub, imgSub, quatSub, velSub, gpsSub;
  nav_msgs::Path pathMsg;

  // Message Storage
  std::vector<geometry_msgs::QuaternionStampedConstPtr> quatMsgs;
  std::vector<geometry_msgs::Vector3StampedConstPtr> velMsgs;
  std::vector<sensor_msgs::NavSatFixConstPtr> gpsMsgs;
  std::vector<std_msgs::Float32ConstPtr> altMsgs;
  std::deque<ImageFrame> imgMsgs;

  // Camera Parameters
  const cv::Mat K = (cv::Mat_<double>(3, 3) << 1372.76165, 0.0, 960.45289,
                     0.0, 1372.14817, 515.00383,
                     0.0, 0.0, 1.0);
  const cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << 0.048300, -0.044905, -0.003731, -0.001349, 0);

  // Status variables
  bool isFirstFrame = true;

  // =========================================================
  // Callbacks
  // =========================================================
  /**
   * @brief Reset Callback
   * @param msg Bool Message
   * @return void
   *
   * This callback handles reset requests.
   * When a reset message with data 'true' is received, it resets the VIO state.
   */
  void resetCallback(const std_msgs::BoolConstPtr &msg)
  {
    if(msg->data)
    {
      OK("Reset requested. Resetting VIO state.");
      // Reset internal states and variables here
      isFirstFrame = true;
      imgMsgs.clear();
    }
  }

  /**
   * @brief Image Callback
   * @param msg Compressed Image Message
   * @return void
   *
   * This callback handles incoming compressed image messages.
   * It calls the 'undistortImage' function to process the image, in order to fill the image buffers.
   */
  void imgCallback(const sensor_msgs::CompressedImageConstPtr &msg)
  {
    // INFO("Received image message with timestamp: " << msg->header.stamp.toSec());
    ImageFrame frame = undistortImage(msg);
    if (frame.image.empty())
      return;

    imgMsgs.push_back(frame);
  }

  // =========================================================
  // Methods
  // =========================================================
  /**
   * @brief Undistort Image
   * @param msg Compressed Image Message
   * @return undistorted greyscale image as cv::Mat
   *
   * This method undistorts the incoming compressed image using the predefined camera matrix and distortion coefficients.
   * It returns the undistorted image as a cv::Mat object, with a greyscale colorformat.
   */
  ImageFrame undistortImage(const sensor_msgs::CompressedImageConstPtr msg)
  {
    ImageFrame frame;
    frame.timestamp = msg->header.stamp;
    cv::Mat raw = cv::imdecode(msg->data, cv::IMREAD_GRAYSCALE);
    if (raw.empty())
      return frame;
    cv::undistort(raw, frame.image, K, distCoeffs);
    return frame;
  }

  /**
   * @brief Feature Detection
   * @param img Input Image as cv::Mat
   * @return Detected Points structure
   */
  Points featureDetection(const cv::Mat &img)
  {
    Points detectedPoints;

    // GFTT Detector

    return detectedPoints;
  }
};

// =========================================================
// Main Function
// =========================================================
int main(int argc, char **argv)
{
  ros::init(argc, argv, "fuims_vio");
  vioManager manager;
  ERROR("VIO Node has been terminated.");
  return 0;
}