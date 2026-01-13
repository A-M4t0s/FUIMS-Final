#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <std_msgs/UInt8.h>
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

#include <dynamic_reconfigure/server.h>
#include <fuims/vioParamsConfig.h>

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
#define CLEAR "\033[0m"

#define INFO(msg) ROS_INFO_STREAM(CYAN << msg << CLEAR)
#define OK(msg) ROS_INFO_STREAM(GREEN << msg << CLEAR)
#define WARN(msg) ROS_WARN_STREAM(YELLOW << msg << CLEAR)
#define ERROR(msg) ROS_ERROR_STREAM(RED << msg << CLEAR)

// =========================================================
//  Enum
// =========================================================
enum class vioState
{
  IDLE = 0,
  RUNNING = 1,
  RESET = 2
};

// =========================================================
//  Data Structures
// =========================================================
/**
 * @brief Image Frame Structure
 * @param image Undistorted Image as cv::Mat
 * @param timestamp ROS Time Stamp
 * @return ImageFrame structure
 */
struct ImageFrame
{
  cv::Mat image;
  ros::Time timestamp;
};

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

/**
 * @brief Keyframe Structure
 * @param timestamp ROS Time Stamp
 * @param image Image associated with the keyframe as cv::Mat
 * @param points Points associated with the keyframe
 * @param pose Camera Pose as gtsam::Pose3
 * @return Keyframe structure
 */
struct Keyframe
{
  ros::Time timestamp;
  cv::Mat image;
  Points points;
  gtsam::Pose3 pose;
};

// =========================================================
// Signal Handling
// =========================================================
volatile sig_atomic_t g_requestShutdown = 0;

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

    // Setting ROS Params
    loadParams();

    // Start dynamic reconfigure server
    drCallback = boost::bind(&vioManager::configCallback, this, _1, _2);
    drServer.setCallback(drCallback);

    // Load ROSBAG and messages
    bufferRosbagMessages(BAG_PATH);

    // Initializing ROS Publishers
    matchingPub = nh.advertise<sensor_msgs::Image>("vio/feature_matches", 1);
    featurePub = nh.advertise<sensor_msgs::Image>("vio/feature_ages", 1);

    // Initializing ROS Subscribers
    stateSub = nh.subscribe<std_msgs::UInt8>("vio/state", 1, &vioManager::stateCallback, this);

    cv::initUndistortRectifyMap(
        K,                    // Intrinsic matrix
        distCoeffs,           // Distortion coefficients
        cv::Mat(),            // Optional rectification matrix (empty -> no stereo)
        K,                    // New camera matrix (same as original)
        cv::Size(1920, 1080), // Image size
        CV_16SC2,             // Output map format
        map1, map2            // Output maps
    );

    // =========================================================
    // Processing Loop
    // =========================================================
    ros::Rate rate(200);
    while (ros::ok() && !g_requestShutdown)
    {
      switch (vio_state_)
      {
      case vioState::IDLE:
        ROS_INFO_THROTTLE(2, "[VIO Manager] Waiting for Start State");
        break;

      case vioState::RUNNING:
        if (frameIdx == imgBuffer.size())
        {
          OK("[VIO Manager] Processing Ended! Changing state to RESET");
          vio_state_ = vioState::RESET;
        }
        else
        {
          // Feature Detection + Feature Tracking
          bool success = frameProcessing();

          

          if (success)
          {
            // [DEBUG] - Feature Ages Image Publisher
            debugImageFeatureAges();

            // [DEBUG] - Feature Matching Image Publisher
            debugImageFeatureMatching();

            // Update previous frame data
            prevFrame = currFrame;
            prevPoints = currPoints;
          }
        }
        break;

      case vioState::RESET:
        WARN("[VIO Manager] Reset Command Received! Setting variables to default...");
        statesReset();
        OK("[VIO Manager] Variables resetted!");
        vio_state_ = vioState::IDLE;
        break;

      default:
        break;
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
  ros::Subscriber stateSub, resetSub;
  nav_msgs::Path pathMsg;

  // Dynamic Reconfigure
  dynamic_reconfigure::Server<fuims::vioParamsConfig> drServer;
  dynamic_reconfigure::Server<fuims::vioParamsConfig>::CallbackType drCallback;

  // Global Variables for ROS Parameters
  // === General VIO Parameters ===
  int MAX_FEATURES, MIN_FEATURES, MIN_TRACKED_FEATURES, MAX_TRACKING_AGE,
      KF_FEATURE_THRESHOLD, KF_PARALLAX_THRESHOLD, GPS_PRIOR_INTERVAL;
  float MAX_TRACKING_ERROR_PX;

  // === GFTT Parameters ===
  int GFTT_MAX_FEATURES, GFTT_BLOCK_SIZE;
  double GFTT_QUALITY, GFTT_MIN_DIST;

  // === KLT Parameters ===
  int KLT_MAX_LEVEL, KLT_ITERS, KLT_BORDER_MARGIN;
  double KLT_EPS, KLT_MIN_EIG;
  float KLT_FB_THRESH_PX;

  // === ROS Topic Message Rates ===
  int CAMERA_RATE, INERTIAL_RATE;

  // ROS Topic Messages Buffers
  std::deque<sensor_msgs::CompressedImageConstPtr> imgBuffer;
  std::deque<sensor_msgs::NavSatFixConstPtr> gpsBuffer;
  std::deque<geometry_msgs::Vector3StampedConstPtr> velBuffer;
  std::deque<geometry_msgs::QuaternionStampedConstPtr> quatBuffer;

  // Camera Parameters
  const cv::Mat K = (cv::Mat_<double>(3, 3) << 1372.76165, 0.0, 960.45289,
                     0.0, 1372.14817, 515.00383,
                     0.0, 0.0, 1.0);
  const cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << 0.048300, -0.044905, -0.003731, -0.001349, 0);
  cv::Mat map1, map2;

  // Status variables
  vioState vio_state_ = vioState::IDLE;
  bool isFirstFrame = true;
  int frameIdx = 0;

  // VO-Related Variables
  ImageFrame currFrame, prevFrame;
  Points currPoints, prevPoints;
  int nextFeatureID = 0;

  // =========================================================
  // Helpers
  // =========================================================
  /**
   * @brief ROS Parameter Loader
   * @param void
   * @return void
   */
  void loadParams()
  {
    // === General VIO Parameters ===
    nh.param("MIN_FEATURES", MIN_FEATURES, 25);
    nh.param("MAX_FEATURES", MAX_FEATURES, 250);
    nh.param("MIN_TRACKED_FEATURES", MIN_TRACKED_FEATURES, 35);
    nh.param("MAX_TRACKING_ERROR_PX", MAX_TRACKING_ERROR_PX, 7.5f);
    nh.param("MAX_TRACKING_AGE", MAX_TRACKING_AGE, 5);
    nh.param("KF_FEATURE_THRESHOLD", KF_FEATURE_THRESHOLD, 75);
    nh.param("KF_PARALLAX_THRESHOLD", KF_PARALLAX_THRESHOLD, 28);
    nh.param("GPS_PRIOR_INTERVAL", GPS_PRIOR_INTERVAL, 10);

    // === GFTT Parameters ===
    nh.param("GFTT_MAX_FEATURES", GFTT_MAX_FEATURES, 500);
    nh.param("GFTT_BLOCK_SIZE", GFTT_BLOCK_SIZE, 3);
    nh.param("GFTT_QUALITY", GFTT_QUALITY, 0.15);
    nh.param("GFTT_MIN_DIST", GFTT_MIN_DIST, 26.0);

    // === KLT Parameters ===
    nh.param("KLT_MAX_LEVEL", KLT_MAX_LEVEL, 4);
    nh.param("KLT_ITERS", KLT_ITERS, 30);
    nh.param("KLT_BORDER_MARGIN", KLT_BORDER_MARGIN, 10);
    nh.param("KLT_EPS", KLT_EPS, 0.01);
    nh.param("KLT_MIN_EIG", KLT_MIN_EIG, 1e-4);
    nh.param("KLT_FB_THRESH_PX", KLT_FB_THRESH_PX, 1.0f);

    // === ROS Topic Message Rates ===
    nh.param("CAMERA_RATE", CAMERA_RATE, 15);
    nh.param("INERTIAL_RATE", INERTIAL_RATE, 30);
  }

  /**
   * @brief ROSBAG Buffer Messages
   * @param bag_path  Path to the desired ROSBAG to be processed
   * @return void
   */
  void bufferRosbagMessages(const std::string &bag_path)
  {
    rosbag::Bag bag;
    bag.open(bag_path, rosbag::bagmode::Read);
    bool firstImage = true;

    std::vector<std::string> topics = {
        CAMERA_TOPIC,
        GPS_TOPIC,
        VELOCITY_TOPIC,
        QUATERNION_TOPIC};

    rosbag::View view(bag, rosbag::TopicQuery(topics));

    for (const rosbag::MessageInstance &m : view)
    {
      if (m.getTopic() == CAMERA_TOPIC)
      {
        auto img = m.instantiate<sensor_msgs::CompressedImage>();
        if (img)
          imgBuffer.push_back(img);
      }
      else if (m.getTopic() == GPS_TOPIC)
      {
        auto gps = m.instantiate<sensor_msgs::NavSatFix>();
        if (gps)
          gpsBuffer.push_back(gps);
      }
      else if (m.getTopic() == VELOCITY_TOPIC)
      {
        auto vel = m.instantiate<geometry_msgs::Vector3Stamped>();
        if (vel)
          velBuffer.push_back(vel);
      }
      else if (m.getTopic() == QUATERNION_TOPIC)
      {
        auto quat = m.instantiate<geometry_msgs::QuaternionStamped>();
        if (quat)
          quatBuffer.push_back(quat);
      }
    }

    INFO("ROSBAG Messages Processed");
    OK("Image messages total: " << imgBuffer.size());
    OK("Quaternion messages total: " << quatBuffer.size());
    OK("Velocity messages total: " << velBuffer.size());
    OK("GPS messages total: " << gpsBuffer.size());

    bag.close();
  }

  void statesReset()
  {
    isFirstFrame = true;
    frameIdx = 0;
  }

  /**
   *
   */
  bool inImage(const cv::Point2f &p, const cv::Mat &img)
  {
    return p.x >= 16 && p.y >= 16 && p.x < img.cols - 16 && p.y < img.rows - 16;
  }

  cv::Mat drawFeatureMatches(const cv::Mat &img1, const cv::Mat &img2,
                             const std::vector<cv::Point2f> &pts1,
                             const std::vector<cv::Point2f> &pts2)
  {
    cv::Mat img1BGR, img2BGR;
    cv::cvtColor(img1, img1BGR, cv::COLOR_GRAY2BGR);
    cv::cvtColor(img2, img2BGR, cv::COLOR_GRAY2BGR);

    cv::Mat vis;
    cv::hconcat(img1BGR, img2BGR, vis);

    for (size_t i = 0; i < std::min(pts1.size(), pts2.size()); ++i)
    {
      cv::Point2f pt1 = pts1[i];
      cv::Point2f pt2 = pts2[i] + cv::Point2f(static_cast<float>(img1.cols), 0); // offset for right image
      cv::line(vis, pt1, pt2, cv::Scalar(0, 255, 0), 1);
      cv::circle(vis, pt1, 3, cv::Scalar(255, 0, 0), -1);
      cv::circle(vis, pt2, 3, cv::Scalar(0, 0, 255), -1);
    }

    return vis;
  }

  cv::Mat drawFeatureAges(const cv::Mat &img, const Points &points)
  {
    cv::Mat vis;
    cv::cvtColor(img, vis, cv::COLOR_GRAY2BGR);

    for (size_t i = 0; i < points.pts.size(); ++i)
    {
      int age = points.age[i];
      cv::Scalar color;

      if (age >= MAX_TRACKING_AGE)
        color = cv::Scalar(0, 255, 0); // Green for old
      else
        color = cv::Scalar(0, 0, 255); // Red for young

      cv::circle(vis, points.pts[i], 3, color, -1);
    }

    return vis;
  }

  void debugImageFeatureAges()
  {
    cv::Mat ageVis = drawFeatureAges(currFrame.image, currPoints);
    sensor_msgs::ImagePtr ageMsg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", ageVis).toImageMsg();
    ageMsg->header.stamp = currFrame.timestamp;
    featurePub.publish(ageMsg);
  }

  void debugImageFeatureMatching()
  {
    std::unordered_map<int, cv::Point2f> currPointMap;
    for (size_t i = 0; i < currPoints.ids.size(); ++i)
    {
      currPointMap[currPoints.ids[i]] = currPoints.pts[i];
    }
    std::vector<cv::Point2f> matchedPrevPts, matchedCurrPts;
    for (size_t i = 0; i < prevPoints.ids.size(); ++i)
    {
      int id = prevPoints.ids[i];
      auto it = currPointMap.find(id);
      if (it != currPointMap.end())
      {
        matchedPrevPts.push_back(prevPoints.pts[i]);
        matchedCurrPts.push_back(it->second);
      }
    }
    cv::Mat matchVis = drawFeatureMatches(prevFrame.image, currFrame.image,
                                          matchedPrevPts, matchedCurrPts);
    sensor_msgs::ImagePtr matchMsg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", matchVis).toImageMsg();
    matchMsg->header.stamp = currFrame.timestamp;
    matchingPub.publish(matchMsg);
  }

  // =========================================================
  // Callbacks
  // =========================================================
  /**
   * @brief Dynamic Reconfigure Callback
   * @param config vioParams Configuration
   * @param level Bit mask indicating modified params
   * @return void
   */
  void configCallback(fuims::vioParamsConfig &config, uint32_t level)
  {
    MAX_FEATURES = config.MAX_FEATURES;
    MIN_TRACKED_FEATURES = config.MIN_TRACKED_FEATURES;
    MAX_TRACKING_ERROR_PX = config.MAX_TRACKING_ERROR_PX;
    MAX_TRACKING_AGE = config.MAX_TRACKING_AGE;
    KF_FEATURE_THRESHOLD = config.KF_FEATURE_THRESHOLD;
    KF_PARALLAX_THRESHOLD = config.KF_PARALLAX_THRESHOLD;
    GPS_PRIOR_INTERVAL = config.GPS_PRIOR_INTERVAL;
    GFTT_MAX_FEATURES = config.GFTT_MAX_FEATURES;
    GFTT_BLOCK_SIZE = config.GFTT_BLOCK_SIZE;
    GFTT_QUALITY = config.GFTT_QUALITY;
    GFTT_MIN_DIST = config.GFTT_MIN_DIST;
    KLT_MAX_LEVEL = config.KLT_MAX_LEVEL;
    KLT_ITERS = config.KLT_ITERS;
    KLT_BORDER_MARGIN = config.KLT_BORDER_MARGIN;
    KLT_EPS = config.KLT_EPS;
    KLT_MIN_EIG = config.KLT_MIN_EIG;
    KLT_FB_THRESH_PX = config.KLT_FB_THRESH_PX;
    CAMERA_RATE = config.CAMERA_RATE;
    INERTIAL_RATE = config.INERTIAL_RATE;
    INFO("Dynamic reconfigure updated VIO parameters.");
    WARN("Changing VIO State to RESET");
    vio_state_ = vioState::RESET;
  }

  /**
   * @brief Reset Callback
   * @param msg UInt8 Message
   * @return void
   *
   * This callback changes the execution state of this node
   */
  void stateCallback(const std_msgs::UInt8ConstPtr &msg)
  {
    vioState new_state = static_cast<vioState>(msg->data);

    switch (new_state)
    {
    case vioState::IDLE:
      vio_state_ = vioState::IDLE;
      INFO("VIO State changed to IDLE");
      break;

    case vioState::RUNNING:
      vio_state_ = vioState::RUNNING;
      INFO("VIO State changed to RUNNING");
      break;

    case vioState::RESET:
      vio_state_ = vioState::RESET;
      INFO("VIO State changed to RESET");
      break;

    default:
      WARN("Unknown VIO state received: " << static_cast<int>(msg->data));
      break;
    }
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

    // If needed, resize maps to match current image size
    if (map1.empty() || map1.size() != raw.size())
    {
      cv::initUndistortRectifyMap(K, distCoeffs, cv::Mat(), K,
                                  raw.size(), CV_16SC2, map1, map2);
    }

    // Use remap for efficient undistortion
    cv::remap(raw, frame.image, map1, map2, cv::INTER_LINEAR);
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
    std::vector<cv::Point2f> features;
    cv::goodFeaturesToTrack(
        img,
        features,
        MAX_FEATURES,
        GFTT_QUALITY,
        GFTT_MIN_DIST,
        cv::noArray(),
        GFTT_BLOCK_SIZE,
        false,
        0.04);

    // Fill the points structure
    for (const auto &pt : features)
    {
      detectedPoints.pts.push_back(pt);
      detectedPoints.ids.push_back(nextFeatureID++);
      detectedPoints.isTracked.push_back(false);
      detectedPoints.age.push_back(1);
    }

    return detectedPoints;
  }

  /**
   *
   */
  Points featureTracking()
  {
    // Temporary variables
    Points filteredPoints;
    std::vector<cv::Point2f> trackedPoints;
    std::vector<uchar> status;
    std::vector<float> error;

    // Optical Flow
    cv::calcOpticalFlowPyrLK(
        prevFrame.image,
        currFrame.image,
        prevPoints.pts,
        trackedPoints,
        status,
        error,
        cv::Size(21, 21),
        2);

    // Results Processing
    for (size_t i = 0; i < status.size(); i++)
    {
      // Checking if tracking was successful
      if (!status[i])
        continue;

      // Check tracking error
      if (error[i] > MAX_TRACKING_ERROR_PX)
        continue;

      // Check if points are inside image borders
      if (!inImage(trackedPoints[i], currFrame.image) || !inImage(prevPoints.pts[i], prevFrame.image))
        continue;

      // Store tracked point
      filteredPoints.pts.push_back(trackedPoints[i]);
      filteredPoints.ids.push_back(prevPoints.ids[i]);
      filteredPoints.isTracked.push_back(true);
      filteredPoints.age.push_back(prevPoints.age[i] + 1);
    }

    return filteredPoints;
  }

  /**
   *
   */
  void replenishFeatures(const cv::Mat &img)
  {
    int featuresToAdd = MAX_FEATURES - static_cast<int>(currPoints.pts.size());
    if (featuresToAdd <= 0)
      return;

    std::vector<cv::Point2f> detectedPts;
    cv::goodFeaturesToTrack(
        img,
        detectedPts,
        MAX_FEATURES,
        GFTT_QUALITY,
        GFTT_MIN_DIST,
        cv::noArray(),
        GFTT_BLOCK_SIZE,
        false,
        0.04);

    for (const auto &pt : detectedPts)
    {
      bool alreadyTracked = false;
      for (const auto &existingPt : currPoints.pts)
      {
        if (cv::norm(pt - existingPt) < 1.0)
        {
          alreadyTracked = true;
          break;
        }
      }
      if (alreadyTracked)
        continue;

      currPoints.pts.push_back(pt);
      currPoints.ids.push_back(nextFeatureID++);
      currPoints.isTracked.push_back(false); // Not yet tracked
      currPoints.age.push_back(1);           // New feature

      featuresToAdd--;
      if (featuresToAdd <= 0)
        break;
    }
  }

  /**
   *
   */
  bool frameProcessing()
  {
    // First Frame Processing
    if (isFirstFrame)
    {
      prevFrame = undistortImage(imgBuffer[frameIdx++]);
      prevPoints = featureDetection(prevFrame.image);
      isFirstFrame = false;
      OK("[Frame " << frameIdx << "] First Frame Processed");
      return false;
    }

    // Not enough 'prevPoints' or Image Acquisition Error
    if (prevPoints.pts.size() < MIN_FEATURES || prevFrame.image.empty())
    {
      WARN("[Frame " << frameIdx + 1 << "] No previous points to track. Detecting new features.");
      prevFrame = undistortImage(imgBuffer[frameIdx++]);
      prevPoints = featureDetection(prevFrame.image);
      return false;
    }

    currFrame = undistortImage(imgBuffer[frameIdx++]);

    // Feature Tracking [Using LK Optical Flow]
    currPoints = featureTracking();

    // Feature Replenishing to maxFeatures
    replenishFeatures(currFrame.image);

    return true;
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