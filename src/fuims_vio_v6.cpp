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

#include <opencv2/core/eigen.hpp>
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
/**
 * VIO State Machine
 * - IDLE: waiting for start command
 * - RUNNING: processing frames
 * - RESET: reset all variables to default
 */
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
 * Image Frame Structure
 * @param image: Undistorted image frame
 * @param timestamp: ROS timestamp of the image frame
 */
struct ImageFrame
{
  cv::Mat image;
  ros::Time timestamp;
};

/**
 * Points Structure
 * @param ids: Unique IDs for each feature point
 * @param pts: 2D coordinates of each feature point
 * @param isTracked: Boolean flags indicating if the point is currently tracked
 * @param age: Age of each feature point in frames
 */
struct Points
{
  std::vector<int> ids;
  std::vector<cv::Point2f> pts;
  std::vector<bool> isTracked;
  std::vector<int> age;
};

/**
 * Keyframe Structure
 * @param index: Keyframe index
 * @param timestamp: ROS timestamp of the keyframe
 * @param image: Undistorted image of the keyframe
 * @param frameID: Original frame index from which the keyframe was created
 * @param points: Feature points associated with the keyframe
 * @param pose: Estimated pose of the keyframe (gtsam::Pose3)
 */
struct Keyframe
{
  int index;
  ros::Time timestamp;
  cv::Mat image;
  int frameID;
  Points points;
  gtsam::Pose3 pose;
};

/** * ENU Structure
 * @param timestamp: ROS timestamp of the GPS fix
 * @param x: East coordinate
 * @param y: North coordinate
 * @param z: Up coordinate
 */
struct ENU
{
  ros::Time timestamp;
  double x, y, z;
};

/**
 * Barometric Altitude Structure
 * @param timestamp: ROS timestamp of the altitude measurement
 * @param altitude: Altitude value (stored as RELATIVE altitude in ENU frame)
 */
struct BaroAltitude
{
  ros::Time timestamp;
  double altitude; // NOTE: this is stored as RELATIVE altitude (ENU up) after fixes
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
//  WGS84 Constants
// =========================================================
constexpr double a = 6378137.0;
constexpr double f = 1.0 / 298.257223563;
constexpr double b = a * (1 - f);
constexpr double e2 = 1 - (b * b) / (a * a);

// =========================================================
// Referential Coordinate Conversion Functions
// =========================================================
/**
 * Convert geodetic coordinates (lat, lon, alt) to ECEF coordinates (X, Y, Z).
 * @param lat: Latitude in radians
 * @param lon: Longitude in radians
 * @param alt: Altitude in meters
 * @param X: Output ECEF X coordinate
 * @param Y: Output ECEF Y coordinate
 * @param Z: Output ECEF Z coordinate
 *
 * IMPORTANT NOTE:
 * - lat/lon MUST be in RADIANS when calling this function.
 */
void geodeticToECEF(double lat, double lon, double alt, double &X, double &Y, double &Z)
{
  // lat/lon MUST be in radians
  double sinLat = sin(lat), cosLat = cos(lat);
  double sinLon = sin(lon), cosLon = cos(lon);

  double N = a / sqrt(1 - e2 * sinLat * sinLat);

  X = (N + alt) * cosLat * cosLon;
  Y = (N + alt) * cosLat * sinLon;
  Z = (b * b / (a * a) * N + alt) * sinLat;
}

/**
 * Convert ECEF coordinates (X, Y, Z) to ENU coordinates (x, y, z) relative to a reference point (lat0, lon0, alt0).
 * @param X: ECEF X coordinate
 * @param Y: ECEF Y coordinate
 * @param Z: ECEF Z coordinate
 * @param lat0: Reference latitude in radians
 * @param lon0: Reference longitude in radians
 * @param alt0: Reference altitude in meters
 * @return ENU structure containing the converted coordinates
 *
 * IMPORTANT NOTE:
 * - lat0/lon0 MUST be in RADIANS when calling this function.
 */
ENU ecefToENU(double X, double Y, double Z, double lat0, double lon0, double alt0)
{
  double X0, Y0, Z0;
  geodeticToECEF(lat0, lon0, alt0, X0, Y0, Z0);

  double dx = X - X0;
  double dy = Y - Y0;
  double dz = Z - Z0;

  double sinLat = sin(lat0), cosLat = cos(lat0);
  double sinLon = sin(lon0), cosLon = cos(lon0);

  ENU enu;
  enu.x = -sinLon * dx + cosLon * dy;                                 // East
  enu.y = -cosLon * sinLat * dx - sinLon * sinLat * dy + cosLat * dz; // North
  enu.z = cosLon * cosLat * dx + sinLon * cosLat * dy + sinLat * dz;  // Up

  return enu;
}

/**
 * Convert a quaternion expressed in NED world frame to ENU world frame.
 *
 * IMPORTANT NOTE:
 * DJI/flight-stacks can represent attitude quaternions with different conventions:
 * - body->world or world->body
 * - NED or ENU
 *
 * Here we assume input quaternion represents: R_ned_body (body frame orientation wrt NED world),
 * and we want: R_enu_body.
 *
 * For that case, the correct operation is:
 *   R_enu_body = R_enu_ned * R_ned_body
 * which corresponds to:
 *   q_enu = q_tf * q_ned
 */
Eigen::Quaterniond convertNEDtoENU(const Eigen::Quaterniond &q_ned)
{
  Eigen::Matrix3d R_ned_to_enu;
  R_ned_to_enu << 0, 1, 0,
      1, 0, 0,
      0, 0, -1;

  Eigen::Quaterniond q_tf(R_ned_to_enu);

  // Transform composition (NOT sandwich) for body->world convention:
  return (q_tf * q_ned).normalized();
}

// =========================================================
// TrajectoryEvaluator Class
// =========================================================
class TrajectoryEvaluator
{
public:
  /**
   * Add a pose estimate (no GPS association here anymore).
   */
  void addPose(const ros::Time &stamp, const gtsam::Pose3 &estPose)
  {
    Eigen::Vector3d p_est(estPose.translation().x(),
                          estPose.translation().y(),
                          estPose.translation().z());

    estPositions_.push_back(p_est);
    estTimestamps_.push_back(stamp);
  }

  /**
   * Compute and print RMSE and MAX position errors (uses simple nearest-neighbor for console output).
   */
  void computeMetrics(const std::deque<ENU> &gpsEnuBuffer, double maxGpsDt = 0.25)
  {
    if (estPositions_.empty())
    {
      ERROR("[TrajectoryEvaluator] No estimated poses to evaluate.");
      return;
    }

    double sumSq = 0.0;
    double maxErr = 0.0;
    int validCount = 0;

    for (size_t i = 0; i < estPositions_.size(); ++i)
    {
      // Find closest GPS
      double minDt = 1e9;
      Eigen::Vector3d p_gt;
      for (const auto &enu : gpsEnuBuffer)
      {
        double dt = std::abs((enu.timestamp - estTimestamps_[i]).toSec());
        if (dt < minDt)
        {
          minDt = dt;
          p_gt = Eigen::Vector3d(enu.x, enu.y, enu.z);
        }
      }

      if (minDt < maxGpsDt)
      {
        Eigen::Vector3d err = estPositions_[i] - p_gt;
        double norm = err.norm();
        sumSq += norm * norm;
        if (norm > maxErr)
          maxErr = norm;
        validCount++;
      }
    }

    if (validCount == 0)
    {
      ERROR("[TrajectoryEvaluator] No valid pose-GPS matches found.");
      return;
    }

    double rmse = std::sqrt(sumSq / validCount);

    OK("[TrajectoryEvaluator] Pose Evaluation:");
    OK("  APE RMSE: " << std::fixed << std::setprecision(3) << rmse << " m");
    OK("  APE MAX:  " << std::fixed << std::setprecision(3) << maxErr << " m");
    OK("  Samples:  " << validCount << " / " << estPositions_.size());
  }

  /**
   * Write trajectory data to a CSV file.
   * Writes estimated poses AND full GPS ground truth separately.
   */
  void writeCSVWithParams(const std::unordered_map<std::string, std::string> &params,
                          const std::deque<ENU> &gpsEnuBuffer)
  {
    const std::string dir = "/home/tony/Desktop/MEEC-SA/2º Ano/FUIMS/Resultados/";

    // Get current time
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm *tm_ptr = std::localtime(&now_time);

    std::ostringstream oss;
    oss << "vio_results_"
        << std::put_time(tm_ptr, "%Y-%m-%d_%H-%M-%S")
        << ".csv";

    std::string filename = dir + oss.str();

    std::ofstream file(filename);
    if (!file.is_open())
    {
      ERROR("[TrajectoryEvaluator] Failed to open file: " << filename);
      return;
    }

    // --- Write parameters ---
    file << "# ========== VIO PARAMETERS ==========\n";
    for (const auto &[key, value] : params)
    {
      file << "# " << key << ": " << value << "\n";
    }
    file << "# ====================================\n";
    file << "\n";

    // --- Write ESTIMATED trajectory ---
    file << "# ========== ESTIMATED TRAJECTORY ==========\n";
    file << "est_timestamp,est_x,est_y,est_z\n";
    for (size_t i = 0; i < estPositions_.size(); ++i)
    {
      file << std::fixed << std::setprecision(6)
           << estTimestamps_[i].toSec() << ","
           << estPositions_[i].x() << "," 
           << estPositions_[i].y() << "," 
           << estPositions_[i].z() << "\n";
    }

    file << "\n";

    // --- Write GROUND TRUTH (full GPS) trajectory ---
    file << "# ========== GROUND TRUTH TRAJECTORY ==========\n";
    file << "gt_timestamp,gt_x,gt_y,gt_z\n";
    for (const auto &enu : gpsEnuBuffer)
    {
      file << std::fixed << std::setprecision(6)
           << enu.timestamp.toSec() << ","
           << enu.x << "," 
           << enu.y << "," 
           << enu.z << "\n";
    }

    file.close();
    OK("[TrajectoryEvaluator] CSV written to: " << filename);
  }

  /**
   * Clear all stored trajectory data.
   */
  void clear()
  {
    estPositions_.clear();
    estTimestamps_.clear();
  }

private:
  std::vector<Eigen::Vector3d> estPositions_;
  std::vector<ros::Time> estTimestamps_;
};

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
    // Setup signal handler
    signal(SIGINT, signalHandler);

    // Loading parameters and Dynamic Reconfigure
    loadParams();
    drCallback = boost::bind(&vioManager::configCallback, this, _1, _2);
    drServer.setCallback(drCallback);

    // Buffer ROSBAG messages
    bufferRosbagMessages(BAG_PATH);

    // ROS Publishers and Subscribers
    matchingPub = nh.advertise<sensor_msgs::Image>("vio/feature_matches", 1);
    featurePub = nh.advertise<sensor_msgs::Image>("vio/feature_ages", 1);
    posePub = nh.advertise<geometry_msgs::PoseStamped>("vio/pose", 1);
    pathPub = nh.advertise<nav_msgs::Path>("vio/path", 1);
    gtPathPub = nh.advertise<nav_msgs::Path>("vio/ground_truth", 1, true);
    stateSub = nh.subscribe<std_msgs::UInt8>("vio/state", 1, &vioManager::stateCallback, this);

    // Camera Undistortion Maps
    cv::initUndistortRectifyMap(
        K,
        distCoeffs,
        cv::Mat(),
        K,
        cv::Size(1920, 1080),
        CV_16SC2,
        map1, map2);

    // ====== ISAM2 Setup ======
    isamParams.relinearizeThreshold = 0.1;
    isamParams.relinearizeSkip = 1;
    rebuildISAM(); // adds x0 prior

    // Initialize keyframes
    prevKeyframe.index = -1;
    currKeyframe.index = -1;

    // Publishing the Ground Truth
    nav_msgs::Path gtPath;
    gtPath.header.frame_id = "map";

    for (const auto &gps : gpsEnuBuffer)
    {
      geometry_msgs::PoseStamped pose;
      pose.header.stamp = gps.timestamp;
      pose.header.frame_id = "map";
      pose.pose.position.x = gps.x;
      pose.pose.position.y = gps.y;
      pose.pose.position.z = gps.z;
      pose.pose.orientation.w = 1.0;

      gtPath.poses.push_back(pose);
    }
    gtPathPub.publish(gtPath);
    ROS_INFO_STREAM(GREEN << "Published ground truth path with " << gtPath.poses.size() << " poses" << CLEAR);

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
        if (frameIdx >= static_cast<int>(imgBuffer.size()))
        {
          stopLoopTimerAndReport();
          OK("[VIO Manager] Processing Ended! Changing state to RESET");

          evaluator_.computeMetrics(gpsEnuBuffer);
          evaluator_.writeCSVWithParams(paramLog_, gpsEnuBuffer);

          vio_state_ = vioState::RESET;
        }
        else
        {
          // Feature Detection + Feature Tracking
          bool frameSuccess = frameProcessing();

          // Keyframe decision + creation
          bool newKeyframe = false;
          if (frameSuccess) // only evaluate keyframes when we have a valid current frame
          {
            newKeyframe = keyframeProcessing();
          }

          // Estimation: ONLY on NEW keyframes
          if (newKeyframe)
          {
            bool estimateSuccess = poseEstimation();
            (void)estimateSuccess;
          }

          if (frameSuccess)
          {
            debugImageFeatureAges();
            debugImageFeatureMatching();

            prevFrame = currFrame;
            prevPoints = currPoints;
          }
        }
        break;

      case vioState::RESET:
        WARN("[VIO Manager] Reset Command Received! Setting variables to default...");
        stopLoopTimerAndReport();
        statesReset();
        OK("[VIO Manager] Variables reset!");
        vio_state_ = vioState::IDLE;
        break;

      default:
        ERROR("[VIO Manager] State Machine is Broken!!");
        ERROR("[VIO Manager] Terminating process!!");
        g_requestShutdown = true;
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
  // ROS Variables
  ros::NodeHandle nh;
  ros::Publisher matchingPub, featurePub, posePub, pathPub, gtPathPub;
  ros::Subscriber stateSub, resetSub;
  nav_msgs::Path pathMsg;

  // Dynamic Reconfigure
  dynamic_reconfigure::Server<fuims::vioParamsConfig> drServer;
  dynamic_reconfigure::Server<fuims::vioParamsConfig>::CallbackType drCallback;

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

  // === VO and Inertial Prior Noise ===
  double VO_NOISE_ROT, VO_NOISE_TRANS, ROT_PRIOR_NOISE, TRANS_PRIOR_NOISE, ALT_PRIOR_NOISE, GPS_NOISE;

  // ROS Topic Messages Buffers
  std::deque<sensor_msgs::CompressedImageConstPtr> imgBuffer;
  std::deque<sensor_msgs::NavSatFixConstPtr> gpsBuffer; // (still unused but kept)
  std::deque<ENU> gpsEnuBuffer;
  std::deque<BaroAltitude> altBuffer;
  std::deque<geometry_msgs::Vector3StampedConstPtr> velBuffer;
  std::deque<geometry_msgs::QuaternionStampedConstPtr> quatBuffer;

  // ENU Origin Reference
  bool enuOriginInitialized = false;
  double lat0_rad = 0.0;
  double lon0_rad = 0.0;
  double alt0 = 0.0;

  // Camera Parameters
  const cv::Mat K = (cv::Mat_<double>(3, 3) << 1372.76165, 0.0, 960.45289,
                     0.0, 1372.14817, 515.00383,
                     0.0, 0.0, 1.0);
  const cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << 0.048300, -0.044905, -0.003731, -0.001349, 0);
  cv::Mat map1, map2;

  // Status variables
  vioState vio_state_ = vioState::IDLE;
  vioState last_vio_state_ = vioState::IDLE;
  bool isFirstFrame = true;
  bool hasFirstKF = false;
  int frameIdx = 0;
  int keyframeIdx = 0;

  // VO-Related Variables
  ImageFrame currFrame, prevFrame;
  Points currPoints, prevPoints, trackedPoints;
  int nextFeatureID = 0;

  // ===== Execution Time Metric =====
  bool loopTimingActive_ = false;
  ros::WallTime loopStartWall_;

  // Keyframes
  Keyframe prevKeyframe, currKeyframe;

  // GTSAM related
  gtsam::ISAM2Params isamParams;
  gtsam::ISAM2 isam;
  gtsam::NonlinearFactorGraph graph;
  gtsam::Values values;

  // Trajectory Evaluator
  TrajectoryEvaluator evaluator_;
  std::unordered_map<std::string, std::string> paramLog_;

  // =========================================================
  // Helpers
  // =========================================================
  /**
   * Rebuild the ISAM2 object from scratch, resetting graph and values.
   * This includes adding the prior factor at x0.
   */
  void rebuildISAM()
  {
    isam = gtsam::ISAM2(isamParams);
    graph.resize(0);
    values.clear();

    // Prior at x0
    gtsam::Pose3 prior;
    auto priorNoise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1).finished());

    graph.add(gtsam::PriorFactor<gtsam::Pose3>(gtsam::Symbol('x', 0), prior, priorNoise));
    values.insert(gtsam::Symbol('x', 0), prior);

    isam.update(graph, values);
    graph.resize(0);
    values.clear();
  }

  /**
   * Load parameters from ROS parameter server.
   */
  void loadParams()
  {
    nh.param("MIN_FEATURES", MIN_FEATURES, 25);
    nh.param("MAX_FEATURES", MAX_FEATURES, 250);
    nh.param("MIN_TRACKED_FEATURES", MIN_TRACKED_FEATURES, 35);
    nh.param("MAX_TRACKING_ERROR_PX", MAX_TRACKING_ERROR_PX, 7.5f);
    nh.param("MAX_TRACKING_AGE", MAX_TRACKING_AGE, 5);
    nh.param("KF_FEATURE_THRESHOLD", KF_FEATURE_THRESHOLD, 75);
    nh.param("KF_PARALLAX_THRESHOLD", KF_PARALLAX_THRESHOLD, 28);
    nh.param("GPS_PRIOR_INTERVAL", GPS_PRIOR_INTERVAL, 10);

    nh.param("GFTT_MAX_FEATURES", GFTT_MAX_FEATURES, 500);
    nh.param("GFTT_BLOCK_SIZE", GFTT_BLOCK_SIZE, 3);
    nh.param("GFTT_QUALITY", GFTT_QUALITY, 0.15);
    nh.param("GFTT_MIN_DIST", GFTT_MIN_DIST, 26.0);

    nh.param("KLT_MAX_LEVEL", KLT_MAX_LEVEL, 4);
    nh.param("KLT_ITERS", KLT_ITERS, 30);
    nh.param("KLT_BORDER_MARGIN", KLT_BORDER_MARGIN, 10);
    nh.param("KLT_EPS", KLT_EPS, 0.01);
    nh.param("KLT_MIN_EIG", KLT_MIN_EIG, 1e-4);
    nh.param("KLT_FB_THRESH_PX", KLT_FB_THRESH_PX, 1.0f);

    nh.param("CAMERA_RATE", CAMERA_RATE, 15);
    nh.param("INERTIAL_RATE", INERTIAL_RATE, 30);

    nh.param("VO_NOISE_ROT", VO_NOISE_ROT, 0.2);
    nh.param("VO_NOISE_TRANS", VO_NOISE_TRANS, 0.5);
    nh.param("ROT_PRIOR_NOISE", ROT_PRIOR_NOISE, 0.05);
    nh.param("TRANS_PRIOR_NOISE", TRANS_PRIOR_NOISE, 0.1);
    nh.param("ALT_PRIOR_NOISE", ALT_PRIOR_NOISE, 0.5);
    nh.param("GPS_NOISE", GPS_NOISE, 1.0);

    paramLog_ = {
        {"MAX_FEATURES", std::to_string(MAX_FEATURES)},
        {"MIN_FEATURES", std::to_string(MIN_FEATURES)},
        {"MIN_TRACKED_FEATURES", std::to_string(MIN_TRACKED_FEATURES)},
        {"MAX_TRACKING_ERROR_PX", std::to_string(MAX_TRACKING_ERROR_PX)},
        {"MAX_TRACKING_AGE", std::to_string(MAX_TRACKING_AGE)},
        {"KF_FEATURE_THRESHOLD", std::to_string(KF_FEATURE_THRESHOLD)},
        {"KF_PARALLAX_THRESHOLD", std::to_string(KF_PARALLAX_THRESHOLD)},
        {"GPS_PRIOR_INTERVAL", std::to_string(GPS_PRIOR_INTERVAL)},
        {"GFTT_MAX_FEATURES", std::to_string(GFTT_MAX_FEATURES)},
        {"GFTT_BLOCK_SIZE", std::to_string(GFTT_BLOCK_SIZE)},
        {"GFTT_QUALITY", std::to_string(GFTT_QUALITY)},
        {"GFTT_MIN_DIST", std::to_string(GFTT_MIN_DIST)},
        {"KLT_MAX_LEVEL", std::to_string(KLT_MAX_LEVEL)},
        {"KLT_ITERS", std::to_string(KLT_ITERS)},
        {"KLT_BORDER_MARGIN", std::to_string(KLT_BORDER_MARGIN)},
        {"KLT_EPS", std::to_string(KLT_EPS)},
        {"KLT_MIN_EIG", std::to_string(KLT_MIN_EIG)},
        {"KLT_FB_THRESH_PX", std::to_string(KLT_FB_THRESH_PX)},
        {"CAMERA_RATE", std::to_string(CAMERA_RATE)},
        {"INERTIAL_RATE", std::to_string(INERTIAL_RATE)},
        {"VO_NOISE_ROT", std::to_string(VO_NOISE_ROT)},
        {"VO_NOISE_TRANS", std::to_string(VO_NOISE_TRANS)},
        {"ROT_PRIOR_NOISE", std::to_string(ROT_PRIOR_NOISE)},
        {"TRANS_PRIOR_NOISE", std::to_string(TRANS_PRIOR_NOISE)},
        {"ALT_PRIOR_NOISE", std::to_string(ALT_PRIOR_NOISE)},
        {"GPS_NOISE", std::to_string(GPS_NOISE)}};
  }

  /**
   * Buffer messages from a ROSBAG file into local buffers for processing.
   * @param bag_path: Path to the ROSBAG file
   */
  void bufferRosbagMessages(const std::string &bag_path)
  {
    rosbag::Bag bag;
    bag.open(bag_path, rosbag::bagmode::Read);

    std::vector<std::string> topics = {
        CAMERA_TOPIC,
        GPS_TOPIC,
        VELOCITY_TOPIC,
        QUATERNION_TOPIC};

    rosbag::View view(bag, rosbag::TopicQuery(topics));

    for (const rosbag::MessageInstance &m : view)
    {
      /* ================= CAMERA ================= */
      if (m.getTopic() == CAMERA_TOPIC)
      {
        auto img = m.instantiate<sensor_msgs::CompressedImage>();
        if (img)
          imgBuffer.push_back(img);
      }

      /* ================= GPS ================= */
      else if (m.getTopic() == GPS_TOPIC)
      {
        auto gps = m.instantiate<sensor_msgs::NavSatFix>();
        if (!gps)
          continue;

        // USER NOTE APPLIED:
        // You stated that LAT/LON arrive in RADIANS already.
        // Therefore, DO NOT convert degrees -> radians here.
        // If this is wrong for your bag, ENU will be totally wrong.
        const double lat_rad = gps->latitude;
        const double lon_rad = gps->longitude;
        const double alt = gps->altitude;

        // Optional sanity check: if lat/lon look like degrees, warn
        if (std::abs(lat_rad) > M_PI + 0.2 || std::abs(lon_rad) > 2.0 * M_PI + 0.2)
        {
          WARN("[GPS] lat/lon magnitude looks like degrees, but code assumes radians!");
          WARN("  lat=" << lat_rad << " lon=" << lon_rad << " (expected radians)");
        }

        // Initialize ENU origin at first GPS fix
        if (!enuOriginInitialized)
        {
          lat0_rad = lat_rad;
          lon0_rad = lon_rad;
          alt0 = alt;
          enuOriginInitialized = true;

          INFO("[ENU Origin] Set from first GPS fix (ASSUMED RADIANS):");
          INFO("  lat(rad) = " << lat0_rad << "  lat(deg) = " << lat0_rad * 180.0 / M_PI);
          INFO("  lon(rad) = " << lon0_rad << "  lon(deg) = " << lon0_rad * 180.0 / M_PI);
          INFO("  alt(m)   = " << alt0);
        }

        // WGS-84 → ECEF
        double X, Y, Z;
        geodeticToECEF(lat_rad, lon_rad, alt, X, Y, Z);

        // ECEF → ENU
        ENU enu = ecefToENU(X, Y, Z, lat0_rad, lon0_rad, alt0);
        enu.timestamp = gps->header.stamp;
        gpsEnuBuffer.push_back(enu);

        // Store altitude RELATIVE (consistent with ENU/map frame)
        BaroAltitude baro;
        baro.timestamp = gps->header.stamp;
        baro.altitude = enu.z; // RELATIVE vertical displacement
        altBuffer.push_back(baro);

        // DEBUG: First point should be ~0,0,0
        if (gpsEnuBuffer.size() == 1)
        {
          INFO("[ENU Conversion] First GPS ENU should be near (0,0,0)");
          INFO("  ENU: x=" << enu.x << " y=" << enu.y << " z=" << enu.z);
        }

        // DEBUG: show first few
        if (gpsEnuBuffer.size() <= 5)
        {
          INFO("[ENU Conversion] GPS ENU #" << gpsEnuBuffer.size());
          INFO("  Raw GPS: lat(rad)=" << gps->latitude
                                      << ", lon(rad)=" << gps->longitude
                                      << ", alt=" << alt);
          INFO("  ENU: x=" << enu.x << ", y=" << enu.y << ", z=" << enu.z);
          INFO("  Altitude stored (REL, enu.z): " << baro.altitude);
        }
      }

      /* ================= VELOCITY ================= */
      else if (m.getTopic() == VELOCITY_TOPIC)
      {
        auto vel = m.instantiate<geometry_msgs::Vector3Stamped>();
        if (!vel)
          continue;

        geometry_msgs::Vector3StampedPtr enuVel(new geometry_msgs::Vector3Stamped(*vel));

        // Assumption: DJI publishes NEU (North, East, Up)
        // Convert to ENU (East, North, Up)
        enuVel->header = vel->header;
        enuVel->vector.x = vel->vector.y; // East
        enuVel->vector.y = vel->vector.x; // North
        enuVel->vector.z = vel->vector.z; // Up (if your source is NED, you MUST flip sign!)

        velBuffer.push_back(enuVel);
      }

      /* ================= QUATERNION ================= */
      else if (m.getTopic() == QUATERNION_TOPIC)
      {
        auto quat = m.instantiate<geometry_msgs::QuaternionStamped>();
        if (!quat)
          continue;

        Eigen::Quaterniond q_ned(
            quat->quaternion.w,
            quat->quaternion.x,
            quat->quaternion.y,
            quat->quaternion.z);

        Eigen::Quaterniond q_enu = convertNEDtoENU(q_ned);

        geometry_msgs::QuaternionStampedPtr enuQuat(new geometry_msgs::QuaternionStamped(*quat));
        enuQuat->header = quat->header;
        enuQuat->quaternion.w = q_enu.w();
        enuQuat->quaternion.x = q_enu.x();
        enuQuat->quaternion.y = q_enu.y();
        enuQuat->quaternion.z = q_enu.z();

        quatBuffer.push_back(enuQuat);
      }
    }

    if (!altBuffer.empty())
    {
      double min_alt = altBuffer.front().altitude;
      double max_alt = altBuffer.front().altitude;

      for (const auto &baro : altBuffer)
      {
        min_alt = std::min(min_alt, baro.altitude);
        max_alt = std::max(max_alt, baro.altitude);
      }

      INFO("[DEBUG] Relative altitude range (stored as ENU.z):");
      INFO("  Min: " << min_alt << " m");
      INFO("  Max: " << max_alt << " m");
      INFO("  Δ:   " << (max_alt - min_alt) << " m");
    }
    else
    {
      WARN("[DEBUG] Altitude buffer is empty. No altitude range to show.");
    }

    bag.close();

    INFO("ROSBAG Messages Processed (ENU)");
    OK("Image messages total: " << imgBuffer.size());
    OK("Quaternion ENU messages total: " << quatBuffer.size());
    OK("Velocity ENU messages total: " << velBuffer.size());
    OK("GPS ENU messages total: " << gpsEnuBuffer.size());
    OK("Altitude (REL) messages total: " << altBuffer.size());
  }

  /**
   * Reset all VIO states and variables to default.
   */
  void statesReset()
  {
    // Core runtime state
    isFirstFrame = true;
    hasFirstKF = false;
    frameIdx = 0;
    keyframeIdx = 0;
    nextFeatureID = 0;

    // Frames / points
    currFrame = ImageFrame();
    prevFrame = ImageFrame();
    currPoints = Points();
    prevPoints = Points();
    trackedPoints = Points();

    // Keyframes
    currKeyframe = Keyframe();
    prevKeyframe = Keyframe();
    prevKeyframe.index = -1;
    currKeyframe.index = -1;

    // Path
    pathMsg.poses.clear();
    pathMsg.header.frame_id = "map";

    // Reset trajectory evaluator (IMPORTANT: prevents duplicate data in CSV)
    evaluator_.clear();

    // Reset optimizer (VERY IMPORTANT)
    rebuildISAM();
  }

  /**
   * Clear the estimated path in RViz.
   */
  void clearEstimatedPath()
  {
    pathMsg.poses.clear();
    pathMsg.header.frame_id = "map";
    pathMsg.header.stamp = ros::Time::now();

    // Publish empty path so RViz clears immediately
    pathPub.publish(pathMsg);

    INFO("[RVIZ] Estimated path cleared.");
  }

  /**
   * Check if a point is within the valid image area (considering a margin).
   * @param p: 2D point to check
   * @param img: Image for boundary reference
   * @return true if the point is within the valid area, false otherwise
   */
  bool inImage(const cv::Point2f &p, const cv::Mat &img)
  {
    return p.x >= 16 && p.y >= 16 && p.x < img.cols - 16 && p.y < img.rows - 16;
  }

  /**
   * Draw feature matches between two images.
   * @param img1: First image
   * @param img2: Second image
   * @param pts1: Feature points in the first image
   * @param pts2: Corresponding feature points in the second image
   * @return Visualization image with matches drawn
   */
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
      cv::Point2f pt2 = pts2[i] + cv::Point2f(static_cast<float>(img1.cols), 0);
      cv::line(vis, pt1, pt2, cv::Scalar(0, 255, 0), 1);
      cv::circle(vis, pt1, 3, cv::Scalar(255, 0, 0), -1);
      cv::circle(vis, pt2, 3, cv::Scalar(0, 0, 255), -1);
    }

    return vis;
  }

  /**
   * Draw feature points with colors indicating their ages.
   * @param img: Image to draw on
   * @param points: Points structure containing feature points and their ages
   * @return Visualization image with feature ages drawn
   */
  cv::Mat drawFeatureAges(const cv::Mat &img, const Points &points)
  {
    cv::Mat vis;
    cv::cvtColor(img, vis, cv::COLOR_GRAY2BGR);

    for (size_t i = 0; i < points.pts.size(); ++i)
    {
      int age = points.age[i];
      cv::Scalar color;

      if (age >= MAX_TRACKING_AGE)
        color = cv::Scalar(0, 255, 0);
      else
        color = cv::Scalar(0, 0, 255);

      cv::circle(vis, points.pts[i], 3, color, -1);
    }

    return vis;
  }

  /**
   * Publish debug image showing feature ages.
   */
  void debugImageFeatureAges()
  {
    if (currFrame.image.empty())
      return;
    cv::Mat ageVis = drawFeatureAges(currFrame.image, currPoints);
    sensor_msgs::ImagePtr ageMsg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", ageVis).toImageMsg();
    ageMsg->header.stamp = currFrame.timestamp;
    featurePub.publish(ageMsg);
  }

  /**
   * Publish debug image showing feature matches between previous and current frames.
   */
  void debugImageFeatureMatching()
  {
    if (prevFrame.image.empty() || currFrame.image.empty())
      return;

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

  /**
   * Start the loop execution timer.
   */
  void startLoopTimer()
  {
    loopStartWall_ = ros::WallTime::now();
    loopTimingActive_ = true;
    INFO("[Execution Timer] Loop timer started.");
  }

  /**
   * Stop the loop execution timer and report the elapsed time.
   */
  void stopLoopTimerAndReport()
  {
    if (!loopTimingActive_)
      return;

    ros::WallDuration dt = ros::WallTime::now() - loopStartWall_;
    loopTimingActive_ = false;

    OK("[Execution Timer] Full RUNNING loop execution time: " << std::fixed << std::setprecision(3)
                                                              << dt.toSec() << " seconds");
  }

  // =========================================================
  // Callbacks
  // =========================================================
  /**
   * Dynamic Reconfigure callback to update VIO parameters at runtime.
   * @param config: Configuration object with updated parameters
   * @param level: Level of change (not used)
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

    VO_NOISE_ROT = config.VO_NOISE_ROT;
    VO_NOISE_TRANS = config.VO_NOISE_TRANS;
    ROT_PRIOR_NOISE = config.ROT_PRIOR_NOISE;
    TRANS_PRIOR_NOISE = config.TRANS_PRIOR_NOISE;
    ALT_PRIOR_NOISE = config.ALT_PRIOR_NOISE;
    GPS_NOISE = config.GPS_NOISE;

    paramLog_ = {
        {"MAX_FEATURES", std::to_string(MAX_FEATURES)},
        {"MIN_FEATURES", std::to_string(MIN_FEATURES)},
        {"MIN_TRACKED_FEATURES", std::to_string(MIN_TRACKED_FEATURES)},
        {"MAX_TRACKING_ERROR_PX", std::to_string(MAX_TRACKING_ERROR_PX)},
        {"MAX_TRACKING_AGE", std::to_string(MAX_TRACKING_AGE)},
        {"KF_FEATURE_THRESHOLD", std::to_string(KF_FEATURE_THRESHOLD)},
        {"KF_PARALLAX_THRESHOLD", std::to_string(KF_PARALLAX_THRESHOLD)},
        {"GPS_PRIOR_INTERVAL", std::to_string(GPS_PRIOR_INTERVAL)},
        {"GFTT_MAX_FEATURES", std::to_string(GFTT_MAX_FEATURES)},
        {"GFTT_BLOCK_SIZE", std::to_string(GFTT_BLOCK_SIZE)},
        {"GFTT_QUALITY", std::to_string(GFTT_QUALITY)},
        {"GFTT_MIN_DIST", std::to_string(GFTT_MIN_DIST)},
        {"KLT_MAX_LEVEL", std::to_string(KLT_MAX_LEVEL)},
        {"KLT_ITERS", std::to_string(KLT_ITERS)},
        {"KLT_BORDER_MARGIN", std::to_string(KLT_BORDER_MARGIN)},
        {"KLT_EPS", std::to_string(KLT_EPS)},
        {"KLT_MIN_EIG", std::to_string(KLT_MIN_EIG)},
        {"KLT_FB_THRESH_PX", std::to_string(KLT_FB_THRESH_PX)},
        {"CAMERA_RATE", std::to_string(CAMERA_RATE)},
        {"INERTIAL_RATE", std::to_string(INERTIAL_RATE)},
        {"VO_NOISE_ROT", std::to_string(VO_NOISE_ROT)},
        {"VO_NOISE_TRANS", std::to_string(VO_NOISE_TRANS)},
        {"ROT_PRIOR_NOISE", std::to_string(ROT_PRIOR_NOISE)},
        {"TRANS_PRIOR_NOISE", std::to_string(TRANS_PRIOR_NOISE)},
        {"ALT_PRIOR_NOISE", std::to_string(ALT_PRIOR_NOISE)},
        {"GPS_NOISE", std::to_string(GPS_NOISE)}};

    INFO("Dynamic reconfigure updated VIO parameters.");
    WARN("Changing VIO State to RESET");
    vio_state_ = vioState::RESET;
  }

  /**
   * Callback for VIO state changes.
   * @param msg: Message containing the new VIO state
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
      // If we are starting fresh from IDLE, clear RViz estimated path
      if (last_vio_state_ == vioState::IDLE)
      {
        clearEstimatedPath();
        startLoopTimer();
      }

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

    last_vio_state_ = vio_state_;
  }

  // =========================================================
  // Methods
  // =========================================================
  /**
   * Undistort a compressed image message using precomputed maps.
   * @param msg: Compressed image message
   * @return Undistorted ImageFrame
   */
  ImageFrame undistortImage(const sensor_msgs::CompressedImageConstPtr msg)
  {
    ImageFrame frame;
    frame.timestamp = msg->header.stamp;

    cv::Mat raw = cv::imdecode(msg->data, cv::IMREAD_GRAYSCALE);
    if (raw.empty())
      return frame;

    if (map1.empty() || map1.size() != raw.size())
    {
      cv::initUndistortRectifyMap(K, distCoeffs, cv::Mat(), K,
                                  raw.size(), CV_16SC2, map1, map2);
    }

    cv::remap(raw, frame.image, map1, map2, cv::INTER_LINEAR);
    return frame;
  }

  /**
   * Detect features in an image using the GFTT algorithm.
   * @param img: Input image
   * @return Detected Points structure
   */
  Points featureDetection(const cv::Mat &img)
  {
    Points detectedPoints;

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
   * Track features from the previous frame to the current frame using KLT optical flow.
   * @return Filtered Points structure with successfully tracked features
   */
  Points featureTracking()
  {
    Points filteredPoints;
    std::vector<cv::Point2f> tracked;
    std::vector<uchar> status;
    std::vector<float> error;

    cv::calcOpticalFlowPyrLK(
        prevFrame.image,
        currFrame.image,
        prevPoints.pts,
        tracked,
        status,
        error,
        cv::Size(21, 21),
        KLT_MAX_LEVEL);

    for (size_t i = 0; i < status.size(); i++)
    {
      if (!status[i])
        continue;

      if (error[i] > MAX_TRACKING_ERROR_PX)
        continue;

      if (!inImage(tracked[i], currFrame.image) || !inImage(prevPoints.pts[i], prevFrame.image))
        continue;

      filteredPoints.pts.push_back(tracked[i]);
      filteredPoints.ids.push_back(prevPoints.ids[i]);
      filteredPoints.isTracked.push_back(true);
      filteredPoints.age.push_back(prevPoints.age[i] + 1);
    }

    return filteredPoints;
  }

  /**
   * Replenish features in the current frame to maintain a desired number of features.
   * @param img: Current image
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
      currPoints.isTracked.push_back(false);
      currPoints.age.push_back(0);

      featuresToAdd--;
      if (featuresToAdd <= 0)
        break;
    }
  }

  /**
   * Frame processing: detect and track features between frames.
   * @return true if a current frame is successfully processed, false otherwise
   */
  bool frameProcessing()
  {
    if (isFirstFrame)
    {
      prevFrame = undistortImage(imgBuffer[frameIdx++]);
      if (prevFrame.image.empty())
      {
        WARN("[Frame " << frameIdx << "] First frame image empty!");
        return false;
      }
      prevPoints = featureDetection(prevFrame.image);
      isFirstFrame = false;
      OK("[Frame " << frameIdx << "] First Frame Processed");
      return false; // no current frame yet
    }

    if (prevPoints.pts.size() < static_cast<size_t>(MIN_FEATURES) || prevFrame.image.empty())
    {
      WARN("[Frame " << frameIdx + 1 << "] No previous points to track. Detecting new features.");
      prevFrame = undistortImage(imgBuffer[frameIdx++]);
      if (prevFrame.image.empty())
        return false;
      prevPoints = featureDetection(prevFrame.image);
      return false;
    }

    currFrame = undistortImage(imgBuffer[frameIdx++]);
    if (currFrame.image.empty())
    {
      WARN("[Frame " << frameIdx << "] Current frame image empty!");
      return false;
    }

    currPoints = featureTracking();
    replenishFeatures(currFrame.image);

    return true;
  }

  /**
   * Compute the average parallax between matched feature points in two sets.
   * @param kfPoints: Points from the keyframe
   * @param framePoints: Points from the current frame
   * @return Average parallax value
   */
  double computeParallax(Points kfPoints, Points framePoints)
  {
    if (kfPoints.ids.size() != kfPoints.pts.size())
    {
      WARN("KF points ids/pts size mismatch");
      return 0.0;
    }
    if (framePoints.ids.size() != framePoints.pts.size())
    {
      WARN("framePoints ids/pts size mismatch");
      return 0.0;
    }

    int cnt = 0;
    double sum = 0.0;

    std::unordered_map<int, int> id_to_idx;
    id_to_idx.reserve(framePoints.ids.size());
    for (size_t i = 0; i < framePoints.ids.size(); ++i)
    {
      id_to_idx[framePoints.ids[i]] = static_cast<int>(i);
    }

    for (size_t i = 0; i < kfPoints.ids.size(); i++)
    {
      int kfID = kfPoints.ids[i];
      auto it = id_to_idx.find(kfID);
      if (it == id_to_idx.end())
        continue;

      int j = it->second;

      double dx = framePoints.pts[j].x - kfPoints.pts[i].x;
      double dy = framePoints.pts[j].y - kfPoints.pts[i].y;

      sum += std::sqrt(dx * dx + dy * dy);
      cnt++;
    }

    return (cnt == 0) ? 0.0 : (sum / cnt);
  }

  /**
   * Keyframe processing: determine if the current frame should be a keyframe.
   * @return true if a new keyframe is created, false otherwise
   */
  bool keyframeProcessing()
  {
    trackedPoints.pts.clear();
    trackedPoints.ids.clear();
    trackedPoints.isTracked.clear();
    trackedPoints.age.clear();

    // Filter features by age
    for (size_t i = 0; i < currPoints.pts.size(); i++)
    {
      if (currPoints.age[i] >= MAX_TRACKING_AGE)
      {
        trackedPoints.pts.push_back(currPoints.pts[i]);
        trackedPoints.ids.push_back(currPoints.ids[i]);
      }
    }

    bool isKeyframe = false;

    if (!hasFirstKF && trackedPoints.pts.size() >= static_cast<size_t>(MIN_TRACKED_FEATURES))
    {
      isKeyframe = true;
    }
    else if (hasFirstKF && trackedPoints.pts.size() >= static_cast<size_t>(MIN_TRACKED_FEATURES))
    {
      double kfParallax = computeParallax(currKeyframe.points, trackedPoints);

      if (kfParallax > KF_PARALLAX_THRESHOLD)
        isKeyframe = true;
      if (trackedPoints.pts.size() < static_cast<size_t>(KF_FEATURE_THRESHOLD))
        isKeyframe = true;
    }

    if (!isKeyframe)
      return false;

    OK("[Frame " << frameIdx << "] New Keyframe created with "
                 << trackedPoints.pts.size() << " features.");

    const ros::Time kfTime = currFrame.timestamp;

    if (!hasFirstKF)
    {
      // FIRST keyframe MUST correspond to x0 which already exists in iSAM (prior)
      currKeyframe.index = 0;
      keyframeIdx = 1; // next created keyframe will be 1
      currKeyframe.frameID = frameIdx;
      currKeyframe.image = currFrame.image.clone();
      currKeyframe.points = trackedPoints;
      currKeyframe.timestamp = kfTime;

      // use prior pose as initial KF pose
      currKeyframe.pose = gtsam::Pose3();

      hasFirstKF = true;

      // publish initial pose
      geometry_msgs::PoseStamped poseMsg;
      poseMsg.header.stamp = currKeyframe.timestamp;
      poseMsg.header.frame_id = "map";
      poseMsg.pose.position.x = 0.0;
      poseMsg.pose.position.y = 0.0;
      poseMsg.pose.position.z = 0.0;
      poseMsg.pose.orientation.w = 1.0;
      posePub.publish(poseMsg);

      pathMsg.header = poseMsg.header;
      pathMsg.poses.push_back(poseMsg);
      pathPub.publish(pathMsg);

      // prevKeyframe is not valid yet (needs a second KF)
      prevKeyframe.index = -1;
      return true;
    }

    // Subsequent keyframes
    prevKeyframe = currKeyframe;

    currKeyframe.index = keyframeIdx++;
    currKeyframe.frameID = frameIdx;
    currKeyframe.image = currFrame.image.clone();
    currKeyframe.points = trackedPoints;
    currKeyframe.timestamp = kfTime;

    return true;
  }

  /**
   * Interpolate quaternion at a given timestamp using the quaternion buffer.
   * @param time: Timestamp to interpolate at
   * @return Interpolated Eigen::Quaterniond
   */
  Eigen::Quaterniond getInterpolatedQuat(ros::Time time)
  {
    if (quatBuffer.size() < 2)
    {
      ROS_WARN("Quaternion buffer too small, returning identity.");
      return Eigen::Quaterniond::Identity();
    }

    for (size_t i = 1; i < quatBuffer.size(); ++i)
    {
      auto prev = quatBuffer[i - 1];
      auto next = quatBuffer[i];

      if (prev->header.stamp <= time && next->header.stamp >= time)
      {
        double denom = (next->header.stamp - prev->header.stamp).toSec();
        if (denom <= 1e-9)
          return Eigen::Quaterniond::Identity();

        double t = (time - prev->header.stamp).toSec() / denom;

        Eigen::Quaterniond q1(prev->quaternion.w, prev->quaternion.x, prev->quaternion.y, prev->quaternion.z);
        Eigen::Quaterniond q2(next->quaternion.w, next->quaternion.x, next->quaternion.y, next->quaternion.z);

        return q1.slerp(t, q2).normalized();
      }
    }

    // fallback: closest
    auto closest = quatBuffer[0];
    double minTimeDiff = std::abs((closest->header.stamp - time).toSec());

    for (size_t i = 1; i < quatBuffer.size(); ++i)
    {
      double timeDiff = std::abs((quatBuffer[i]->header.stamp - time).toSec());
      if (timeDiff < minTimeDiff)
      {
        minTimeDiff = timeDiff;
        closest = quatBuffer[i];
        if (timeDiff == 0.0)
          break;
      }
    }

    return Eigen::Quaterniond(closest->quaternion.w, closest->quaternion.x,
                              closest->quaternion.y, closest->quaternion.z)
        .normalized();
  }

  /**
   * Compute the average velocity between two timestamps using the velocity buffer.
   * @param t0: Start timestamp
   * @param t1: End timestamp
   * @return Average velocity as an Eigen::Vector3d
   */
  Eigen::Vector3d getAverageVelocity(ros::Time t0, ros::Time t1)
  {
    Eigen::Vector3d sum(0, 0, 0);
    int count = 0;

    for (const auto &vel : velBuffer)
    {
      if (vel->header.stamp >= t0 && vel->header.stamp <= t1)
      {
        sum += Eigen::Vector3d(vel->vector.x, vel->vector.y, vel->vector.z);
        ++count;
      }
    }

    if (count == 0)
      return Eigen::Vector3d::Zero();

    return sum / count;
  }

  /**
   * Compute the average altitude between two timestamps using the altitude buffer.
   * @param t0: Start timestamp
   * @param t1: End timestamp
   * @return Average altitude as a double
   */
  double getAverageAltitude(ros::Time t0, ros::Time t1)
  {
    double sum = 0.0;
    int count = 0;

    for (const auto &baro : altBuffer)
    {
      if (baro.timestamp >= t0 && baro.timestamp <= t1)
      {
        sum += baro.altitude;
        ++count;
      }
    }

    if (count == 0)
      return 0.0;

    return sum / count;
  }

  /**
   * Perform visual odometry estimation between the previous and current keyframes.
   * @return Estimated relative pose as gtsam::Pose3
   */
  gtsam::Pose3 voEstimation()
  {
    std::vector<cv::Point2f> prevKFPoints, currKFPoints;

    std::unordered_map<int, cv::Point2f> currPointMap;
    currPointMap.reserve(currKeyframe.points.ids.size());

    for (size_t j = 0; j < currKeyframe.points.ids.size(); ++j)
    {
      currPointMap[currKeyframe.points.ids[j]] = currKeyframe.points.pts[j];
    }

    for (size_t i = 0; i < prevKeyframe.points.ids.size(); ++i)
    {
      int id = prevKeyframe.points.ids[i];
      auto it = currPointMap.find(id);
      if (it == currPointMap.end())
        continue;

      prevKFPoints.push_back(prevKeyframe.points.pts[i]);
      currKFPoints.push_back(it->second);
    }

    if (prevKFPoints.size() < 8 || currKFPoints.size() < 8)
    {
      WARN("[VO Estimation] Not enough feature correspondences for essential matrix.");
      return gtsam::Pose3();
    }

    cv::Mat inlierMask;
    cv::Mat E = cv::findEssentialMat(prevKFPoints, currKFPoints, K, cv::RANSAC, 0.999, 1.0, inlierMask);

    if (E.empty())
    {
      WARN("[VO Estimation] Essential matrix computation failed.");
      return gtsam::Pose3();
    }

    cv::Mat R, t;
    int inliers = cv::recoverPose(E, prevKFPoints, currKFPoints, K, R, t, inlierMask);

    if (inliers < 8)
    {
      WARN("[VO Estimation] Too few inliers after pose recovery.");
      return gtsam::Pose3();
    }

    Eigen::Matrix3d R_eigen;
    Eigen::Vector3d t_eigen;
    cv::cv2eigen(R, R_eigen);
    cv::cv2eigen(t, t_eigen);

    gtsam::Rot3 rot(R_eigen);
    gtsam::Point3 trans(t_eigen.x(), t_eigen.y(), t_eigen.z());

    OK("[VO Estimation] Relative pose computed with " << inliers << " inliers.");
    return gtsam::Pose3(rot, trans);
  }

  /**
   * Perform pose estimation using visual odometry and inertial priors.
   * @return true if pose estimation is successful, false otherwise
   */
  bool poseEstimation()
  {
    if (!hasFirstKF)
    {
      WARN("[poseEstimation]: No keyframes available.");
      return false;
    }

    // must have a previous KF to estimate between
    if (prevKeyframe.index < 0 || currKeyframe.index <= prevKeyframe.index)
    {
      WARN("[poseEstimation]: Need at least 2 keyframes (prev invalid or curr not newer).");
      return false;
    }

    if (prevKeyframe.points.pts.empty() || currKeyframe.points.pts.empty())
    {
      WARN("[poseEstimation]: Previous or current keyframe has no feature points.");
      return false;
    }

    ros::Time t0 = prevKeyframe.timestamp;
    ros::Time t1 = currKeyframe.timestamp;
    double dt = (t1 - t0).toSec();

    if (dt <= 0.0)
    {
      WARN("[poseEstimation]: Invalid timestamp interval.");
      return false;
    }

    // Inertial priors
    Eigen::Quaterniond quat = getInterpolatedQuat(t1);
    Eigen::Vector3d avgVel = getAverageVelocity(t0, t1);
    double avgAltRel = getAverageAltitude(t0, t1); // RELATIVE altitude now
    Eigen::Vector3d delta_translation = avgVel * dt;

    // VO estimation
    gtsam::Pose3 relativePose = voEstimation();

    // --- SCALE VO TRANSLATION USING VELOCITY MAGNITUDE ---
    {
      Eigen::Vector3d t_vo(relativePose.translation().x(),
                           relativePose.translation().y(),
                           relativePose.translation().z());

      double scale = delta_translation.norm(); // meters
      if (t_vo.norm() > 1e-9 && scale > 1e-4)
      {
        Eigen::Vector3d t_scaled = t_vo.normalized() * scale;

        // Fix sign ambiguity using velocity direction
        if (t_scaled.dot(delta_translation) < 0)
          t_scaled = -t_scaled;

        relativePose = gtsam::Pose3(relativePose.rotation(),
                                    gtsam::Point3(t_scaled.x(), t_scaled.y(), t_scaled.z()));
      }
    }

    gtsam::Symbol prevSym('x', prevKeyframe.index);
    gtsam::Symbol currSym('x', currKeyframe.index);

    // Get previous pose (must exist)
    gtsam::Pose3 prevPose = isam.calculateEstimate().at<gtsam::Pose3>(prevSym);

    // Initial estimate for curr pose (curr key is new)
    gtsam::Pose3 currPoseGuess = prevPose.compose(relativePose);

    // VO Between factor
    auto voNoise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << VO_NOISE_ROT, VO_NOISE_ROT, VO_NOISE_ROT,
         VO_NOISE_TRANS, VO_NOISE_TRANS, VO_NOISE_TRANS)
            .finished());
    graph.add(gtsam::BetweenFactor<gtsam::Pose3>(prevSym, currSym, relativePose, voNoise));

    // Orientation prior
    gtsam::Rot3 rotPrior(quat);
    auto rotNoise = gtsam::noiseModel::Isotropic::Sigma(3, ROT_PRIOR_NOISE);
    graph.add(gtsam::PoseRotationPrior<gtsam::Pose3>(currSym, rotPrior, rotNoise));

    // Velocity translation prior (full xyz)
    gtsam::Point3 expectedTrans = prevPose.translation() +
                                  gtsam::Point3(delta_translation.x(), delta_translation.y(), delta_translation.z());
    auto transNoise = gtsam::noiseModel::Isotropic::Sigma(3, TRANS_PRIOR_NOISE);
    graph.add(gtsam::PoseTranslationPrior<gtsam::Pose3>(currSym, expectedTrans, transNoise));

    // Altitude prior (Z only): keep x/y from expectedTrans, constrain z
    gtsam::Point3 altPrior(expectedTrans.x(), expectedTrans.y(), avgAltRel);
    auto altNoise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(3) << 1e6, 1e6, ALT_PRIOR_NOISE).finished());
    graph.add(gtsam::PoseTranslationPrior<gtsam::Pose3>(currSym, altPrior, altNoise));

    // ==========================
    // GPS factor every N keyframes
    // ==========================
    bool gpsUsed = false;
    Eigen::Vector3d gpsVec(0.0, 0.0, 0.0);

    if (GPS_PRIOR_INTERVAL > 0 && (currKeyframe.index % GPS_PRIOR_INTERVAL == 0))
    {
      double minDiff = 1e9;
      ENU closestGps;
      bool found = false;

      for (const auto &enu : gpsEnuBuffer)
      {
        double diff = std::abs((enu.timestamp - t1).toSec());
        if (diff < minDiff)
        {
          minDiff = diff;
          closestGps = enu;
          found = true;
        }
      }

      // Optional: time gating (recommended)
      // If your GPS is low-rate, pick a reasonable threshold like 0.2~0.3s
      const double GPS_MAX_DT = 0.25;

      if (found && minDiff <= GPS_MAX_DT)
      {
        gtsam::Point3 gps_meas(closestGps.x, closestGps.y, closestGps.z);
        gpsVec = Eigen::Vector3d(gps_meas.x(), gps_meas.y(), gps_meas.z());
        gpsUsed = true;

        // "Before" should compare GPS to the curr initial guess (not prevPose)
        Eigen::Vector3d before(currPoseGuess.translation().x(),
                               currPoseGuess.translation().y(),
                               currPoseGuess.translation().z());

        INFO("[GPS DEBUG] Before update | ||pose_guess - gps|| = "
             << (before - gpsVec).norm() << " m");

        // Use Isotropic or (recommended) Diagonal noise
        // auto gpsNoise = gtsam::noiseModel::Isotropic::Sigma(3, GPS_NOISE);

        auto gpsNoise = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(3) << GPS_NOISE, GPS_NOISE, GPS_NOISE).finished());

        graph.add(gtsam::GPSFactor(currSym, gps_meas, gpsNoise));

        OK("[GPS Factor] Added at keyframe " << currKeyframe.index
                                             << " (dt=" << minDiff << "s)");
      }
      else if (found)
      {
        WARN("[GPS Factor] Closest GPS too far in time (dt=" << minDiff
                                                             << "s). Skipping factor.");
      }
      else
      {
        WARN("[GPS Factor] No GPS data available for keyframe " << currKeyframe.index);
      }
    }

    // Insert initial estimate for current key
    values.insert(currSym, currPoseGuess);

    // ISAM2 Update
    isam.update(graph, values);
    isam.update();
    graph.resize(0);
    values.clear();

    // Save optimized pose
    gtsam::Pose3 optimizedPose = isam.calculateEstimate().at<gtsam::Pose3>(currSym);
    currKeyframe.pose = optimizedPose;

    // Add pose to trajectory evaluator for metrics computation
    evaluator_.addPose(currKeyframe.timestamp, optimizedPose);

    // GPS debug AFTER update (only if we actually used GPS)
    if (gpsUsed)
    {
      Eigen::Vector3d after(optimizedPose.translation().x(),
                            optimizedPose.translation().y(),
                            optimizedPose.translation().z());

      INFO("[GPS DEBUG] After update  | ||pose_opt - gps||   = "
           << (after - gpsVec).norm() << " m");
    }

    // Publish pose
    geometry_msgs::PoseStamped poseMsg;
    poseMsg.header.stamp = currKeyframe.timestamp;
    poseMsg.header.frame_id = "map";
    poseMsg.pose.position.x = optimizedPose.translation().x();
    poseMsg.pose.position.y = optimizedPose.translation().y();
    poseMsg.pose.position.z = optimizedPose.translation().z();
    poseMsg.pose.orientation.x = optimizedPose.rotation().toQuaternion().x();
    poseMsg.pose.orientation.y = optimizedPose.rotation().toQuaternion().y();
    poseMsg.pose.orientation.z = optimizedPose.rotation().toQuaternion().z();
    poseMsg.pose.orientation.w = optimizedPose.rotation().toQuaternion().w();
    posePub.publish(poseMsg);

    // Publish path
    pathMsg.header = poseMsg.header;
    pathMsg.poses.push_back(poseMsg);
    pathPub.publish(pathMsg);

    OK("[Pose Estimation] Pose added for keyframe " << currKeyframe.index);
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