#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/NavSatFix.h>
#include <geometry_msgs/QuaternionStamped.h>
#include <geometry_msgs/Vector3Stamped.h>

#include <cv_bridge/cv_bridge.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <Eigen/Dense>
#include <unordered_set>
#include <unordered_map>

#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Point3.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video.hpp>

#include <signal.h>
#include <fstream>
#include <cstdlib>
#include <iomanip>

#define BAG_PATH "/home/tony/catkin_ws/bags/bags_fuims/agucadoura.bag"
#define CAMERA_TOPIC "/dji_m350/cameras/main/compressed"
#define QUATERNION_TOPIC "/dji_m350/quaternion"
#define VELOCITY_TOPIC "/dji_m350/velocity"
#define GPS_TOPIC "/dji_m350/gps"

#define MAX_FEATURES 250
#define KF_PARALLAX_THRESHOLD 28
#define KF_FEATURE_THRESHOLD 100

// KLT params (ajusta se necessário)
static constexpr int KLT_MAX_LEVEL = 4;
static constexpr int KLT_ITERS = 30;
static constexpr double KLT_EPS = 0.01;
static constexpr double KLT_MIN_EIG = 1e-4;
static constexpr float FB_THRESH_PX = 1.0f; // forward-backward threshold
static constexpr int BORDER_MARGIN = 10;    // reject perto das bordas

// Shi-Tomasi params
static constexpr int GFTT_BLOCK_SIZE = 5;
static constexpr double GFTT_QUALITY = 0.01;
static constexpr double GFTT_MIN_DIST = 15.0; // minDistance em px (efeito do "radius" antigo)

constexpr double a = 6378137.0;
constexpr double f = 1.0 / 298.257223563;
constexpr double b = a * (1 - f);
constexpr double e2 = 1 - (b * b) / (a * a);

// =========================================================
//  Structs
//    - ENU: East-North-Up coordinates
//    - Points: IDs and 2D keypoints and descriptors
//    - Keyframe: Keyframe data (ID, timestamp, pose, points, image)
//    - GpsSample: GPS data (timestamp, latitude, longitude, altitude)
// =========================================================

struct ENU
{
  double x, y, z;
};

struct Points
{
  std::vector<int> ids;
  std::vector<cv::KeyPoint> kp;
  cv::Mat desc;
};

struct Keyframe
{
  int frameID = -1;
  ros::Time timestamp;
  cv::Mat R, t;
  Points points;
  cv::Mat greyImg;
};

struct GpsSample
{
  ros::Time t;
  double lat;
  double lon;
  double alt;
};

// =========================================================
// Referential Coordinate Conversion Functions
// =========================================================
void geodeticToECEF(double lat, double lon, double alt, double &X, double &Y, double &Z)
{
  double sinLat = sin(lat), cosLat = cos(lat);
  double sinLon = sin(lon), cosLon = cos(lon);

  double N = a / sqrt(1 - e2 * sinLat * sinLat);

  X = (N + alt) * cosLat * cosLon;
  Y = (N + alt) * cosLat * sinLon;
  Z = (b * b / (a * a) * N + alt) * sinLat;
}

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
  enu.x = -sinLon * dx + cosLon * dy;
  enu.y = -cosLon * sinLat * dx - sinLon * sinLat * dy + cosLat * dz;
  enu.z = cosLon * cosLat * dx + sinLon * cosLat * dy + sinLat * dz;

  return enu;
}

Eigen::Quaterniond convertNEDtoENU(const Eigen::Quaterniond &q_ned)
{
  Eigen::Matrix3d R_ned_to_enu;
  R_ned_to_enu << 0, 1, 0,
      1, 0, 0,
      0, 0, -1;

  Eigen::Quaterniond q_tf(R_ned_to_enu);

  // Passive frame transformation: q_enu = R * q_ned * R⁻¹
  return q_tf * q_ned * q_tf.inverse();
}

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

    // Initializing ROS Publishers
    debugPub = nh.advertise<sensor_msgs::Image>("vio/debug_image", 1);

    // Opening ROSBAG
    ROS_INFO("Opening Bag...");
    try
    {
      bag.open(BAG_PATH, rosbag::bagmode::Read);
      ROS_INFO("Opened bag: %s", BAG_PATH);
    }
    catch (rosbag::BagException &e)
    {
      ROS_ERROR("Failed to open bag: %s. Error: %s", BAG_PATH, e.what());
      return;
    }

    // Loading ROSBAG messages
    ROS_INFO("Loading messages from ROSBAG...");

    // Camera Messages
    rosbag::View cam_view(bag, rosbag::TopicQuery(CAMERA_TOPIC));

    // Quaternion messages (converted referential)
    rosbag::View quat_view(bag, rosbag::TopicQuery(QUATERNION_TOPIC));
    for (const rosbag::MessageInstance &m : quat_view)
    {
      auto msg = m.instantiate<geometry_msgs::QuaternionStamped>();
      if (!msg)
        continue;

      Eigen::Quaterniond q_ned(
          msg->quaternion.w,
          msg->quaternion.x,
          msg->quaternion.y,
          msg->quaternion.z);

      // Convert from NED to ENU
      Eigen::Quaterniond q_enu = convertNEDtoENU(q_ned);

      // Store converted quaternion into quatMsgs
      geometry_msgs::QuaternionStampedPtr newMsg(new geometry_msgs::QuaternionStamped(*msg));
      newMsg->quaternion.w = q_enu.w();
      newMsg->quaternion.x = q_enu.x();
      newMsg->quaternion.y = q_enu.y();
      newMsg->quaternion.z = q_enu.z();

      quatMsgs.push_back(newMsg);
    }
    ROS_INFO("Loaded and converted %zu quaternion messages (NED -> ENU)", quatMsgs.size());

    // Velocity messages (converted referential)
    rosbag::View vel_view(bag, rosbag::TopicQuery(VELOCITY_TOPIC));
    for (const rosbag::MessageInstance &m : vel_view)
    {
      auto msg = m.instantiate<geometry_msgs::Vector3Stamped>();
      if (!msg)
        continue;

      // Convert velocity from NED to ENU
      geometry_msgs::Vector3StampedPtr newMsg(new geometry_msgs::Vector3Stamped(*msg));

      const auto &v_ned = msg->vector;

      newMsg->vector.x = v_ned.y;  // ENU X = East  = NED Y
      newMsg->vector.y = v_ned.x;  // ENU Y = North = NED X
      newMsg->vector.z = -v_ned.z; // ENU Z = Up    = -NED Z

      velMsgs.push_back(newMsg);
    }
    ROS_INFO("Loaded and converted %zu velocity messages (NED -> ENU)", velMsgs.size());

    // GPS messages
    rosbag::View gps_view(bag, rosbag::TopicQuery(GPS_TOPIC));
    for (const rosbag::MessageInstance &m : gps_view)
    {
      auto msg = m.instantiate<sensor_msgs::NavSatFix>();
      if (!msg)
        continue;

      GpsSample s;
      s.t = m.getTime();
      s.lat = msg->latitude;
      s.lon = msg->longitude;
      s.alt = msg->altitude;
      gpsSamples.push_back(s);
    }
    ROS_INFO("Loaded %zu GPS messages", gpsSamples.size());

    // ORB Detector Initialization
    orb = cv::ORB::create(750);

    // =========================================================
    // GTSAM Initialization     [WORK IN PROGRESS]
    // =========================================================

    // =========================================================
    // Main Loop
    // =========================================================
    // Helper variables
    bool firstFrame = true;
    int frameIdx = 0;
    ros::WallTime start = ros::WallTime::now();

    for (const rosbag::MessageInstance &m : cam_view)
    {
      // SIGINT Signal Processing
      if (g_requestShutdown)
      {
        ROS_WARN("Shutdown requested. Breaking processing loop.");
        break;
      }

      // Loading Image
      auto imgMsg = m.instantiate<sensor_msgs::CompressedImage>();
      if (!imgMsg)
      {
        frameIdx++;
        continue;
      }

      // Undistorting Image
      undistortImage(imgMsg);
      if (currUndistortedGrey.empty())
      {
        ROS_ERROR("[Frame %d] Undistorting failed!!", frameIdx);
        frameIdx++;
        continue;
      }

      // First Frame Processing
      if (firstFrame)
      {
        featureDetection(currUndistortedGrey);
        prevPoints = currPoints;
        prevUndistortedGrey = currUndistortedGrey.clone();
        firstFrame = false;
        frameIdx++;
        continue;
      }

      // CASE -> Not enough prevPoints: Reset Tracking
      if (prevPoints.kp.size() < 25 || prevUndistortedGrey.empty())
      {
        ROS_WARN("[Frame %d] No previous points to track. Detecting New Features", frameIdx);
        featureDetection(currUndistortedGrey);
        prevPoints = currPoints;
        prevUndistortedGrey = currUndistortedGrey.clone();
        frameIdx++;
        continue;
      }

      // =========================================================
      // Feature Tracking - KLT over Shi-Tomasi points
      // =========================================================
      trackKLT(prevUndistortedGrey, currUndistortedGrey);

      // If too few, replenish using Shi-Tomasi with mask
      if ((int)currPoints.kp.size() < KF_FEATURE_THRESHOLD)
      {
        ROS_WARN("[Frame %d] Low tracked features: %zu. Replenishing...", frameIdx, currPoints.kp.size());
        replenishFeatures(currUndistortedGrey, MAX_FEATURES);
      }

      // Publish debug image with matches (prev | curr + lines)
      publishDebugMatches(prevUndistortedGrey, currUndistortedGrey, m.getTime());

      prevUndistortedGrey = currUndistortedGrey.clone();
      prevPoints = currPoints;
      frameIdx++;
    }

    ROS_INFO("VIO Processing Complete.");
    ros::WallTime finish = ros::WallTime::now();
    ROS_INFO("Total Processing Time: %.2f seconds", (finish - start).toSec());
  }

private:
  // =========================================================
  // Variables
  // =========================================================
  // ROS Related Variables
  ros::NodeHandle nh;
  ros::Publisher debugPub;
  rosbag::Bag bag;

  // Message Storage
  std::vector<geometry_msgs::QuaternionStampedConstPtr> quatMsgs;
  std::vector<geometry_msgs::Vector3StampedConstPtr> velMsgs;
  std::vector<GpsSample> gpsSamples;

  // Camera Parameters
  const cv::Mat K = (cv::Mat_<double>(3, 3) << 1372.76165, 0.0, 960.45289,
                     0.0, 1372.14817, 515.00383,
                     0.0, 0.0, 1.0);
  const cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << 0.048300, -0.044905, -0.003731, -0.001349, 0);

  // ORB Feature Detector
  cv::Ptr<cv::ORB> orb;

  // Images
  cv::Mat currUndistortedGrey, currUndistortedRGB;
  cv::Mat prevUndistortedGrey;

  // Feature related
  Points currPoints, prevPoints;
  int nextFeatureID = 0;

  // Debug: matched point pairs (only accepted tracks)
  std::vector<cv::Point2f> dbgPrevPts;
  std::vector<cv::Point2f> dbgCurrPts;

  // =========================================================
  // Helpers
  // =========================================================
  static inline bool insideBorder(const cv::Point2f &p, int w, int h, int margin)
  {
    return (p.x >= margin && p.y >= margin && p.x < (w - margin) && p.y < (h - margin));
  }

  static inline std::vector<cv::Point2f> keypointsToPoints(const std::vector<cv::KeyPoint> &kps)
  {
    std::vector<cv::Point2f> pts;
    pts.reserve(kps.size());
    for (const auto &k : kps)
      pts.push_back(k.pt);
    return pts;
  }

  static inline std::vector<cv::KeyPoint> pointsToKeypoints(const std::vector<cv::Point2f> &pts)
  {
    std::vector<cv::KeyPoint> kps;
    kps.reserve(pts.size());
    for (const auto &p : pts)
      kps.emplace_back(p, 1.f);
    return kps;
  }

  // =========================================================
  // Methods
  // =========================================================
  void undistortImage(sensor_msgs::CompressedImageConstPtr msg)
  {
    cv::Mat raw = cv::imdecode(msg->data, cv::IMREAD_COLOR);
    if (raw.empty())
    {
      currUndistortedGrey.release();
      currUndistortedRGB.release();
      return;
    }

    cv::undistort(raw, currUndistortedRGB, K, distCoeffs);
    cv::cvtColor(currUndistortedRGB, currUndistortedGrey, cv::COLOR_BGR2GRAY);
  }

  void featureDetection(const cv::Mat &img)
  {
    currPoints.kp.clear();
    currPoints.ids.clear();
    currPoints.desc = cv::Mat(); // not used

    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(
        img,
        corners,
        MAX_FEATURES,
        GFTT_QUALITY,
        GFTT_MIN_DIST,
        cv::noArray(),
        GFTT_BLOCK_SIZE,
        false,
        0.04);

    // Filter borders + convert
    for (const auto &p : corners)
    {
      if (!insideBorder(p, img.cols, img.rows, BORDER_MARGIN))
        continue;
      currPoints.kp.emplace_back(p, 1.f);
      currPoints.ids.push_back(nextFeatureID++);
      if ((int)currPoints.kp.size() >= MAX_FEATURES)
        break;
    }
  }

  // ===== Shi-Tomasi detection without IDs (kept for parity) =====
  void featureDetectionWithoutID(const cv::Mat &img)
  {
    currPoints.kp.clear();
    currPoints.ids.clear();
    currPoints.desc = cv::Mat(); // not used

    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(
        img,
        corners,
        MAX_FEATURES,
        GFTT_QUALITY,
        GFTT_MIN_DIST,
        cv::noArray(),
        GFTT_BLOCK_SIZE,
        false,
        0.04);

    for (const auto &p : corners)
    {
      if (!insideBorder(p, img.cols, img.rows, BORDER_MARGIN))
        continue;
      currPoints.kp.emplace_back(p, 1.f);
      if ((int)currPoints.kp.size() >= MAX_FEATURES)
        break;
    }
  }

  // ===== KLT tracking with forward-backward check =====
  void trackKLT(const cv::Mat &prevImg, const cv::Mat &currImg)
  {
    currPoints.kp.clear();
    currPoints.ids.clear();
    currPoints.desc = cv::Mat();

    dbgPrevPts.clear();
    dbgCurrPts.clear();

    std::vector<cv::Point2f> prevPts = keypointsToPoints(prevPoints.kp);
    std::vector<cv::Point2f> currPts;
    std::vector<uchar> status;
    std::vector<float> err;

    cv::Size winSize(31, 31);
    cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, KLT_ITERS, KLT_EPS);

    cv::calcOpticalFlowPyrLK(
        prevImg,
        currImg,
        prevPts,
        currPts,
        status,
        err,
        winSize,
        KLT_MAX_LEVEL,
        criteria,
        0,
        KLT_MIN_EIG);

    // Forward-backward check
    std::vector<cv::Point2f> backPts;
    std::vector<uchar> statusBack;
    std::vector<float> errBack;

    cv::calcOpticalFlowPyrLK(
        currImg,
        prevImg,
        currPts,
        backPts,
        statusBack,
        errBack,
        winSize,
        KLT_MAX_LEVEL,
        criteria,
        0,
        KLT_MIN_EIG);

    // Keep good tracks and keep same IDs
    const int w = currImg.cols;
    const int h = currImg.rows;

    for (size_t i = 0; i < prevPts.size(); i++)
    {
      if (!status[i] || !statusBack[i])
        continue;

      const cv::Point2f &p0 = prevPts[i];
      const cv::Point2f &p1 = currPts[i];
      const cv::Point2f &pb = backPts[i];

      if (!insideBorder(p1, w, h, BORDER_MARGIN))
        continue;

      float fb = cv::norm(p0 - pb);
      if (fb > FB_THRESH_PX)
        continue;

      currPoints.kp.emplace_back(p1, 1.f);
      currPoints.ids.push_back(prevPoints.ids[i]);
      
      dbgPrevPts.push_back(p0);
      dbgCurrPts.push_back(p1);
    }

    ROS_INFO("[KLT] Tracked features: %zu / %zu",
             currPoints.kp.size(),
             prevPoints.kp.size());
  }

  // ===== Replenish features (adds new IDs) using mask around existing points =====
  void replenishFeatures(const cv::Mat &img, int targetCount)
  {
    int need = targetCount - (int)currPoints.kp.size();
    if (need <= 0)
      return;

    cv::Mat mask(img.size(), CV_8UC1, cv::Scalar(255));

    // mask existing
    const int radius = (int)std::round(GFTT_MIN_DIST);
    for (const auto &k : currPoints.kp)
    {
      cv::circle(mask, k.pt, radius, 0, -1);
    }

    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(
        img,
        corners,
        need,
        GFTT_QUALITY,
        GFTT_MIN_DIST,
        mask,
        GFTT_BLOCK_SIZE,
        false,
        0.04);

    for (const auto &p : corners)
    {
      if (!insideBorder(p, img.cols, img.rows, BORDER_MARGIN))
        continue;
      currPoints.kp.emplace_back(p, 1.f);
      currPoints.ids.push_back(nextFeatureID++);
      if ((int)currPoints.kp.size() >= targetCount)
        break;
    }
  }

  void publishDebugMatches(const cv::Mat &prevImgGray, const cv::Mat &currImgGray, const ros::Time &stamp)
  {
    if (prevImgGray.empty() || currImgGray.empty())
      return;

    cv::Mat prevBGR, currBGR;
    cv::cvtColor(prevImgGray, prevBGR, cv::COLOR_GRAY2BGR);
    cv::cvtColor(currImgGray, currBGR, cv::COLOR_GRAY2BGR);

    cv::Mat canvas;
    cv::hconcat(prevBGR, currBGR, canvas);

    const int xOff = prevBGR.cols;

    // Draw all accepted matches
    for (size_t i = 0; i < dbgPrevPts.size(); i++)
    {
      cv::Point2f p0 = dbgPrevPts[i];
      cv::Point2f p1 = dbgCurrPts[i];
      cv::Point2f p1s(p1.x + xOff, p1.y);

      cv::circle(canvas, p0, 2, cv::Scalar(0, 255, 0), -1);
      cv::circle(canvas, p1s, 2, cv::Scalar(0, 255, 0), -1);
      cv::line(canvas, p0, p1s, cv::Scalar(0, 255, 0), 1);
    }

    // Optional: text overlay
    {
      std::ostringstream oss;
      oss << "matches: " << dbgPrevPts.size();
      cv::putText(canvas, oss.str(), cv::Point(20, 30),
                  cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    }

    sensor_msgs::ImagePtr out =
        cv_bridge::CvImage(std_msgs::Header(), "bgr8", canvas).toImageMsg();
    out->header.stamp = stamp;
    debugPub.publish(out);
  }
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "fuims_vio");
  vioManager manager;
  return 0;
}
