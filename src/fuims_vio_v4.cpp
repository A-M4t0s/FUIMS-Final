#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/NavSatFix.h>
#include <geometry_msgs/QuaternionStamped.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>

#include <cv_bridge/cv_bridge.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <Eigen/Dense>
#include <unordered_map>

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
#include <fstream>
#include <cstdlib>
#include <iomanip>

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
//  Feature Detection and VIO Parameters
// =========================================================
#define MAX_FEATURES 250
#define MIN_TRACKED_FEATURES 35
#define MAX_TRACKING_ERROR_PX 7.5f
#define MAX_TRACKING_AGE 5 // In frames
#define KF_PARALLAX_THRESHOLD 28
#define KF_FEATURE_THRESHOLD 75
#define GPS_PRIOR_INTERVAL 10 // In keyframes

// KLT params (ajusta se necessário)
static constexpr int KLT_MAX_LEVEL = 4;
static constexpr int KLT_ITERS = 30;
static constexpr double KLT_EPS = 0.01;
static constexpr double KLT_MIN_EIG = 1e-4;
static constexpr float FB_THRESH_PX = 1.0f;
static constexpr int BORDER_MARGIN = 10;

// GFTT (Shi-Tomasi) Parameters
static constexpr int GFTT_MAX_FEATURES = 500;
static constexpr double GFTT_QUALITY = 0.15;
static constexpr double GFTT_MIN_DIST = 26.0;
static constexpr int GFTT_BLOCK_SIZE = 3;

// =========================================================
//  WGS84 Constants
// =========================================================
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
  std::vector<int> ids;         // Unique feature IDs
  std::vector<cv::Point2f> pts; // 2D keypoints
  // std::vector<double> quality;  // Quality scores
  std::vector<bool> isTracked; // Tracking status
  std::vector<int> age;        // Feature Age in frames
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
    matchingPub = nh.advertise<sensor_msgs::Image>("vio/feature_matches", 1);
    featurePub = nh.advertise<sensor_msgs::Image>("vio/feature_ages", 1);
    posePub = nh.advertise<geometry_msgs::PoseStamped>("vio/pose", 1);
    pathPub = nh.advertise<nav_msgs::Path>("vio/path", 1);
    gtPathPub = nh.advertise<nav_msgs::Path>("vio/ground_truth", 1, true); // latched
    pathMsg.header.frame_id = "map";

    // Opening ROSBAG
    ROS_INFO_STREAM(CYAN << "Opening Bag..." << RESET);
    try
    {
      bag.open(BAG_PATH, rosbag::bagmode::Read);
      ROS_INFO_STREAM(GREEN << "Opened bag: " << BAG_PATH << RESET);
    }
    catch (rosbag::BagException &e)
    {
      ROS_ERROR_STREAM(RED << "Failed to open bag: " << BAG_PATH << ". Error: " << e.what() << RESET);
      return;
    }

    // Loading ROSBAG messages
    ROS_INFO_STREAM(CYAN << "Loading messages from ROSBAG..." << RESET);

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
    ROS_INFO_STREAM(GREEN << "Loaded and converted " << quatMsgs.size() << " quaternion messages (NED → ENU)" << RESET);

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
    ROS_INFO_STREAM(GREEN << "Loaded and converted " << velMsgs.size() << " velocity messages (NED → ENU)" << RESET);

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
    ROS_INFO_STREAM(GREEN << "Loaded " << gpsSamples.size() << " GPS messages" << RESET);

    // Build Ground Truth Path
    nav_msgs::Path gtPath;
    gtPath.header.frame_id = "map";

    for (const auto &gps : gpsSamples)
    {
      double X, Y, Z;
      geodeticToECEF(gps.lat, gps.lon, gps.alt, X, Y, Z);
      ENU enu = ecefToENU(X, Y, Z,
                          gpsSamples.front().lat,
                          gpsSamples.front().lon,
                          gpsSamples.front().alt);

      geometry_msgs::PoseStamped pose;
      pose.header.stamp = gps.t;
      pose.header.frame_id = "map";
      pose.pose.position.x = enu.x;
      pose.pose.position.y = enu.y;
      pose.pose.position.z = enu.z;

      // Optionally add identity orientation
      pose.pose.orientation.w = 1.0;

      gtPath.poses.push_back(pose);
    }
    gtPathPub.publish(gtPath);
    ROS_INFO_STREAM(GREEN << "Published ground truth path with " << gtPath.poses.size() << " poses" << RESET);

    // =========================================================
    // GTSAM Initialization
    // =========================================================
    // ---------------- ISAM2 Setup ----------------
    gtsam::ISAM2Params params;
    params.relinearizeThreshold = 0.1;
    params.relinearizeSkip = 1;
    isam = gtsam::ISAM2(params);

    // ---------------- Prior ----------------
    gtsam::Pose3 prior;
    auto priorNoise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1).finished());

    graph.add(gtsam::PriorFactor<gtsam::Pose3>(gtsam::Symbol('x', 0), prior, priorNoise));
    values.insert(gtsam::Symbol('x', 0), prior);

    isam.update(graph, values);
    graph.resize(0);
    values.clear();

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
        ROS_WARN_STREAM(YELLOW << "Shutdown requested. Breaking processing loop." << RESET);
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
        ROS_ERROR_STREAM(RED << "[Frame " << frameIdx << "] Undistorting failed!" << RESET);
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
      if (prevPoints.pts.size() < 25 || prevUndistortedGrey.empty())
      {
        ROS_WARN_STREAM(YELLOW << "[Frame " << frameIdx << "] No previous points to track. Detecting new features." << RESET);
        featureDetection(currUndistortedGrey);
        prevPoints = currPoints;
        prevUndistortedGrey = currUndistortedGrey.clone();
        frameIdx++;
        continue;
      }

      // KLT Tracking
      std::vector<cv::Point2f> trackedPoints;
      std::vector<uchar> status;
      std::vector<float> error;
      cv::calcOpticalFlowPyrLK(
          prevUndistortedGrey,
          currUndistortedGrey,
          prevPoints.pts,
          trackedPoints,
          status,
          error,
          cv::Size(21, 21),
          2);

      // KLT Results Processing (Filtering + Storing)
      /*
       * - Ensure 'times tracked' feature consistency
       * - Store successfully tracked points into currPoints with same IDs  (must be seen at least 5 times in previous frames)
       * - Use error thresholding (reprojection error) to filter points
       * - Only points inside image borders are considered valid
       */
      Points validPoints;
      for (size_t i = 0; i < status.size(); i++)
      {
        // Check if tracking was successful
        if (!status[i])
          continue;

        // Check tracking error
        if (error[i] > MAX_TRACKING_ERROR_PX)
          continue;

        // Check if points are inside image borders
        if (!inImage(trackedPoints[i], currUndistortedGrey) || !inImage(prevPoints.pts[i], prevUndistortedGrey))
          continue;

        // Store tracked point
        validPoints.pts.push_back(trackedPoints[i]);
        validPoints.ids.push_back(prevPoints.ids[i]);
        validPoints.isTracked.push_back(true);
        validPoints.age.push_back(prevPoints.age[i] + 1);
      }
      currPoints = validPoints;
      // ROS_INFO_STREAM(CYAN << "[Frame " << frameIdx << "] KLT Tracked Features: "
      //                      << currPoints.pts.size() << " / " << prevPoints.pts.size() << RESET);

      // Feature Replenish to MAX_FEATURES
      if (currPoints.pts.size() < MAX_FEATURES)
      {
        int beforeReplenish = static_cast<int>(currPoints.pts.size());
        replenishFeatures(currUndistortedGrey);
        int afterReplenish = static_cast<int>(currPoints.pts.size());
        // ROS_WARN_STREAM(YELLOW << "[Frame " << frameIdx << "] Replenished Features: "
        //                        << beforeReplenish << " -> " << afterReplenish << RESET);
      }

      // Keyframe Decision (Parallax only)
      bool isKeyframe = false;
      bool hasAlignment = false;
      Points trackedOnly;
      static gtsam::Pose3 T_align;

      // Consider only features that are older than MAX_TRACKING_AGE
      for (size_t i = 0; i < currPoints.pts.size(); ++i)
      {
        if (currPoints.age[i] >= MAX_TRACKING_AGE)
        {
          trackedOnly.pts.push_back(currPoints.pts[i]);
          trackedOnly.ids.push_back(currPoints.ids[i]);
        }
      }

      if (!hasKF && trackedOnly.pts.size() >= MIN_TRACKED_FEATURES) // First Keyframe
      {
        isKeyframe = true;

        if (!hasAlignment && !gpsSamples.empty())
        {
          // Get first GPS pose in ENU
          double X, Y, Z;
          geodeticToECEF(gpsSamples.front().lat, gpsSamples.front().lon, gpsSamples.front().alt, X, Y, Z);
          ENU enu = ecefToENU(X, Y, Z,
                              gpsSamples.front().lat,
                              gpsSamples.front().lon,
                              gpsSamples.front().alt);

          gtsam::Point3 gps_pos(enu.x, enu.y, enu.z);
          gtsam::Pose3 gps_pose(gtsam::Rot3(), gps_pos); // assume flat orientation

          // First VIO pose is at identity
          gtsam::Pose3 vio_pose = gtsam::Pose3(); // or use: isam.calculateEstimate(...)

          // Compute transform from VIO frame to GPS frame
          T_align = gps_pose.compose(vio_pose.inverse());

          hasAlignment = true;
        }
      }
      else if (hasKF && trackedOnly.pts.size() >= MIN_TRACKED_FEATURES) // Subsequent Keyframes
      {
        double kfParallax = computeKFParallax(lastKF, trackedOnly);
        if (kfParallax > KF_PARALLAX_THRESHOLD)
          isKeyframe = true;
        if (trackedOnly.pts.size() < KF_FEATURE_THRESHOLD)
          isKeyframe = true;
      }

      // Keyframe Processing
      if (isKeyframe)
      {
        ROS_INFO_STREAM(GREEN << "[Frame " << frameIdx << "] New Keyframe created with "
                              << trackedOnly.pts.size() << " features." << RESET);
        const ros::Time kfTime = m.getTime();

        // First Keyframe Initialization
        if (!hasKF)
        {
          lastKF.frameID = frameIdx;
          lastKF.greyImg = currUndistortedGrey.clone();
          lastKF.points = trackedOnly;
          lastKF.timestamp = kfTime;
          hasKF = true;
          kfIndex = 0;

          kfTimes.clear();
          kfTimes.push_back(kfTime.toSec());
        }
        else // Subsequent Keyframes
        {
          const int prevKF = kfIndex;
          const int currKF = kfIndex + 1;
          bool hasConstraint = false; // Track if anything was added to the graph

          // Get previous pose estimate
          gtsam::Pose3 prevPose = isam.calculateEstimate<gtsam::Pose3>(gtsam::Symbol('x', prevKF));
          gtsam::Pose3 currGuess = prevPose; // default guess if no info

          // Step 1: Visual Odometry (BetweenFactor)
          std::vector<cv::Point2f> prevKFPoints, currKFPoints;
          buildCorrespondencesById(lastKF.points, trackedOnly, prevKFPoints, currKFPoints);

          if (prevKFPoints.size() >= 16)
          {
            gtsam::Pose3 relativePose = relativeMovementEstimation(prevKFPoints, currKFPoints, lastKF.timestamp, kfTime);

            currGuess = prevPose.compose(relativePose);

            auto voNoise = gtsam::noiseModel::Diagonal::Sigmas(
                (gtsam::Vector(6) << 0.1, 0.1, 0.1, 0.4, 0.4, 0.4).finished());

            graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
                gtsam::Symbol('x', prevKF),
                gtsam::Symbol('x', currKF),
                relativePose,
                voNoise));

            hasConstraint = true;
            OK("[Keyframe " << currKF << "] VO between factor added.");
          }
          else
          {
            ROS_WARN_STREAM(YELLOW << "[Keyframe " << currKF << "] Not enough correspondences for VO Between Factor ("
                                   << prevKFPoints.size() << " found)." << RESET);
            currGuess = prevPose; // fallback
          }

          // Step 2: Quaternion Prior (rotation-only)
          auto qmsg = findNearestQuat(kfTime);
          if (qmsg)
          {
            gtsam::Rot3 Rq = gtsam::Rot3::Quaternion(
                qmsg->quaternion.w,
                qmsg->quaternion.x,
                qmsg->quaternion.y,
                qmsg->quaternion.z);

            auto rotNoise = gtsam::noiseModel::Diagonal::Sigmas(
                (gtsam::Vector(3) << 0.05, 0.05, 0.05).finished());

            graph.add(gtsam::PoseRotationPrior<gtsam::Pose3>(
                gtsam::Symbol('x', currKF),
                Rq,
                rotNoise));

            hasConstraint = true;
            OK("[Keyframe " << currKF << "] Orientation prior added from quaternion.");

            // If VO failed, use orientation as fallback guess
            if (prevKFPoints.size() < 16)
            {
              gtsam::Point3 t = prevPose.translation();
              currGuess = gtsam::Pose3(Rq, t);
            }
          }

          // Step 3: Velocity integration)
          Eigen::Vector3d dp_world;
          {
            dp_world = integrateVelocity(lastKF.timestamp, kfTime);

            // Transform delta to local frame of prev pose
            gtsam::Pose3 prevPose = isam.calculateEstimate<gtsam::Pose3>(gtsam::Symbol('x', prevKF));
            Eigen::Vector3d dp_local = prevPose.rotation().unrotate(dp_world);

            gtsam::Pose3 velDelta(gtsam::Rot3(), dp_local);

            auto velNoise = gtsam::noiseModel::Diagonal::Sigmas(
                (gtsam::Vector(6) << 1e6, 1e6, 1e6, 0.8, 0.8, 0.8).finished());

            graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
                gtsam::Symbol('x', prevKF),
                gtsam::Symbol('x', currKF),
                velDelta,
                velNoise));

            hasConstraint = true;

            OK("[Keyframe " << currKF << "] Velocity-based between factor added.");
          }

          // Step 4: GPS Prior (absolute translation only)
          if (kfIndex % GPS_PRIOR_INTERVAL == 0)
          {
            double t_kf_sec = kfTime.toSec();
            double best_dt = 1e9;
            size_t best_idx = 0;

            for (size_t i = 0; i < gpsSamples.size(); i++)
            {
              double dt = std::abs(gpsSamples[i].t.toSec() - t_kf_sec);
              if (dt < best_dt)
              {
                best_dt = dt;
                best_idx = i;
              }
            }

            const GpsSample &gps = gpsSamples[best_idx];

            double X, Y, Z;
            geodeticToECEF(gps.lat, gps.lon, gps.alt, X, Y, Z);
            ENU enu = ecefToENU(X, Y, Z,
                                gpsSamples.front().lat,
                                gpsSamples.front().lon,
                                gpsSamples.front().alt);

            gtsam::Point3 gps_point(enu.x, enu.y, enu.z);

            auto posNoise = gtsam::noiseModel::Diagonal::Sigmas(
                (gtsam::Vector(3) << 0.1, 0.1, 0.1).finished());

            // Add GPS prior (on translation only)
            graph.add(gtsam::PoseTranslationPrior<gtsam::Pose3>(
                gtsam::Symbol('x', currKF),
                gps_point,
                posNoise));

            ROS_INFO_STREAM(GREEN << "[Keyframe " << currKF << "] GPS translation prior added." << RESET);
          }

          bool hasTranslation =
              (prevKFPoints.size() >= 16) ||
              (dp_world.norm() > 0.05) ||
              (kfIndex % GPS_PRIOR_INTERVAL == 0);

          if (!hasTranslation)
          {
            auto weakPosNoise = gtsam::noiseModel::Diagonal::Sigmas(
                (gtsam::Vector(3) << 5.0, 5.0, 5.0).finished());

            graph.add(gtsam::PoseTranslationPrior<gtsam::Pose3>(
                gtsam::Symbol('x', currKF),
                prevPose.translation(),
                weakPosNoise));

            WARN("[Keyframe " << currKF << "] Weak translation prior added (safety)");
          }

          // Step 5: Insert initial guess and update ISAM
          if (hasConstraint)
          {
            if (isam.getLinearizationPoint().exists(gtsam::Symbol('x', currKF)))
            {
              WARN("x" << currKF << " already in ISAM. Skipping.");
              continue;
            }

            values.insert(gtsam::Symbol('x', currKF), currGuess);
            isam.update(graph, values);

            graph.resize(0);
            values.clear();
          }
          else
          {
            WARN("[Keyframe " << currKF << "] No constraints added to graph. Skipping ISAM update.");
            continue;
          }

          // Publish estimated pose
          gtsam::Pose3 estimatedPose = isam.calculateEstimate<gtsam::Pose3>(gtsam::Symbol('x', currKF));
          if (hasAlignment)
            estimatedPose = T_align.compose(estimatedPose);

          geometry_msgs::PoseStamped poseMsg;
          poseMsg.header.stamp = kfTime;
          poseMsg.header.frame_id = "map";
          poseMsg.pose.position.x = estimatedPose.x();
          poseMsg.pose.position.y = estimatedPose.y();
          poseMsg.pose.position.z = estimatedPose.z();

          gtsam::Rot3 R = estimatedPose.rotation();
          gtsam::Quaternion q = R.toQuaternion();

          poseMsg.pose.orientation.w = q.w();
          poseMsg.pose.orientation.x = q.x();
          poseMsg.pose.orientation.y = q.y();
          poseMsg.pose.orientation.z = q.z();

          // Append and publish path
          pathMsg.header.stamp = kfTime;
          pathMsg.poses.push_back(poseMsg);
          pathPub.publish(pathMsg);
          posePub.publish(poseMsg);

          // Update KF state
          kfIndex = currKF;
          kfTimes.push_back(kfTime.toSec());
          lastKF.frameID = frameIdx;
          lastKF.greyImg = currUndistortedGrey.clone();
          lastKF.points = trackedOnly;
          lastKF.timestamp = kfTime;
        }
      }

      // Debug: Publish matched points image
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
      cv::Mat matchVis = drawFeatureMatches(prevUndistortedGrey, currUndistortedGrey,
                                            matchedPrevPts, matchedCurrPts);
      sensor_msgs::ImagePtr matchMsg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", matchVis).toImageMsg();
      matchMsg->header.stamp = imgMsg->header.stamp;
      matchingPub.publish(matchMsg);

      // Debug: Publish feature state image
      cv::Mat ageVis = drawFeatureAges(currUndistortedGrey, currPoints);
      sensor_msgs::ImagePtr ageMsg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", ageVis).toImageMsg();
      ageMsg->header.stamp = imgMsg->header.stamp;
      featurePub.publish(ageMsg);

      // Update previous frame data
      prevUndistortedGrey = currUndistortedGrey.clone();
      prevPoints = currPoints;
      frameIdx++;
    }
  }

private:
  // =========================================================
  // Variables
  // =========================================================
  // ROS Related Variables
  ros::NodeHandle nh;
  ros::Publisher matchingPub, featurePub, posePub, pathPub, gtPathPub;
  nav_msgs::Path pathMsg;
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

  // Images
  cv::Mat currUndistortedGrey, currUndistortedRGB;
  cv::Mat prevUndistortedGrey;

  // Keyframe related
  Keyframe lastKF;
  int kfIndex = 0;
  bool hasKF = false;
  std::vector<double> kfTimes;

  // Feature related
  Points currPoints, prevPoints;
  int nextFeatureID = 0;

  // GTSAM related
  gtsam::ISAM2 isam;
  gtsam::NonlinearFactorGraph graph;
  gtsam::Values values;

  // =========================================================
  // Helpers
  // =========================================================

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
    // Clear current points
    currPoints.ids.clear();
    currPoints.pts.clear();
    // currPoints.quality.clear();
    currPoints.isTracked.clear();
    currPoints.age.clear();

    // GFTT Detector
    std::vector<cv::Point2f> detectedPts;
    cv::Mat qualityValues;
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

    // Fill currPoints structure
    for (const auto &pt : detectedPts)
    {
      currPoints.pts.push_back(pt);
      currPoints.ids.push_back(nextFeatureID++);
      // currPoints.quality.push_back(1.0); // Initial quality
      currPoints.isTracked.push_back(false); // Initially not tracked
      currPoints.age.push_back(1);           // Initial age
    }
  }

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

  double computeKFParallax(Keyframe &KF, Points &currPts)
  {
    // optional defensive checks
    if (KF.points.ids.size() != KF.points.pts.size())
    {
      ROS_WARN("KF points ids/pts size mismatch");
    }
    if (currPts.ids.size() != currPts.pts.size())
    {
      ROS_WARN("currPts ids/pts size mismatch");
    }

    // build fast lookup: id -> index in currPts.pts
    std::unordered_map<int, int> id_to_idx;
    id_to_idx.reserve(currPts.ids.size());
    for (size_t i = 0; i < currPts.ids.size(); ++i)
    {
      id_to_idx[currPts.ids[i]] = static_cast<int>(i);
    }

    double sum = 0.0;
    int cnt = 0;

    for (size_t i = 0; i < KF.points.ids.size(); ++i)
    {
      int idKF = KF.points.ids[i];
      auto it = id_to_idx.find(idKF);
      if (it == id_to_idx.end())
        continue;

      int j = it->second;

      double dx = currPts.pts[j].x - KF.points.pts[i].x;
      double dy = currPts.pts[j].y - KF.points.pts[i].y;

      sum += std::sqrt(dx * dx + dy * dy);
      cnt++;
    }

    return (cnt == 0) ? 0.0 : (sum / cnt);
  }

  void buildCorrespondencesById(const Points &ref, const Points &cur,
                                std::vector<cv::Point2f> &refPts,
                                std::vector<cv::Point2f> &curPts)
  {
    refPts.clear();
    curPts.clear();
    refPts.reserve(ref.ids.size());

    for (size_t i = 0; i < ref.ids.size(); i++)
    {
      int id = ref.ids[i];
      auto it = std::find(cur.ids.begin(), cur.ids.end(), id);
      if (it == cur.ids.end())
        continue;

      int j = std::distance(cur.ids.begin(), it);

      refPts.push_back(ref.pts[i]);
      curPts.push_back(cur.pts[j]);
    }
  }

  gtsam::Pose3 relativeMovementEstimation(std::vector<cv::Point2f> &prevValid,
                                          std::vector<cv::Point2f> &currValid,
                                          const ros::Time &t0,
                                          const ros::Time &t1)
  {
    if (prevValid.size() < 8 || currValid.size() < 8)
    {
      ROS_WARN("[relativeMovementEstimation] Not enough features");
      return gtsam::Pose3();
    }

    cv::Mat E = cv::findEssentialMat(prevValid, currValid, K, cv::RANSAC, 0.999, 1.0);
    if (E.empty())
    {
      ROS_WARN("[relativeMovementEstimation] Essential Matrix not valid");
      return gtsam::Pose3();
    }

    cv::Mat R, t;
    cv::recoverPose(E, prevValid, currValid, K, R, t);

    // Recover rotation
    Eigen::Matrix3d Rg_mat;
    Rg_mat << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);

    // Recover unscaled translation
    Eigen::Vector3d t_dir(t.at<double>(0), t.at<double>(1), t.at<double>(2));

    // Scale using integrated velocity (rotated)
    Eigen::Vector3d dp = integrateVelocity(t0, t1);
    double scale = dp.norm();

    if (scale < 1e-2)
    {
      ROS_WARN("Velocity integration too small for scaling. Skipping scale correction.");
      scale = 1.0; // fallback
    }

    Eigen::Vector3d t_scaled = t_dir.normalized() * scale;

    gtsam::Rot3 Rg(Rg_mat);
    gtsam::Point3 tg(t_scaled.x(), t_scaled.y(), t_scaled.z());

    return gtsam::Pose3(Rg, tg);
  }

  geometry_msgs::QuaternionStampedConstPtr findNearestQuat(const ros::Time &t)
  {
    geometry_msgs::QuaternionStampedConstPtr best = nullptr;
    double best_dt = 1e9;

    for (const auto &q : quatMsgs)
    {
      double dt = fabs((q->header.stamp - t).toSec());
      if (dt < best_dt)
      {
        best_dt = dt;
        best = q;
      }
    }

    if (best_dt > 0.05)
      return nullptr;

    return best;
  }

  Eigen::Quaterniond interpolateQuat(const ros::Time &t)
  {
    if (quatMsgs.size() < 2)
      return Eigen::Quaterniond::Identity();

    for (size_t i = 1; i < quatMsgs.size(); ++i)
    {
      const auto &q1 = quatMsgs[i - 1];
      const auto &q2 = quatMsgs[i];

      if (q1->header.stamp <= t && q2->header.stamp >= t)
      {
        double t1 = q1->header.stamp.toSec();
        double t2 = q2->header.stamp.toSec();
        double alpha = (t.toSec() - t1) / (t2 - t1);

        Eigen::Quaterniond q_start(q1->quaternion.w, q1->quaternion.x, q1->quaternion.y, q1->quaternion.z);
        Eigen::Quaterniond q_end(q2->quaternion.w, q2->quaternion.x, q2->quaternion.y, q2->quaternion.z);

        return q_start.slerp(alpha, q_end);
      }
    }

    // Fallback if outside range
    return Eigen::Quaterniond(quatMsgs.back()->quaternion.w,
                              quatMsgs.back()->quaternion.x,
                              quatMsgs.back()->quaternion.y,
                              quatMsgs.back()->quaternion.z);
  }

  Eigen::Vector3d integrateVelocity(const ros::Time &t0, const ros::Time &t1)
  {
    Eigen::Vector3d dp = Eigen::Vector3d::Zero();

    for (size_t i = 1; i < velMsgs.size(); i++)
    {
      ros::Time ta = velMsgs[i - 1]->header.stamp;
      ros::Time tb = velMsgs[i]->header.stamp;

      if (tb <= t0 || ta >= t1)
        continue;

      double dt = (tb - ta).toSec();
      if (dt <= 0.0)
        continue;

      Eigen::Vector3d v_body(
          velMsgs[i]->vector.x,
          velMsgs[i]->vector.y,
          velMsgs[i]->vector.z);

      Eigen::Quaterniond q_wb = interpolateQuat(velMsgs[i]->header.stamp);

      Eigen::Vector3d v_world = q_wb * v_body;
      dp += v_world * dt;
    }

    return dp;
  }

  bool inImage(const cv::Point2f &p, const cv::Mat &img)
  {
    return p.x >= 16 && p.y >= 16 && p.x < img.cols - 16 && p.y < img.rows - 16;
  }
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "fuims_vio");
  vioManager manager;
  return 0;
}