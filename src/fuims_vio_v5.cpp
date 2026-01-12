#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/NavSatFix.h>
#include <geometry_msgs/QuaternionStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Vector3Stamped.h>
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
#include <gtsam/nonlinear/ISAM2.h>
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

// =========================================================
//  ROSBAG Defines
// =========================================================
#define BAG_PATH "/home/tony/catkin_ws/bags/bags_fuims/agucadoura.bag"
#define CAMERA_TOPIC "/dji_m350/cameras/main/compressed"
#define QUATERNION_TOPIC "/dji_m350/quaternion"
#define VELOCITY_TOPIC "/dji_m350/velocity"
#define GPS_TOPIC "/dji_m350/gps"

// =========================================================
//  ANSI Colors
// =========================================================
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
#define MAX_TRACKING_ERROR_PX 7.5f
#define MAX_TRACKING_AGE 5
#define KF_PARALLAX_THRESHOLD 28
#define KF_FEATURE_THRESHOLD 100
#define GPS_PRIOR_INTERVAL 15

// =========================================================
//  Structs
// =========================================================
struct ENU { double x, y, z; };

struct Points
{
  std::vector<int> ids;
  std::vector<cv::Point2f> pts;
  std::vector<bool> isTracked;
  std::vector<int> age;
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
  double lat, lon, alt;
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
    signal(SIGINT, signalHandler);

    // ---------------- ROS Publishers ----------------
    matchingPub = nh.advertise<sensor_msgs::Image>("vio/feature_matches", 1);
    featurePub  = nh.advertise<sensor_msgs::Image>("vio/feature_ages", 1);
    posePub     = nh.advertise<geometry_msgs::PoseStamped>("vio/pose", 1);
    pathPub     = nh.advertise<nav_msgs::Path>("vio/path", 1);

    pathMsg.header.frame_id = "map";

    // ---------------- ISAM2 Setup ----------------
    gtsam::ISAM2Params params;
    params.relinearizeThreshold = 0.1;
    params.relinearizeSkip = 1;
    isam = gtsam::ISAM2(params);

    // ---------------- Prior ----------------
    gtsam::Pose3 prior;
    auto priorNoise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 0.1,0.1,0.1,0.1,0.1,0.1).finished());

    graph.add(gtsam::PriorFactor<gtsam::Pose3>(gtsam::Symbol('x',0), prior, priorNoise));
    values.insert(gtsam::Symbol('x',0), prior);

    isam.update(graph, values);
    graph.resize(0);
    values.clear();

    // =========================================================
    // Main Loop (ROSBAG Playback)
    // =========================================================
    rosbag::Bag bag;
    bag.open(BAG_PATH, rosbag::bagmode::Read);
    rosbag::View cam_view(bag, rosbag::TopicQuery(CAMERA_TOPIC));

    bool firstFrame = true;
    int frameIdx = 0;

    for (const rosbag::MessageInstance &m : cam_view)
    {
      if (g_requestShutdown) break;

      auto imgMsg = m.instantiate<sensor_msgs::CompressedImage>();
      if (!imgMsg) continue;

      undistortImage(imgMsg);
      if (currUndistortedGrey.empty()) continue;

      if (firstFrame)
      {
        featureDetection(currUndistortedGrey);
        prevPoints = currPoints;
        prevUndistortedGrey = currUndistortedGrey.clone();
        firstFrame = false;
        continue;
      }

      // ================= Keyframe Creation =================
      bool isKeyframe = !hasKF;

      if (isKeyframe)
      {
        ros::Time t_kf = m.getTime();
        const int prevK = kfIndex;
        const int curK  = kfIndex + 1;

        gtsam::Pose3 seed = isam.calculateEstimate().at<gtsam::Pose3>(gtsam::Symbol('x',prevK));
        values.insert(gtsam::Symbol('x',curK), seed);

        graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
            gtsam::Symbol('x',prevK),
            gtsam::Symbol('x',curK),
            gtsam::Pose3(),
            gtsam::noiseModel::Diagonal::Sigmas(
                (gtsam::Vector(6)<<0.1,0.1,0.1,0.1,0.1,0.1).finished()
            )));

        // ---------------- ISAM2 UPDATE ----------------
        isam.update(graph, values);
        graph.resize(0);
        values.clear();

        // ---------------- Publish Pose ----------------
        gtsam::Pose3 pose = isam.calculateEstimate().at<gtsam::Pose3>(gtsam::Symbol('x',curK));

        geometry_msgs::PoseStamped poseMsg;
        poseMsg.header.frame_id = "map";
        poseMsg.header.stamp = t_kf;
        poseMsg.pose.position.x = pose.x();
        poseMsg.pose.position.y = pose.y();
        poseMsg.pose.position.z = pose.z();

        auto q = pose.rotation().toQuaternion();
        poseMsg.pose.orientation.x = q.x();
        poseMsg.pose.orientation.y = q.y();
        poseMsg.pose.orientation.z = q.z();
        poseMsg.pose.orientation.w = q.w();

        posePub.publish(poseMsg);

        pathMsg.header.stamp = t_kf;
        pathMsg.poses.push_back(poseMsg);
        pathPub.publish(pathMsg);

        hasKF = true;
        kfIndex = curK;
      }

      prevUndistortedGrey = currUndistortedGrey.clone();
      prevPoints = currPoints;
      frameIdx++;
    }
  }

private:
  // =========================================================
  // Variables
  // =========================================================
  ros::NodeHandle nh;
  ros::Publisher matchingPub, featurePub, posePub, pathPub;
  nav_msgs::Path pathMsg;

  cv::Mat currUndistortedGrey, currUndistortedRGB, prevUndistortedGrey;
  Points currPoints, prevPoints;

  Keyframe lastKF;
  bool hasKF = false;
  int kfIndex = 0;

  gtsam::ISAM2 isam;
  gtsam::NonlinearFactorGraph graph;
  gtsam::Values values;

  // =========================================================
  // Methods (unchanged from your code)
  // =========================================================
  void undistortImage(sensor_msgs::CompressedImageConstPtr msg)
  {
    cv::Mat raw = cv::imdecode(msg->data, cv::IMREAD_COLOR);
    if (raw.empty()) return;
    cv::cvtColor(raw, currUndistortedGrey, cv::COLOR_BGR2GRAY);
  }

  void featureDetection(const cv::Mat &img)
  {
    currPoints.ids.clear();
    currPoints.pts.clear();
    currPoints.age.clear();
    currPoints.isTracked.clear();

    std::vector<cv::Point2f> pts;
    cv::goodFeaturesToTrack(img, pts, MAX_FEATURES, 0.1, 20.0);
    for (auto &p : pts)
    {
      currPoints.pts.push_back(p);
      currPoints.ids.push_back(currPoints.ids.size());
      currPoints.age.push_back(1);
      currPoints.isTracked.push_back(false);
    }
  }
};

// =========================================================
// Main
// =========================================================
int main(int argc, char **argv)
{
  ros::init(argc, argv, "fuims_vio");
  vioManager manager;
  return 0;
}
