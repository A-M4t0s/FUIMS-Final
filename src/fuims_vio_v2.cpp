#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/QuaternionStamped.h>
#include <geometry_msgs/Vector3Stamped.h>

#include <nav_msgs/Path.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <cv_bridge/cv_bridge.h>
#include <cmath>
#include <algorithm>
#include <numeric>

#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Point3.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/video.hpp>
#include <opencv2/viz.hpp>

#define BAG_PATH "/home/tony/catkin_ws/bags/bags_fuims/agucadoura.bag"
#define CAMERA_TOPIC "/dji_m350/cameras/main/compressed"
#define QUATERNION_TOPIC "/dji_m350/quaternion"
#define VELOCITY_TOPIC "/dji_m350/velocity"
#define GPS_TOPIC "/dji_m350/gps"

#define ORB_N_BEST 250
#define KF_PARALLAX_THRESHOLD 28
#define KF_FEATURE_THRESHOLD 100

constexpr double a = 6378137.0;
constexpr double f = 1.0 / 298.257223563;
constexpr double b = a * (1 - f);
constexpr double e2 = 1 - (b * b) / (a * a);

/*
 * ================================================================================================================
 * Data Structure Definitions
 * ================================================================================================================
 */
struct ENU
{
    double x, y, z;
};

struct Points
{
    std::vector<int> ids;
    std::vector<cv::Point2f> pts;
};

struct Keyframe
{
    int frameID;
    cv::Mat R, t;
    Points points;
    cv::Mat greyImg;
};

/*
 * ================================================================================================================
 * GPS <-> ENU Conversions
 * ================================================================================================================
 */
void geodeticToECEF(double lat, double lon, double alt, double &X, double &Y, double &Z)
{
    double sinLat = sin(lat), cosLat = cos(lat);
    double sinLon = sin(lon), cosLon = cos(lon);

    double N = a / sqrt(1 - e2 * sinLat * sinLat);

    X = (N + alt) * cosLat * cosLon;
    Y = (N + alt) * cosLat * sinLon;
    Z = (b * b / (a * a) * N + alt) * sinLat;
}

ENU ecefToENU(double X, double Y, double Z,
              double lat0, double lon0, double alt0)
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

/*
 * ================================================================================================================
 * Class: vioManager
 * ================================================================================================================
 */
class vioManager
{
public:
    vioManager()
    {
        // =========================================================
        // Starting Publishers
        // =========================================================
        debugPub = nh.advertise<sensor_msgs::Image>("vio/debug_image", 1);

        // =========================================================
        // Opening bag
        // =========================================================
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

        // =========================================================
        // Loading Messages from ROSBAG
        // =========================================================
        ROS_INFO("Loading messages from ROSBAG...");
        //  -> Camera Images
        rosbag::View cam_view(bag, rosbag::TopicQuery(CAMERA_TOPIC));

        // -> Quaternion Messages
        rosbag::View quat_view(bag, rosbag::TopicQuery(QUATERNION_TOPIC));
        for (const rosbag::MessageInstance &m : quat_view)
        {
            auto msg = m.instantiate<geometry_msgs::QuaternionStamped>();
            if (msg)
                quatMsgs.push_back(msg);
        }
        ROS_INFO("Loaded %zu quaternion messages", quatMsgs.size());

        // -> Velocity Messages
        rosbag::View vel_view(bag, rosbag::TopicQuery(VELOCITY_TOPIC));
        for (const rosbag::MessageInstance &m : vel_view)
        {
            auto msg = m.instantiate<geometry_msgs::Vector3Stamped>();
            if (msg)
                velMsgs.push_back(msg);
        }
        ROS_INFO("Loaded %zu velocity messages", velMsgs.size());

        // -> GPS Messages
        rosbag::View gps_view(bag, rosbag::TopicQuery(GPS_TOPIC));
        for (const rosbag::MessageInstance &m : gps_view)
        {
            auto msg = m.instantiate<sensor_msgs::NavSatFix>();
            if (msg)
                gpsMsgs.push_back(msg);
        }
        ROS_INFO("Loaded %zu GPS messages", gpsMsgs.size());

        // =========================================================
        // Initializing ORB - 500 Features
        // =========================================================
        orb = cv::ORB::create(750);

        // =========================================================
        // Initializing GTSAM
        // =========================================================
        gtsam::Pose3 prior;
        noise = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1).finished());
        graph.add(gtsam::PriorFactor<gtsam::Pose3>(gtsam::Symbol('x', 0), prior, noise));
        values.insert(gtsam::Symbol('x', 0), prior);

        bool firstFrame = true;
        int frameIdx = 0;
        ros::WallTime start = ros::WallTime::now();
        // =========================================================
        // Main Loop - Processing each frame
        // =========================================================
        for (const rosbag::MessageInstance &m : cam_view)
        {
            ROS_INFO("Processing frame %d", frameIdx++);

            auto imgMsg = m.instantiate<sensor_msgs::CompressedImage>();
            if (!imgMsg)
                continue;

            // =========================================================
            // Undistorting Image
            // =========================================================
            undistortImage(imgMsg);

            // =========================================================
            // If first frame, just detect features
            // =========================================================
            if (firstFrame)
            {
                featureDetection(currUndistortedGrey);
                prevPoints = currPoints;
                prevUndistortedGrey = currUndistortedGrey.clone();
                firstFrame = false;
                continue;
            }

            // =========================================================
            // Verifiying we have previous points to track
            // =========================================================
            if (prevPoints.pts.size() < 25) // At least 25 points to track
            {
                ROS_WARN("No previous points to track, detecting new features");
                featureDetection(currUndistortedGrey);
                prevPoints = currPoints;
                prevUndistortedGrey = currUndistortedGrey.clone();
                continue;
            }

            // =========================================================
            // Feature Tracking (KLT)
            // =========================================================
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

            // =========================================================
            // Build filtered correspondences
            // =========================================================
            Points currFiltered;
            std::vector<cv::Point2f> prevValid, currValid;

            auto inImage = [&](const cv::Point2f &p, const cv::Mat &img)
            {
                return p.x >= 8 && p.y >= 8 &&
                       p.x < img.cols - 8 &&
                       p.y < img.rows - 8;
            };

            for (size_t k = 0; k < status.size(); k++)
            {
                if (!status[k])
                    continue;

                const cv::Point2f &p_prev = prevPoints.pts[k];
                const cv::Point2f &p_curr = trackedPoints[k];

                if (!inImage(p_prev, prevUndistortedGrey) ||
                    !inImage(p_curr, currUndistortedGrey))
                    continue;

                if (cv::norm(p_curr - p_prev) > 60.0f)
                    continue;

                prevValid.push_back(p_prev);
                currValid.push_back(p_curr);

                currFiltered.pts.push_back(p_curr);
                currFiltered.ids.push_back(prevPoints.ids[k]);
            }

            currPoints = currFiltered;

            // =========================================================
            // If number of tracked points below threshold, detect new features
            // =========================================================
            if (currPoints.pts.size() < KF_FEATURE_THRESHOLD)
            {
                ROS_INFO("[Frame %d] Replenishing features. Current count: %zu",
                         frameIdx, currPoints.pts.size());
                replenishFeatures(currUndistortedGrey);
            }

            // =========================================================
            // Debug Image Publishing
            // =========================================================
            cv::Mat prevBgr, currBgr;
            cv::cvtColor(prevUndistortedGrey, prevBgr, cv::COLOR_GRAY2BGR);
            cv::cvtColor(currUndistortedGrey, currBgr, cv::COLOR_GRAY2BGR);

            cv::Mat debugImg;
            cv::hconcat(prevBgr, currBgr, debugImg);

            int offsetX = prevBgr.cols;

            for (size_t k = 0; k < prevValid.size(); k++)
            {
                cv::Point2f p1 = prevValid[k];
                cv::Point2f p2 = currValid[k] + cv::Point2f((float)offsetX, 0.0f);

                cv::circle(debugImg, p1, 3, cv::Scalar(0, 255, 0), -1);
                cv::circle(debugImg, p2, 3, cv::Scalar(0, 0, 255), -1);
                cv::line(debugImg, p1, p2, cv::Scalar(255, 0, 0), 1);
            }

            sensor_msgs::ImagePtr debugMsg =
                cv_bridge::CvImage(std_msgs::Header(), "bgr8", debugImg).toImageMsg();

            debugMsg->header.stamp = m.getTime();
            debugPub.publish(debugMsg);

            // =========================================================
            // Preparing for next iteration
            // =========================================================
            prevUndistortedGrey = currUndistortedGrey.clone();
            prevPoints = currPoints;
        }

        ROS_INFO("VIO Processing Complete.");
        ros::WallTime finish = ros::WallTime::now();
        ROS_INFO("Total Processing Time: %.2f seconds", (finish - start).toSec());
        bag.close();
    }

private:
    /*
     * ================================================================================================================
     * Variables
     * ================================================================================================================
     */
    // ROS Variables
    ros::NodeHandle nh;
    ros::Publisher debugPub, pathPub, cloudPub;
    rosbag::Bag bag;

    // ROSBAG Messages Vectors
    std::vector<sensor_msgs::CompressedImageConstPtr> camMsgs;
    std::vector<geometry_msgs::QuaternionStampedConstPtr> quatMsgs;
    std::vector<geometry_msgs::Vector3StampedConstPtr> velMsgs;
    std::vector<sensor_msgs::NavSatFixConstPtr> gpsMsgs;

    // Camera Parameters
    const cv::Mat K = (cv::Mat_<double>(3, 3) << 1372.76165, 0.0, 960.45289,
                       0.0, 1372.14817, 515.00383,
                       0.0, 0.0, 1.0);
    const cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << 0.048300, -0.044905, -0.003731, -0.001349, 0);

    // ORB Detector
    cv::Ptr<cv::ORB> orb;

    // Images
    cv::Mat currUndistortedGrey, currUndistortedRGB;
    cv::Mat prevUndistortedGrey, prevUndistortedRGB;

    // Feature Points
    Points currPoints, prevPoints;
    int nextFeatureID = 0;

    // GTSAM Variables
    gtsam::NonlinearFactorGraph graph;
    gtsam::noiseModel::Diagonal::shared_ptr noise;
    gtsam::Values values;

    /*
     * ================================================================================================================
     * Methods
     * ================================================================================================================
     */
    // Method for Undistorting an Image
    void undistortImage(sensor_msgs::CompressedImageConstPtr msg)
    {
        cv::Mat raw = cv::imdecode(msg->data, cv::IMREAD_COLOR);
        cv::undistort(raw, currUndistortedRGB, K, distCoeffs);
        cv::cvtColor(currUndistortedRGB, currUndistortedGrey, cv::COLOR_BGR2GRAY);
    }

    // Method for Feature Detection using ORB
    void featureDetection(const cv::Mat &img)
    {
        currPoints.pts.clear();
        currPoints.ids.clear();

        // Variable Declarations
        std::vector<cv::KeyPoint> kp;

        // ORB Keypoint Detection
        orb->detect(img, kp);

        // Ordering Keypoints by Response
        std::vector<int> idx(kp.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b)
                  { return kp[a].response > kp[b].response; });

        // Defining a Mask in order to set a minimum distance between keypoints
        cv::Mat mask(img.size(), CV_8UC1, cv::Scalar(255));
        int radius = 21;

        // Selecting the ORB_N_BEST Keypoints
        int kept = 0;
        for (int id : idx)
        {
            cv::Point2f p = kp[id].pt;
            int x = cvRound(p.x);
            int y = cvRound(p.y);

            if (x < 0 || x >= img.cols || y < 0 || y >= img.rows)
                continue;

            if (mask.at<uint8_t>(y, x) == 255)
            {
                currPoints.pts.push_back(p);
                currPoints.ids.push_back(nextFeatureID++);
                cv::circle(mask, cv::Point(x, y), radius, 0, -1);

                kept++;
                if (kept == ORB_N_BEST)
                    break;
            }
        }
    }

    // Method for Replenishing Tracked Features
    void replenishFeatures(const cv::Mat &img)
    {
        // Calculating number of features to detect
        int nToDetect = ORB_N_BEST - static_cast<int>(currPoints.pts.size());
        if (nToDetect <= 0)
            return;

        // Mask Creation
        // Used to mask the existing features
        cv::Mat mask(img.size(), CV_8UC1, cv::Scalar(255));
        int radius = 21;

        // Masking existing features
        for (const auto &p : currPoints.pts)
        {
            int x = cvRound(p.x), y = cvRound(p.y);
            if (p.x >= 0 && p.x < img.cols && p.y >= 0 && p.y < img.rows)
                cv::circle(mask, p, radius, cv::Scalar(0), -1);
        }

        // Variable Declaration for Feature Detection
        std::vector<cv::KeyPoint> kpNew;

        // ORB Keypoint Detection
        orb->detect(img, kpNew, mask);

        // Ordering Keypoints by Response
        std::vector<int> idx(kpNew.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b)
                  { return kpNew[a].response > kpNew[b].response; });

        // Selecting the required number of keypoints
        int added = 0;
        for (int j : idx)
        {
            if (added >= nToDetect)
                break;

            const cv::Point2f p = kpNew[j].pt;
            int x = cvRound(p.x), y = cvRound(p.y);

            if (x < 0 || x >= img.cols || y < 0 || y >= img.rows)
                continue;

            // still free?
            if (mask.at<uint8_t>(y, x) == 0)
                continue;

            currPoints.pts.push_back(p);
            currPoints.ids.push_back(nextFeatureID++);

            // block area so new points don't cluster
            cv::circle(mask, cv::Point(x, y), radius, cv::Scalar(0), -1);

            added++;
        }
    }
};

/*
 * ================================================================================================================
 * Main
 * ================================================================================================================
 */
int main(int argc, char **argv)
{
    ros::init(argc, argv, "fuims_vio");
    vioManager manager;
    return 0;
}