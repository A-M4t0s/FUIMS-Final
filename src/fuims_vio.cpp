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

#define BAG_PATH "/home/tony/catkin_ws/bags/bags_Agucadoura/bom/2025-07-03-13-03-16.bag"
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

// ============================================================================
// Structures
// ============================================================================

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

// ============================================================================
// Helper geo functions
// ============================================================================

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

// ============================================================================
// vioManager
// ============================================================================

class vioManager
{
public:
    vioManager()
    {
        debugPub = nh.advertise<sensor_msgs::Image>("vio/debug_image", 1);
        pathPub = nh.advertise<nav_msgs::Path>("vio/path", 1);
        cloudPub = nh.advertise<sensor_msgs::PointCloud2>("vio/points", 1);
        pathMsg.header.frame_id = "map";

        // ==========================
        // Open Bag
        // ==========================
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

        orb = cv::ORB::create(500);

        // ==========================
        // Load Camera Messages
        // ==========================
        rosbag::View cam_view(bag, rosbag::TopicQuery(CAMERA_TOPIC));
        camMsgs.reserve(cam_view.size());
        for (const rosbag::MessageInstance &m : cam_view)
        {
            auto msg = m.instantiate<sensor_msgs::CompressedImage>();
            if (msg)
                camMsgs.push_back(msg);
        }

        // ==========================
        // Load Quaternion Messages
        // ==========================
        rosbag::View quat_view(bag, rosbag::TopicQuery(QUATERNION_TOPIC));
        for (const rosbag::MessageInstance &m : quat_view)
        {
            auto msg = m.instantiate<geometry_msgs::QuaternionStamped>();
            if (msg)
                quatMsgs.push_back(msg);
        }
        ROS_INFO("Loaded %zu quaternion messages", quatMsgs.size());

        // ==========================
        // Load Velocity Messages
        // ==========================
        rosbag::View vel_view(bag, rosbag::TopicQuery(VELOCITY_TOPIC));
        for (const rosbag::MessageInstance &m : vel_view)
        {
            auto msg = m.instantiate<geometry_msgs::Vector3Stamped>();
            if (msg)
                velMsgs.push_back(msg);
        }
        ROS_INFO("Loaded %zu velocity messages", velMsgs.size());

        uint32_t nImages = camMsgs.size();

        // ==========================
        // First frame
        // ==========================
        undistortImage(0);
        featureDetection(currUndistortedGrey);
        std::swap(prevUndistortedGrey, currUndistortedGrey);
        std::swap(prevUndistortedRGB, currUndistortedRGB);
        prevPoints = currPoints;

        Keyframe firstKF;
        firstKF.frameID = 0;
        firstKF.points = prevPoints;
        firstKF.greyImg = prevUndistortedGrey.clone();
        firstKF.R = cv::Mat::eye(3, 3, CV_64F);
        firstKF.t = cv::Mat::zeros(3, 1, CV_64F);
        keyframes.push_back(firstKF);

        ROS_INFO("Initialized first keyframe with %zu features", prevPoints.pts.size());

        // ==========================
        // Init GTSAM
        // ==========================
        gtsam::Pose3 prior;
        noise = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1).finished());
        gtsam::Symbol key0('x', poseIndex);
        graph.add(gtsam::PriorFactor<gtsam::Pose3>(key0, prior, noise));
        values.insert(key0, prior);
        poseIndex++;

        ros::WallTime start_time = ros::WallTime::now();
        ROS_INFO("Starting Feature Tracking");

        // =====================================================================
        // MAIN LOOP
        // =====================================================================
        for (int i = 1; i < (int)nImages; i++)
        {
            undistortImage(i);

            // ------------------------------------------
            // IMU-LIKE SYNC (quaternion + velocity)
            // ------------------------------------------
            ros::Time tStamp = camMsgs[i]->header.stamp;

            int qi = findNearestIdx(quatMsgs, tStamp);
            int vi = findNearestIdx(velMsgs, tStamp);

            // Orientação a partir do quaternion DJI
            if (qi >= 0)
            {
                const auto &qmsg = quatMsgs[qi]->quaternion;
                tf2::Quaternion tfq(qmsg.x, qmsg.y, qmsg.z, qmsg.w);
                tf2::Matrix3x3 tfm(tfq);

                if (global_R.empty())
                    global_R = cv::Mat::eye(3, 3, CV_64F);

                for (int r = 0; r < 3; ++r)
                {
                    for (int c = 0; c < 3; ++c)
                    {
                        global_R.at<double>(r, c) = tfm[r][c];
                    }
                }

                ROS_DEBUG("Frame %d | Quaternion: [%.3f %.3f %.3f %.3f]",
                          i, qmsg.x, qmsg.y, qmsg.z, qmsg.w);
            }

            if (vi >= 0)
            {
                const auto &v = velMsgs[vi]->vector;
                ROS_DEBUG("Frame %d | Velocity: [%.3f %.3f %.3f]",
                          i, v.x, v.y, v.z);
            }

            // ------------------------------------------
            // Tracking + VO
            // ------------------------------------------
            if (prevPoints.pts.size() < 120)
                replenishFeatures(currUndistortedGrey, ORB_N_BEST);

            if (prevPoints.pts.size() < 10)
            {
                featureDetection(currUndistortedGrey);
                prevPoints = currPoints;
                continue;
            }

            // Optical flow
            std::vector<cv::Point2f> nextPoints;
            std::vector<uchar> status;
            std::vector<float> error;
            cv::calcOpticalFlowPyrLK(prevUndistortedGrey,
                                     currUndistortedGrey,
                                     prevPoints.pts,
                                     nextPoints,
                                     status,
                                     error,
                                     cv::Size(21, 21),
                                     2);

            currPoints.pts = nextPoints;
            currPoints.ids = prevPoints.ids;

            // Filter valid tracks
            Points prevBestPts, currBestPts;
            for (int k = 0; k < (int)prevPoints.pts.size(); k++)
            {
                if (status[k])
                {
                    prevBestPts.pts.push_back(prevPoints.pts[k]);
                    currBestPts.pts.push_back(currPoints.pts[k]);
                    prevBestPts.ids.push_back(prevPoints.ids[k]);
                    currBestPts.ids.push_back(currPoints.ids[k]);
                }
            }

            // RANSAC filtering
            if (prevBestPts.pts.size() >= 8)
            {
                std::vector<uchar> inlierMask;
                cv::Mat F = cv::findFundamentalMat(
                    prevBestPts.pts, currBestPts.pts,
                    cv::FM_RANSAC, 3.0, 0.99, inlierMask);

                if (!F.empty())
                {
                    Points prevInliers, currInliers;
                    for (int k = 0; k < (int)inlierMask.size(); k++)
                    {
                        if (inlierMask[k])
                        {
                            prevInliers.pts.push_back(prevBestPts.pts[k]);
                            currInliers.pts.push_back(currBestPts.pts[k]);
                            prevInliers.ids.push_back(prevBestPts.ids[k]);
                            currInliers.ids.push_back(currBestPts.ids[k]);
                        }
                    }
                    prevBestPts = prevInliers;
                    currBestPts = currInliers;
                }
            }

            // Essential + Pose
            if (prevBestPts.pts.size() >= 8)
            {
                cv::Mat E = cv::findEssentialMat(
                    prevBestPts.pts,
                    currBestPts.pts,
                    K,
                    cv::RANSAC,
                    0.999,
                    1.0);

                if (!E.empty())
                {
                    cv::Mat R, t;
                    int inl = cv::recoverPose(
                        E,
                        prevBestPts.pts,
                        currBestPts.pts,
                        K,
                        R, t);

                    last_R = R.clone();
                    last_t = t.clone();

                    // -------------------------------------------------
                    // Corrigir ESCALA com base na velocidade IMU
                    // -------------------------------------------------
                    double vo_speed = cv::norm(t);
                    if (vi >= 0 && vo_speed > 1e-4)
                    {
                        const auto &v = velMsgs[vi]->vector;
                        double imu_speed = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);

                        if (imu_speed > 0.1)
                        {
                            double scale = imu_speed / vo_speed;
                            t *= scale;
                        }
                    }

                    // Integração da translação em frame global
                    // global_R já vem do IMU (quaternion)
                    global_t = global_t + global_R * t;

                    publishPoseToPath(global_R, global_t);

                    // GTSAM VO factor (usa t escalado)
                    gtsam::Matrix3 gR;
                    for (int r = 0; r < 3; r++)
                        for (int c = 0; c < 3; c++)
                            gR(r, c) = R.at<double>(r, c);

                    gtsam::Pose3 dpose(
                        gtsam::Rot3(gR),
                        gtsam::Point3(t.at<double>(0),
                                      t.at<double>(1),
                                      t.at<double>(2)));

                    gtsam::Symbol keyPrev('x', poseIndex - 1);
                    gtsam::Symbol keyCurr('x', poseIndex);

                    if (values.exists(keyPrev))
                    {
                        graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
                            keyPrev, keyCurr, dpose, noise));

                        gtsam::Pose3 prevPose = values.at<gtsam::Pose3>(keyPrev);
                        values.insert(keyCurr, prevPose.compose(dpose));
                        poseIndex++;
                    }

                    ROS_DEBUG("Frame %d: recoverPose inliers = %d", i, inl);
                }
            }

            keyframeCompare(currBestPts, i);

            prevPoints = currBestPts;
            std::swap(prevUndistortedGrey, currUndistortedGrey);
            std::swap(prevUndistortedRGB, currUndistortedRGB);
            currPoints.ids.clear();
            currPoints.pts.clear();
        }

        ros::WallTime finish = ros::WallTime::now();
        ROS_INFO("Finished Feature Tracking in %.3f s",
                 (finish - start_time).toSec());
    }

private:
    // =====================================================================
    // Nearest message lookup (USED FOR IMU SYNC)
    // =====================================================================
    template <typename MsgPtr>
    int findNearestIdx(const std::vector<MsgPtr> &msgs, const ros::Time &t)
    {
        if (msgs.empty())
            return -1;

        double best_dt = 1e9;
        int best = -1;

        for (int i = 0; i < (int)msgs.size(); i++)
        {
            double dt = fabs((msgs[i]->header.stamp - t).toSec());
            if (dt < best_dt)
            {
                best_dt = dt;
                best = i;
            }
        }
        return best;
    }

    // =====================================================================
    // Members
    // =====================================================================

    ros::NodeHandle nh;
    ros::Publisher debugPub, pathPub, cloudPub;

    rosbag::Bag bag;
    std::vector<sensor_msgs::CompressedImageConstPtr> camMsgs;

    std::vector<geometry_msgs::QuaternionStampedConstPtr> quatMsgs;
    std::vector<geometry_msgs::Vector3StampedConstPtr> velMsgs;

    nav_msgs::Path pathMsg;

    const cv::Mat K = (cv::Mat_<double>(3, 3) << 1372.76165, 0.0, 960.45289,
                       0.0, 1372.14817, 515.00383,
                       0.0, 0.0, 1.0);

    const cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << 0.048300, -0.044905, -0.003731, -0.001349, 0);

    cv::Ptr<cv::ORB> orb;

    cv::Mat currUndistortedGrey, currUndistortedRGB;
    cv::Mat prevUndistortedGrey, prevUndistortedRGB;

    Points currPoints, prevPoints;
    int nextFeatureID = 0;

    int poseIndex = 0;

    std::vector<Keyframe> keyframes;

    cv::Mat last_R = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat last_t = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat global_R = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat global_t = cv::Mat::zeros(3, 1, CV_64F);

    gtsam::NonlinearFactorGraph graph;
    gtsam::noiseModel::Diagonal::shared_ptr noise;
    gtsam::Values values;

    // =====================================================================
    // Image functions + KF + Triangulation
    // =====================================================================

    void undistortImage(int frameIdx)
    {
        const auto &msg = camMsgs[frameIdx];
        cv::Mat raw = cv::imdecode(msg->data, cv::IMREAD_COLOR);

        cv::undistort(raw, currUndistortedRGB, K, distCoeffs);
        cv::cvtColor(currUndistortedRGB, currUndistortedGrey, cv::COLOR_BGR2GRAY);
    }

    void featureDetection(const cv::Mat &img)
    {
        currPoints.ids.clear();
        currPoints.pts.clear();

        std::vector<cv::KeyPoint> kp;
        cv::Mat desc;
        orb->detectAndCompute(img, cv::noArray(), kp, desc);
        if (kp.empty())
            return;

        std::vector<int> idx(kp.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b)
                  { return kp[a].response > kp[b].response; });

        cv::Mat mask(img.size(), CV_8UC1, cv::Scalar(255));
        int radius = 21;

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

    void replenishFeatures(cv::Mat frame, int target)
    {
        if (prevPoints.pts.size() >= static_cast<size_t>(target))
            return;

        int need = target - static_cast<int>(prevPoints.pts.size());
        if (need <= 0)
            return;

        cv::Mat mask(frame.size(), CV_8UC1, cv::Scalar(255));
        int radius = 21;

        for (const auto &p : prevPoints.pts)
        {
            if (p.x >= 0 && p.x < frame.cols && p.y >= 0 && p.y < frame.rows)
                cv::circle(mask, p, radius, cv::Scalar(0), -1);
        }

        std::vector<cv::KeyPoint> kp_new;
        cv::Mat desc_new;
        orb->detectAndCompute(frame, mask, kp_new, desc_new);
        if (kp_new.empty())
            return;

        std::vector<int> idx(kp_new.size());
        std::iota(idx.begin(), idx.end(), 0);

        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b)
                  { return kp_new[a].response > kp_new[b].response; });

        int keep = std::min(need, static_cast<int>(kp_new.size()));

        for (int i = 0; i < keep; i++)
        {
            int id = idx[i];
            prevPoints.pts.emplace_back(kp_new[id].pt);
            prevPoints.ids.emplace_back(nextFeatureID++);
        }
    }

    void publishDebugImage(const cv::Mat &img)
    {
        if (img.empty())
            return;

        cv_bridge::CvImage cvImg;
        cvImg.header.stamp = ros::Time::now();
        cvImg.encoding = sensor_msgs::image_encodings::BGR8;
        cvImg.image = img;

        debugPub.publish(cvImg.toImageMsg());
    }

    void keyframeCompare(const Points &currFramePts, int currFrame)
    {
        bool makeKF = false;
        double parallax = 0.0;
        int cnt = 0;
        double sum = 0.0;

        if (keyframes.empty())
            return;

        const Keyframe &KF = keyframes.back();

        for (size_t i = 0; i < KF.points.ids.size(); i++)
        {
            int idKF = KF.points.ids[i];
            auto it = std::find(currFramePts.ids.begin(), currFramePts.ids.end(), idKF);
            if (it == currFramePts.ids.end())
                continue;

            int j = std::distance(currFramePts.ids.begin(), it);

            double dx = currFramePts.pts[j].x - KF.points.pts[i].x;
            double dy = currFramePts.pts[j].y - KF.points.pts[i].y;

            sum += std::sqrt(dx * dx + dy * dy);
            cnt++;
        }

        if (cnt > 0)
            parallax = sum / cnt;

        if (parallax > KF_PARALLAX_THRESHOLD)
            makeKF = true;

        int survived = 0;
        for (int idKF : KF.points.ids)
        {
            auto it = std::find(currFramePts.ids.begin(), currFramePts.ids.end(), idKF);
            if (it != currFramePts.ids.end())
                survived++;
        }

        if (survived < 30)
            makeKF = true;

        if (!makeKF)
            return;

        Keyframe newKF;
        newKF.frameID = currFrame;
        newKF.points = currFramePts;
        newKF.greyImg = currUndistortedGrey.clone();

        newKF.R = last_R.clone();
        newKF.t = last_t.clone();

        keyframes.push_back(newKF);

        ROS_WARN("NEW KEYFRAME %d | parallax = %.2f | features = %ld",
                 currFrame,
                 parallax,
                 static_cast<long>(currFramePts.pts.size()));

        triangulateLastKeyframes();
    }

    void triangulateLastKeyframes()
    {
        if (keyframes.size() < 2)
            return;

        const Keyframe &KF1 = keyframes[keyframes.size() - 2];
        const Keyframe &KF2 = keyframes[keyframes.size() - 1];

        std::vector<cv::Point2f> pts1, pts2;
        for (size_t k = 0; k < KF1.points.ids.size(); k++)
        {
            int id = KF1.points.ids[k];
            auto it = std::find(KF2.points.ids.begin(), KF2.points.ids.end(), id);
            if (it == KF2.points.ids.end())
                continue;

            int j = std::distance(KF2.points.ids.begin(), it);
            pts1.push_back(KF1.points.pts[k]);
            pts2.push_back(KF2.points.pts[j]);
        }

        if (pts1.size() < 8)
        {
            ROS_WARN("Triangulation skipped: only %zu correspondences", pts1.size());
            return;
        }

        std::vector<uchar> inlierMask;
        cv::Mat E = cv::findEssentialMat(
            pts1,
            pts2,
            K,
            cv::RANSAC,
            0.999,
            1.0,
            inlierMask);

        if (E.empty())
        {
            ROS_WARN("Triangulation: Essential matrix is empty");
            return;
        }

        std::vector<cv::Point2f> inPts1, inPts2;
        for (size_t i = 0; i < inlierMask.size(); ++i)
        {
            if (inlierMask[i])
            {
                inPts1.push_back(pts1[i]);
                inPts2.push_back(pts2[i]);
            }
        }

        if (inPts1.size() < 8)
        {
            ROS_WARN("Triangulation skipped after RANSAC: only %zu inliers",
                     inPts1.size());
            return;
        }

        cv::Mat R, t;
        int inliers = cv::recoverPose(
            E,
            inPts1,
            inPts2,
            K,
            R,
            t);

        cv::Mat P1 = cv::Mat::zeros(3, 4, CV_64F);
        P1.at<double>(0, 0) = 1.0;
        P1.at<double>(1, 1) = 1.0;
        P1.at<double>(2, 2) = 1.0;

        cv::Mat P2 = cv::Mat::zeros(3, 4, CV_64F);
        R.copyTo(P2(cv::Range(0, 3), cv::Range(0, 3)));
        P2.at<double>(0, 3) = t.at<double>(0);
        P2.at<double>(1, 3) = t.at<double>(1);
        P2.at<double>(2, 3) = t.at<double>(2);

        P1 = K * P1;
        P2 = K * P2;

        cv::Mat points4D;
        cv::triangulatePoints(P1, P2, inPts1, inPts2, points4D);

        std::vector<cv::Point3f> pts3D;
        pts3D.reserve(points4D.cols);

        for (int i = 0; i < points4D.cols; i++)
        {
            cv::Mat col = points4D.col(i);
            double w = col.at<float>(3);

            if (std::fabs(w) < 1e-6)
                continue;

            double X = col.at<float>(0) / w;
            double Y = col.at<float>(1) / w;
            double Z = col.at<float>(2) / w;

            if (Z <= 0.0)
                continue;

            pts3D.emplace_back(static_cast<float>(X),
                               static_cast<float>(Y),
                               static_cast<float>(Z));
        }

        ROS_INFO("Triangulated %zu 3D points between KF %d and KF %d (inliers=%d, matches=%zu)",
                 pts3D.size(), KF1.frameID, KF2.frameID, inliers, inPts1.size());

        if (!pts3D.empty())
            publishPointCloud(pts3D);
    }

    void publishPoseToPath(const cv::Mat &R, const cv::Mat &t)
    {
        geometry_msgs::PoseStamped ps;
        ps.header.stamp = ros::Time::now();
        ps.header.frame_id = "map";

        ps.pose.position.x = t.at<double>(0);
        ps.pose.position.y = t.at<double>(1);
        ps.pose.position.z = t.at<double>(2);

        cv::Mat rvec;
        cv::Rodrigues(R, rvec);
        double angle = cv::norm(rvec);

        tf2::Quaternion q;

        if (angle < 1e-8)
        {
            q.setValue(0.0, 0.0, 0.0, 1.0);
        }
        else
        {
            cv::Vec3d axis(
                rvec.at<double>(0) / angle,
                rvec.at<double>(1) / angle,
                rvec.at<double>(2) / angle);
            q.setRotation(tf2::Vector3(axis[0], axis[1], axis[2]), angle);
        }

        ps.pose.orientation.x = q.x();
        ps.pose.orientation.y = q.y();
        ps.pose.orientation.z = q.z();
        ps.pose.orientation.w = q.w();

        pathMsg.poses.push_back(ps);
        pathMsg.header.stamp = ps.header.stamp;

        pathPub.publish(pathMsg);
    }

    void publishPointCloud(const std::vector<cv::Point3f> &pts)
    {
        sensor_msgs::PointCloud2 cloud;
        cloud.header.stamp = ros::Time::now();
        cloud.header.frame_id = "map";

        cloud.height = 1;
        cloud.width = pts.size();

        sensor_msgs::PointCloud2Modifier modifier(cloud);
        modifier.setPointCloud2FieldsByString(1, "xyz");
        modifier.resize(pts.size());

        sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");

        for (size_t i = 0; i < pts.size(); i++, ++iter_x, ++iter_y, ++iter_z)
        {
            *iter_x = pts[i].x;
            *iter_y = pts[i].y;
            *iter_z = pts[i].z;
        }

        cloudPub.publish(cloud);
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "fuims_vio");
    vioManager manager;
    return 0;
}
