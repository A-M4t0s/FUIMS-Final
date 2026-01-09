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
    int frameID = -1;
    ros::Time timestamp;
    cv::Mat R, t;
    Points points;
    cv::Mat greyImg;
};

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

class vioManager
{
public:
    vioManager()
    {
        signal(SIGINT, signalHandler);

        debugPub = nh.advertise<sensor_msgs::Image>("vio/debug_image", 1);

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

        ROS_INFO("Loading messages from ROSBAG...");

        rosbag::View cam_view(bag, rosbag::TopicQuery(CAMERA_TOPIC));

        rosbag::View quat_view(bag, rosbag::TopicQuery(QUATERNION_TOPIC));
        for (const rosbag::MessageInstance &m : quat_view)
        {
            auto msg = m.instantiate<geometry_msgs::QuaternionStamped>();
            if (msg)
                quatMsgs.push_back(msg);
        }
        ROS_INFO("Loaded %zu quaternion messages", quatMsgs.size());

        rosbag::View vel_view(bag, rosbag::TopicQuery(VELOCITY_TOPIC));
        for (const rosbag::MessageInstance &m : vel_view)
        {
            auto msg = m.instantiate<geometry_msgs::Vector3Stamped>();
            if (msg)
                velMsgs.push_back(msg);
        }
        ROS_INFO("Loaded %zu velocity messages", velMsgs.size());

        rosbag::View gps_view(bag, rosbag::TopicQuery(GPS_TOPIC));
        for (const rosbag::MessageInstance &m : gps_view)
        {
            auto msg = m.instantiate<sensor_msgs::NavSatFix>();
            if (msg)
                gpsMsgs.push_back(msg);
        }
        ROS_INFO("Loaded %zu GPS messages", gpsMsgs.size());

        orb = cv::ORB::create(750);

        // =========================================================
        // GTSAM init
        // =========================================================
        gtsam::Pose3 prior; // identity
        noise = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1).finished());

        graph.add(gtsam::PriorFactor<gtsam::Pose3>(gtsam::Symbol('x', 0), prior, noise));
        values.insert(gtsam::Symbol('x', 0), prior);

        bool firstFrame = true;
        int frameIdx = 0;

        ros::WallTime start = ros::WallTime::now();

        // =========================================================
        // Main Loop
        // =========================================================
        for (const rosbag::MessageInstance &m : cam_view)
        {
            if (g_requestShutdown)
            {
                ROS_WARN("Shutdown requested. Breaking processing loop.");
                break;
            }

            ROS_INFO("Processing frame %d", frameIdx);

            auto imgMsg = m.instantiate<sensor_msgs::CompressedImage>();
            if (!imgMsg)
            {
                frameIdx++;
                continue;
            }

            undistortImage(imgMsg);

            if (firstFrame)
            {
                featureDetection(currUndistortedGrey);
                prevPoints = currPoints;
                prevUndistortedGrey = currUndistortedGrey.clone();
                firstFrame = false;
                frameIdx++;
                continue;
            }

            // =========================================================
            // If too few prev points -> reset tracking (mas NÃO destruímos o KF global)
            // =========================================================
            if (prevPoints.pts.size() < 25)
            {
                ROS_WARN("No previous points to track, detecting new features");
                featureDetection(currUndistortedGrey);
                prevPoints = currPoints;
                prevUndistortedGrey = currUndistortedGrey.clone();
                frameIdx++;
                continue;
            }

            // =========================================================
            // KLT tracking
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

            Points currFiltered;
            std::vector<cv::Point2f> prevValid, currValid;

            auto inImage = [&](const cv::Point2f &p, const cv::Mat &img)
            {
                return p.x >= 8 && p.y >= 8 && p.x < img.cols - 8 && p.y < img.rows - 8;
            };

            for (size_t k = 0; k < status.size(); k++)
            {
                if (!status[k])
                    continue;

                const cv::Point2f &p_prev = prevPoints.pts[k];
                const cv::Point2f &p_curr = trackedPoints[k];

                if (!inImage(p_prev, prevUndistortedGrey) || !inImage(p_curr, currUndistortedGrey))
                    continue;

                if (cv::norm(p_curr - p_prev) > 60.0f)
                    continue;

                prevValid.push_back(p_prev);
                currValid.push_back(p_curr);

                currFiltered.pts.push_back(p_curr);
                currFiltered.ids.push_back(prevPoints.ids[k]);
            }

            currPoints = currFiltered;
            Points trackedOnly = currPoints; // antes de replenish

            if (currPoints.pts.size() < KF_FEATURE_THRESHOLD)
            {
                ROS_WARN("[Frame %d] Replenishing features. Current count: %zu",
                         frameIdx, currPoints.pts.size());
                replenishFeatures(currUndistortedGrey);
                ROS_INFO("New feature count: %zu", currPoints.pts.size());
            }

            // =========================================================
            // Keyframe decision
            // =========================================================
            bool isKeyframe = false;

            if (!hasKF)
            {
                isKeyframe = true;
            }
            else
            {
                double kfParallax = 0.0;

                // Só faz sentido medir paralaxe se tens tracking minimamente válido
                if (trackedOnly.pts.size() >= 15)
                    kfParallax = computeKFParallax(lastKF, trackedOnly);

                ROS_INFO("KF Parallax = %.2f", kfParallax);

                if (kfParallax > KF_PARALLAX_THRESHOLD)
                    isKeyframe = true;

                // tracking fraco -> força KF
                if (trackedOnly.pts.size() < KF_FEATURE_THRESHOLD)
                    isKeyframe = true;
            }

            // =========================================================
            // Handle keyframe (KF SEMPRE COMMIT)
            // =========================================================
            if (isKeyframe)
            {
                ROS_WARN("=== New Keyframe at frame %d ===", frameIdx);
                const ros::Time t_kf = m.getTime();

                // Primeiro KF: só define referência visual
                if (!hasKF)
                {
                    lastKF.frameID = frameIdx;
                    lastKF.greyImg = currUndistortedGrey.clone();
                    lastKF.points = currPoints;
                    lastKF.timestamp = t_kf;
                    hasKF = true;
                    kfIndex = 0;
                }
                else
                {
                    const int prevK = kfIndex;
                    const int curK = kfIndex + 1;

                    // Correspondências KF->cur por ID
                    std::vector<cv::Point2f> kfPts, curPtsFromKF;
                    buildCorrespondencesById(lastKF.points, currPoints, kfPts, curPtsFromKF);

                    bool hasVisual = false;

                    // --- Visual factor se possível
                    if (kfPts.size() >= 8)
                    {
                        hasVisual = true;

                        gtsam::Pose3 T_kf_curr = relativeMovementEstimation(kfPts, curPtsFromKF);

                        auto odomNoise = gtsam::noiseModel::Diagonal::Sigmas(
                            (gtsam::Vector(6) << 0.2, 0.2, 0.2, 0.1, 0.1, 0.1).finished());

                        graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
                            gtsam::Symbol('x', prevK),
                            gtsam::Symbol('x', curK),
                            T_kf_curr,
                            odomNoise));

                        gtsam::Pose3 seed =
                            values.at<gtsam::Pose3>(gtsam::Symbol('x', prevK)).compose(T_kf_curr);

                        values.insert(gtsam::Symbol('x', curK), seed);
                    }
                    else
                    {
                        // --- Seed dead-reckoning (mantém pose anterior)
                        values.insert(
                            gtsam::Symbol('x', curK),
                            values.at<gtsam::Pose3>(gtsam::Symbol('x', prevK)));

                        ROS_WARN("VO unavailable (KF correspondences=%zu) -> dead-reckoning KF", kfPts.size());
                    }

                    // --- Velocity factor (sempre)
                    {
                        Eigen::Vector3d dp = integrateVelocity(lastKF.timestamp, t_kf);

                        gtsam::Pose3 velDelta(
                            gtsam::Rot3(),
                            gtsam::Point3(dp.x(), dp.y(), dp.z()));

                        auto velNoise = gtsam::noiseModel::Diagonal::Sigmas(
                            (gtsam::Vector(6) << 0.3, 0.3, 0.3,
                             1e6, 1e6, 1e6)
                                .finished());

                        graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
                            gtsam::Symbol('x', prevK),
                            gtsam::Symbol('x', curK),
                            velDelta,
                            velNoise));
                    }

                    // --- Quaternion prior (sempre que existir)
                    {
                        auto qmsg = findNearestQuat(t_kf);
                        if (qmsg)
                        {
                            gtsam::Rot3 Rq = gtsam::Rot3::Quaternion(
                                qmsg->quaternion.w,
                                qmsg->quaternion.x,
                                qmsg->quaternion.y,
                                qmsg->quaternion.z);

                            gtsam::Pose3 rotPrior(Rq, gtsam::Point3(0, 0, 0));

                            auto rotNoise = gtsam::noiseModel::Diagonal::Sigmas(
                                (gtsam::Vector(6) << 1e6, 1e6, 1e6,
                                 0.05, 0.05, 0.05)
                                    .finished());

                            graph.add(gtsam::PriorFactor<gtsam::Pose3>(
                                gtsam::Symbol('x', curK),
                                rotPrior,
                                rotNoise));
                        }
                    }

                    // Commit KF SEMPRE
                    kfIndex = curK;

                    // Atualiza referência KF SEMPRE (para evitar “parallax 0 para sempre”)
                    lastKF.frameID = frameIdx;
                    lastKF.greyImg = currUndistortedGrey.clone();
                    lastKF.points = currPoints;
                    lastKF.timestamp = t_kf;
                }
            }

            // =========================================================
            // Debug publish (sempre dentro do loop)
            // =========================================================
            if (!prevUndistortedGrey.empty() && !currUndistortedGrey.empty())
            {
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
            }

            // Next iteration
            prevUndistortedGrey = currUndistortedGrey.clone();
            prevPoints = currPoints;

            frameIdx++;
        }

        ROS_INFO("VIO Processing Complete.");
        ros::WallTime finish = ros::WallTime::now();
        ROS_INFO("Total Processing Time: %.2f seconds", (finish - start).toSec());

        // =========================================================
        // Optimize
        // =========================================================
        ROS_INFO("Optimizing factor graph...");
        gtsam::LevenbergMarquardtOptimizer optimizer(graph, values);
        values = optimizer.optimize();
        ROS_INFO("Optimization complete.");

        // =========================================================
        // Extract optimized trajectory
        // =========================================================
        std::vector<gtsam::Pose3> optimizedPoses;
        optimizedPoses.reserve(kfIndex + 1);

        for (int k = 0; k <= kfIndex; k++)
        {
            gtsam::Pose3 pose = values.at<gtsam::Pose3>(gtsam::Symbol('x', k));
            optimizedPoses.push_back(pose);
        }

        // =========================================================
        // GPS -> ENU
        // =========================================================
        if (gpsMsgs.empty())
        {
            ROS_ERROR("No GPS messages available!");
            bag.close();
            return;
        }

        std::vector<ENU> gpsENU;
        gpsENU.reserve(gpsMsgs.size());

        double lat0 = gpsMsgs.front()->latitude * M_PI / 180.0;
        double lon0 = gpsMsgs.front()->longitude * M_PI / 180.0;
        double alt0 = gpsMsgs.front()->altitude;

        for (const auto &g : gpsMsgs)
        {
            double lat = g->latitude * M_PI / 180.0;
            double lon = g->longitude * M_PI / 180.0;
            double alt = g->altitude;

            double X, Y, Z;
            geodeticToECEF(lat, lon, alt, X, Y, Z);

            ENU enu = ecefToENU(X, Y, Z, lat0, lon0, alt0);
            gpsENU.push_back(enu);
        }

        // =========================================================
        // Drift XY final (comparação simples)
        // =========================================================
        std::vector<Eigen::Vector3d> vioPoints;
        vioPoints.reserve(optimizedPoses.size());

        Eigen::Vector3d p0 = optimizedPoses.front().translation();
        for (const auto &pose : optimizedPoses)
        {
            Eigen::Vector3d p = pose.translation() - p0;
            vioPoints.push_back(p);
        }

        const auto &p_vio_end = vioPoints.back();
        const auto &p_gps_end = gpsENU.back();

        double dx = p_vio_end.x() - p_gps_end.x;
        double dy = p_vio_end.y() - p_gps_end.y;

        double drift_xy = std::sqrt(dx * dx + dy * dy);
        ROS_INFO("Final XY drift vs GPS: %.2f m", drift_xy);

        // =========================================================
        // Save trajectory to CSV (for 3D plot)
        // =========================================================
        std::string home = std::getenv("HOME");
        std::string csv_path = home + "/vio_vs_gps.csv";

        std::ofstream csv(csv_path);
        if (!csv.is_open())
        {
            ROS_ERROR("Failed to open CSV file: %s", csv_path.c_str());
        }
        else
        {
            csv << "t,vio_x,vio_y,vio_z,gps_x,gps_y,gps_z\n";

            const size_t N = std::min(vioPoints.size(), gpsENU.size());

            for (size_t i = 0; i < N; i++)
            {
                csv << i << ","
                    << vioPoints[i].x() << ","
                    << vioPoints[i].y() << ","
                    << vioPoints[i].z() << ","
                    << gpsENU[i].x << ","
                    << gpsENU[i].y << ","
                    << gpsENU[i].z << "\n";
            }

            csv.close();
            ROS_INFO("Saved trajectory CSV to %s", csv_path.c_str());
        }

        bag.close();
    }

private:
    ros::NodeHandle nh;
    ros::Publisher debugPub;
    rosbag::Bag bag;

    std::vector<geometry_msgs::QuaternionStampedConstPtr> quatMsgs;
    std::vector<geometry_msgs::Vector3StampedConstPtr> velMsgs;
    std::vector<sensor_msgs::NavSatFixConstPtr> gpsMsgs;

    const cv::Mat K = (cv::Mat_<double>(3, 3) << 1372.76165, 0.0, 960.45289,
                       0.0, 1372.14817, 515.00383,
                       0.0, 0.0, 1.0);

    const cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << 0.048300, -0.044905, -0.003731, -0.001349, 0);

    cv::Ptr<cv::ORB> orb;

    cv::Mat currUndistortedGrey, currUndistortedRGB;
    cv::Mat prevUndistortedGrey;

    Keyframe lastKF;
    int kfIndex = 0;
    bool hasKF = false;

    Points currPoints, prevPoints;
    int nextFeatureID = 0;

    gtsam::NonlinearFactorGraph graph;
    gtsam::noiseModel::Diagonal::shared_ptr noise;
    gtsam::Values values;

private:
    void undistortImage(sensor_msgs::CompressedImageConstPtr msg)
    {
        cv::Mat raw = cv::imdecode(msg->data, cv::IMREAD_COLOR);
        if (raw.empty())
            return;

        cv::undistort(raw, currUndistortedRGB, K, distCoeffs);
        cv::cvtColor(currUndistortedRGB, currUndistortedGrey, cv::COLOR_BGR2GRAY);
    }

    void featureDetection(const cv::Mat &img)
    {
        currPoints.pts.clear();
        currPoints.ids.clear();

        std::vector<cv::KeyPoint> kp;
        orb->detect(img, kp);

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

    void replenishFeatures(const cv::Mat &img)
    {
        int nToDetect = ORB_N_BEST - static_cast<int>(currPoints.pts.size());
        if (nToDetect <= 0)
            return;

        cv::Mat mask(img.size(), CV_8UC1, cv::Scalar(255));
        int radius = 21;

        for (const auto &p : currPoints.pts)
        {
            if (p.x >= 0 && p.x < img.cols && p.y >= 0 && p.y < img.rows)
                cv::circle(mask, p, radius, cv::Scalar(0), -1);
        }

        std::vector<cv::KeyPoint> kpNew;
        orb->detect(img, kpNew, mask);

        std::vector<int> idx(kpNew.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b)
                  { return kpNew[a].response > kpNew[b].response; });

        int added = 0;
        for (int j : idx)
        {
            if (added >= nToDetect)
                break;

            const cv::Point2f p = kpNew[j].pt;
            int x = cvRound(p.x), y = cvRound(p.y);

            if (x < 0 || x >= img.cols || y < 0 || y >= img.rows)
                continue;

            if (mask.at<uint8_t>(y, x) == 0)
                continue;

            currPoints.pts.push_back(p);
            currPoints.ids.push_back(nextFeatureID++);
            cv::circle(mask, cv::Point(x, y), radius, cv::Scalar(0), -1);

            added++;
        }
    }

    gtsam::Pose3 relativeMovementEstimation(
        std::vector<cv::Point2f> &prevValid,
        std::vector<cv::Point2f> &currValid)
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

        Eigen::Matrix3d Rg_mat;
        Rg_mat << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
            R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
            R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);

        gtsam::Rot3 Rg(Rg_mat);

        gtsam::Point3 tg(
            t.at<double>(0),
            t.at<double>(1),
            t.at<double>(2));

        return gtsam::Pose3(Rg, tg);
    }

    double computeKFParallax(Keyframe &KF, Points &currPts)
    {
        double sum = 0.0;
        int cnt = 0;

        for (size_t i = 0; i < KF.points.ids.size(); i++)
        {
            int idKF = KF.points.ids[i];
            auto it = std::find(currPts.ids.begin(), currPts.ids.end(), idKF);
            if (it == currPts.ids.end())
                continue;

            int j = std::distance(currPts.ids.begin(), it);

            double dx = currPts.pts[j].x - KF.points.pts[i].x;
            double dy = currPts.pts[j].y - KF.points.pts[i].y;

            sum += std::sqrt(dx * dx + dy * dy);
            cnt++;
        }

        if (cnt == 0)
            return 0.0;

        return sum / cnt;
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

            Eigen::Vector3d v(
                velMsgs[i]->vector.x,
                velMsgs[i]->vector.y,
                velMsgs[i]->vector.z);

            dp += v * dt;
        }

        return dp;
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "fuims_vio");
    vioManager manager;
    return 0;
}
