#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/NavSatFix.h>

#include <cmath>
#include <algorithm>
#include <numeric>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

// ROSBAG Defines
#define BAG_PATH "/home/tony/catkin_ws/bags/bags_Agucadoura/bom/2025-07-03-13-03-16.bag"
#define CAMERA_TOPIC "/dji_m350/cameras/main/compressed"
#define QUATERNION_TOPIC "/dji_m350/quaternion"
#define GPS_TOPIC "/dji_m350/gps"

// WGS84 Constants
constexpr double a = 6378137.0;
constexpr double f = 1.0 / 298.257223563;
constexpr double b = a * (1 - f);
constexpr double e2 = 1 - (b * b) / (a * a);

/*
 * =========================================================================
 *   Data Structures
 *      - ENU Structure
 * =========================================================================
 */
struct ENU
{
    double x, y, z;
};

/*
 * =========================================================================
 *   Geodetic <-> ECEF Convertions
 * =========================================================================
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
 * =========================================================================
 *   Class: VIO-Manager
 * =========================================================================
 */

class vioManager
{
public:
    vioManager()
    {
        // Opening ROSBAG
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

        // Creating ORB Detector (maxFeatures = 1000 | score = Harris)
        // orb = cv::ORB::create(1000);

        // nFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma
        sift = cv::SIFT::create();
        // sift = cv::SIFT::create(4000, 3, 0.03, 10, 1.6);

        // ROSBAG Processing
        processBag();

        // Feature Detection
        featureDetection();

        // Feature Matching
        featureMatching();
    }

private:
    // ROSBAG Variable
    rosbag::Bag bag;

    // OpenCV Variables
    // cv::Ptr<cv::ORB> orb; // ORB Features Detection
    cv::Ptr<cv::SIFT> sift; // SIFT Features Detection

    // Camera Params
    const cv::Mat K = (cv::Mat_<double>(3, 3) << 1372.76165, 0.0, 960.45289,
                       0.0, 1372.14817, 515.00383,
                       0.0, 0.0, 1.0);

    const cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << 0.048300,
                                -0.044905,
                                -0.003731,
                                -0.001349,
                                0.000000);

    // Images and ENU Coordinates from GPS
    std::vector<cv::Mat> undistortedRGB;
    std::vector<cv::Mat> undistortedGray;
    std::vector<ENU> gpsENU;

    // Detector-Related Variables
    std::vector<std::vector<cv::KeyPoint>> keypoints;
    std::vector<cv::Mat> descriptors;

    // Undistorting an image
    cv::Mat undistortImage(const cv::Mat &image)
    {
        cv::Mat undistorted;
        cv::undistort(image, undistorted, K, distCoeffs);
        return undistorted;
    }

    // Bag Processing
    void processBag()
    {
        ros::WallTime start_time = ros::WallTime::now();
        ROS_INFO("Starting processBag()");

        // Bag views -> Camera & GPS
        rosbag::View cam_view(bag, rosbag::TopicQuery(CAMERA_TOPIC));
        rosbag::View gps_view(bag, rosbag::TopicQuery(GPS_TOPIC));
        uint32_t nImages = cam_view.size();
        uint32_t nGPS = gps_view.size();

        // Reading GPS
        std::vector<sensor_msgs::NavSatFixConstPtr> gpsMsgs;
        gpsMsgs.reserve(nGPS);

        for (const rosbag::MessageInstance &m : gps_view)
        {
            auto msg = m.instantiate<sensor_msgs::NavSatFix>();
            if (msg)
                gpsMsgs.push_back(msg);
        }

        if (gpsMsgs.empty())
        {
            ROS_ERROR("No GPS messages found!");
            return;
        }

        // GPS -> ENU
        double refLat = gpsMsgs[0]->latitude * M_PI / 180.0;
        double refLon = gpsMsgs[0]->longitude * M_PI / 180.0;
        double refAlt = gpsMsgs[0]->altitude;

        gpsENU.reserve(gpsMsgs.size());

        for (auto &g : gpsMsgs)
        {
            double lat = g->latitude * M_PI / 180.0;
            double lon = g->longitude * M_PI / 180.0;
            double alt = g->altitude;

            double X, Y, Z;
            geodeticToECEF(lat, lon, alt, X, Y, Z);
            gpsENU.push_back(ecefToENU(X, Y, Z, refLat, refLon, refAlt));
        }

        // Normalize origin
        ENU p0 = gpsENU[0];
        for (auto &p : gpsENU)
        {
            p.x -= p0.x;
            p.y -= p0.y;
            p.z -= p0.z;
        }

        // Reading & Processing Images
        uint32_t idx = 0;
        undistortedRGB.reserve(nImages / 5 + 1);
        undistortedGray.reserve(nImages / 5 + 1);

        for (const rosbag::MessageInstance &m : cam_view)
        {
            auto msg = m.instantiate<sensor_msgs::CompressedImage>();
            if (!msg)
                continue;

            // Skip frames: process only every 3rd image
            if (idx % 5 != 0)
            {
                ++idx;
                continue;
            }

            cv::Mat raw = cv::imdecode(msg->data, cv::IMREAD_COLOR);
            cv::Mat rgb;
            cv::cvtColor(raw, rgb, cv::COLOR_BGR2RGB);
            if (rgb.empty())
            {
                ++idx;
                continue;
            }

            cv::Mat undist = undistortImage(rgb);
            undistortedRGB.push_back(undist);

            cv::Mat gray;
            cv::cvtColor(undist, gray, cv::COLOR_BGR2GRAY);
            undistortedGray.push_back(gray);

            // ROS_INFO("Processed frame %u / %u (step=3)", idx + 1, nImages);
            ++idx;
        }

        ros::WallTime finish_time = ros::WallTime::now();
        ros::WallDuration dt = finish_time - start_time;
        ROS_INFO("Finished bagProcessing() in %.4f seconds", dt.toSec());
    }

    void featureDetection()
    {
        ros::WallTime start_time = ros::WallTime::now();
        ROS_INFO("Starting featureDetection()");

        keypoints.clear();
        descriptors.clear();

        const int nBest = 300; // nº máximo de features por imagem

        std::vector<cv::KeyPoint> kp;
        cv::Mat desc;

        for (int i = 0; i < static_cast<int>(undistortedGray.size()); i++)
        {
            kp.clear();
            desc.release();

            // Detetar SIFT na imagem cinzenta
            sift->detectAndCompute(undistortedGray[i], cv::noArray(), kp, desc);

            // Se não houver keypoints/descs, salta
            if (kp.empty() || desc.empty())
            {
                keypoints.push_back(std::vector<cv::KeyPoint>()); // placeholder
                descriptors.push_back(cv::Mat());
                continue;
            }

            // Selecionar explicitamente as 300 MAIS FORTES (maior response)
            if (static_cast<int>(kp.size()) > nBest)
            {
                // índices 0..N-1
                std::vector<int> indices(kp.size());
                std::iota(indices.begin(), indices.end(), 0);

                // ordenar indices por response DESC
                std::sort(indices.begin(), indices.end(),
                          [&](int a, int b)
                          {
                              return kp[a].response > kp[b].response;
                          });

                int keep = std::min(nBest, static_cast<int>(kp.size()));

                std::vector<cv::KeyPoint> kp_best;
                kp_best.reserve(keep);

                cv::Mat desc_best(keep, desc.cols, desc.type());

                for (int k = 0; k < keep; ++k)
                {
                    int idx = indices[k];
                    kp_best.push_back(kp[idx]);
                    desc.row(idx).copyTo(desc_best.row(k));
                }

                kp.swap(kp_best);
                desc = desc_best; // agora só 300 linhas
            }

            keypoints.push_back(kp);
            descriptors.push_back(desc);
        }

        ros::WallTime finish_time = ros::WallTime::now();
        ros::WallDuration dt = finish_time - start_time;
        ROS_INFO("Finished featureDetection() in %.4f seconds", dt.toSec());
    }

    void featureMatching()
    {
        ros::WallTime start_time = ros::WallTime::now();
        ROS_INFO("Starting featureMatching()");

        const int minInliersRequired = 8;
        const double minInlierRatioThreshold = 0.05;

        // Output directory
        const std::string outDir = "/home/tony/Desktop/MEEC-SA/2º Ano/FUIMS/Debug Images/";

        // Log file (append mode)
        std::ofstream logFile(outDir + "matching_log.txt", std::ios::app);
        if (!logFile.is_open())
        {
            ROS_ERROR("Failed to open log file for writing!");
            return;
        }

        cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_L2);

        for (int i = 1; i < descriptors.size(); i++)
        {
            int prevFrame = i - 1;
            int currFrame = i;

            // Log & ROS output
            ROS_INFO("Frame %d → %d", prevFrame, currFrame);
            logFile << "Frame " << prevFrame << " → " << currFrame << "\n";

            cv::Mat desc1 = descriptors[prevFrame];
            cv::Mat desc2 = descriptors[currFrame];

            if (desc1.empty() || desc2.empty())
            {
                ROS_WARN("  → SKIPPED (empty descriptors)");
                logFile << "  → SKIPPED (empty descriptors)\n\n";
                continue; // aqui não há matches para desenhar
            }

            // ============================
            // 1. Forward matching
            // ============================
            std::vector<std::vector<cv::DMatch>> fwd_knn;
            matcher->knnMatch(desc1, desc2, fwd_knn, 2);

            std::vector<cv::DMatch> fwd;
            for (auto &m : fwd_knn)
            {
                if (m.size() >= 2 && m[0].distance < 0.7f * m[1].distance)
                    fwd.push_back(m[0]);
            }

            // ============================
            // 2. Backward matching
            // ============================
            std::vector<std::vector<cv::DMatch>> bwd_knn;
            matcher->knnMatch(desc2, desc1, bwd_knn, 2);

            std::vector<cv::DMatch> bwd;
            for (auto &m : bwd_knn)
            {
                if (m.size() >= 2 && m[0].distance < 0.7f * m[1].distance)
                    bwd.push_back(m[0]);
            }

            // ============================
            // 3. Mutual matching
            // ============================
            std::vector<cv::DMatch> mutual;
            for (auto &m_f : fwd)
            {
                for (auto &m_b : bwd)
                {
                    if (m_f.queryIdx == m_b.trainIdx && m_f.trainIdx == m_b.queryIdx)
                    {
                        mutual.push_back(m_f);
                    }
                }
            }

            int numMutual = mutual.size();
            ROS_INFO("  - Mutual Matches: %d", numMutual);
            logFile << "  - Mutual Matches: " << numMutual << "\n";

            // Se não houver mutual, ainda assim vamos desenhar imagens vazias
            if (numMutual == 0)
            {
                ROS_WARN("  → No mutual matches");
                logFile << "  → No mutual matches\n";
            }

            // ============================
            // 4. Preparar pontos para RANSAC (se possível)
            // ============================
            std::vector<cv::DMatch> matchesToDraw = mutual; // por defeito
            bool goodGeom = false;

            if (numMutual >= minInliersRequired)
            {
                std::vector<cv::Point2f> pts1, pts2;
                pts1.reserve(numMutual);
                pts2.reserve(numMutual);

                for (auto &m : mutual)
                {
                    pts1.push_back(keypoints[prevFrame][m.queryIdx].pt);
                    pts2.push_back(keypoints[currFrame][m.trainIdx].pt);
                }

                // ============================
                // 5. Fundamental Matrix (RANSAC)
                // ============================
                std::vector<uchar> inlierMask;
                cv::Mat F = cv::findFundamentalMat(
                    pts1, pts2, cv::FM_RANSAC, 1.0, 0.995, inlierMask);

                if (!F.empty() && !inlierMask.empty())
                {
                    int numInliers = 0;
                    for (auto f : inlierMask)
                        if (f)
                            numInliers++;

                    double inlierRatio = double(numInliers) / double(numMutual);

                    ROS_INFO("  - Inliers: %d / %d (%.2f%%)",
                             numInliers, numMutual, 100 * inlierRatio);
                    logFile << "  - Inliers: " << numInliers << " / " << numMutual
                            << " (" << 100 * inlierRatio << "%)\n";

                    if (numInliers >= minInliersRequired &&
                        inlierRatio >= minInlierRatioThreshold)
                    {
                        // usa só inliers
                        std::vector<cv::DMatch> inlierMatches;
                        inlierMatches.reserve(numInliers);
                        for (int k = 0; k < mutual.size(); k++)
                            if (inlierMask[k])
                                inlierMatches.push_back(mutual[k]);

                        matchesToDraw = inlierMatches;
                        goodGeom = true;
                    }
                    else
                    {
                        ROS_WARN("  → bad inlier quality, using mutual matches only for debug");
                        logFile << "  → bad inlier quality, using mutual matches only for debug\n";
                    }
                }
                else
                {
                    ROS_WARN("  → RANSAC failed, using mutual matches only for debug");
                    logFile << "  → RANSAC failed, using mutual matches only for debug\n";
                }
            }
            else
            {
                ROS_WARN("  → not enough mutual matches for RANSAC, using mutual for debug");
                logFile << "  → not enough mutual matches for RANSAC, using mutual for debug\n";
            }

            // ============================
            // 6. Desenho manual (cores distintas)
            // ============================
            cv::Mat img1 = undistortedRGB[prevFrame].clone();
            cv::Mat img2 = undistortedRGB[currFrame].clone();

            if (img1.empty() || img2.empty())
            {
                ROS_WARN("  → Empty RGB images for drawing");
                logFile << "  → Empty RGB images for drawing\n\n";
                continue;
            }

            // Cores: BGR
            cv::Scalar colorPts1(0, 255, 0); // verde
            cv::Scalar colorPts2(0, 0, 255); // vermelho
            cv::Scalar colorLine(255, 0, 0); // azul

            int radius = 3;
            int thicknessPt = 2;
            int thicknessLine = 1;

            for (const auto &m : matchesToDraw)
            {
                cv::Point2f p1 = keypoints[prevFrame][m.queryIdx].pt;
                cv::Point2f p2 = keypoints[currFrame][m.trainIdx].pt;

                cv::circle(img1, p1, radius, colorPts1, thicknessPt, cv::LINE_AA);
                cv::circle(img2, p2, radius, colorPts2, thicknessPt, cv::LINE_AA);
            }

            // Concatenar lado a lado
            cv::Mat img_matches;
            cv::hconcat(img1, img2, img_matches);

            int offsetX = img1.cols; // deslocamento da imagem da direita

            for (const auto &m : matchesToDraw)
            {
                cv::Point2f p1 = keypoints[prevFrame][m.queryIdx].pt;
                cv::Point2f p2 = keypoints[currFrame][m.trainIdx].pt;
                cv::Point2f p2_shifted(p2.x + offsetX, p2.y);

                cv::line(img_matches, p1, p2_shifted, colorLine, thicknessLine, cv::LINE_AA);
            }

            float scale = 0.4f;
            cv::resize(img_matches, img_matches, cv::Size(), scale, scale);

            // ============================
            // 7. Save every 25 loop indices
            // ============================
            if (i % 25 == 0)
            {
                std::ostringstream oss;
                oss << outDir << "frame_"
                    << std::setw(5) << std::setfill('0') << i << ".png";

                std::string filename = oss.str();

                if (cv::imwrite(filename, img_matches))
                {
                    ROS_INFO("Saved debug image: %s", filename.c_str());
                    logFile << "Saved debug image: " << filename << "\n";
                }
                else
                {
                    ROS_WARN("Failed to write debug image: %s", filename.c_str());
                    logFile << "Failed to write debug image: " << filename << "\n";
                }
            }

            cv::imshow("Inlier / Mutual Matches", img_matches);
            cv::waitKey(1);

            logFile << "\n"; // spacing
        }

        logFile.close();

        ros::WallTime finish_time = ros::WallTime::now();
        ros::WallDuration dt = finish_time - start_time;
        ROS_INFO("Finished featureMatching() in %.4f seconds", dt.toSec());
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "fuims_vio");
    vioManager manager;
    return 0;
}