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

#include <opencv2/opencv.hpp>

#define INPUT_BAG_PATH "/home/tony/catkin_ws/bags/bags_Agucadoura/bom/2025-07-03-13-03-16.bag"
#define OUTPUT_BAG_PATH "/home/tony/catkin_ws/bags/bags_fuims/agucadoura.bag"
#define CAMERA_TOPIC "/dji_m350/cameras/main/compressed"
#define QUATERNION_TOPIC "/dji_m350/quaternion"
#define VELOCITY_TOPIC "/dji_m350/velocity"
#define GPS_TOPIC "/dji_m350/gps"

class rosbagUpdater
{
public:
    rosbagUpdater()
    {
        // Opening Input and Output Bags
        input.open(INPUT_BAG_PATH, rosbag::bagmode::Read);
        output.open(OUTPUT_BAG_PATH, rosbag::bagmode::Write);

        // Topics list
        std::vector<std::string> topics = {
            CAMERA_TOPIC,
            QUATERNION_TOPIC,
            VELOCITY_TOPIC,
            GPS_TOPIC};
        rosbag::View view(input, rosbag::TopicQuery(topics)); // View of all the needed topics

        size_t frame_id = 0;

        // Input bag processing
        for (const rosbag::MessageInstance &m : view)
        {
            // Get current time
            ros::Time t = m.getTime();

            // Camera Message Processing
            if (m.getTopic() == CAMERA_TOPIC)
            {
                sensor_msgs::CompressedImageConstPtr comp = m.instantiate<sensor_msgs::CompressedImage>();
                if (!comp)
                    continue;

                // Converting CompressedImage to Image
                cv::Mat img = cv::imdecode(cv::Mat(comp->data), cv::IMREAD_COLOR);
                if (img.empty())
                    continue;

                sensor_msgs::Image raw;
                raw.header.stamp = t;
                raw.header.frame_id = comp->header.frame_id;
                raw.height = img.rows;
                raw.width = img.cols;
                raw.encoding = sensor_msgs::image_encodings::BGR8;
                raw.step = img.cols * 3;
                raw.data.assign(img.data, img.data + img.total() * img.elemSize());

                output.write("/dji_m350/cameras/main/raw", t, raw);

                frame_id++;
                continue;
            }

            // QUATERNION
            if (m.getTopic() == QUATERNION_TOPIC)
            {
                geometry_msgs::QuaternionStampedConstPtr q =
                    m.instantiate<geometry_msgs::QuaternionStamped>();

                if (q)
                    output.write(QUATERNION_TOPIC, t, *q);

                continue;
            }

            // VELOCITY
            if (m.getTopic() == VELOCITY_TOPIC)
            {
                geometry_msgs::Vector3StampedConstPtr v =
                    m.instantiate<geometry_msgs::Vector3Stamped>();

                if (v)
                    output.write(VELOCITY_TOPIC, t, *v);

                continue;
            }

            // GPS
            if (m.getTopic() == GPS_TOPIC)
            {
                sensor_msgs::NavSatFixConstPtr gps =
                    m.instantiate<sensor_msgs::NavSatFix>();

                if (gps)
                    output.write(GPS_TOPIC, t, *gps);

                continue;
            }
        }

        input.close();
        output.close();
    }

private:
    // ROSBAG Variables
    rosbag::Bag input, output;
};

// Main
int main(int argc, char **argv)
{
    ros::init(argc, argv, "rosbag_updater");
    rosbagUpdater upd;
    return 0;
}