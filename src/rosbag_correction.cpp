#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/NavSatFix.h>
#include <geometry_msgs/QuaternionStamped.h>
#include <geometry_msgs/Vector3Stamped.h>

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
        input.open(INPUT_BAG_PATH, rosbag::bagmode::Read);
        output.open(OUTPUT_BAG_PATH, rosbag::bagmode::Write);

        std::vector<std::string> topics = {
            CAMERA_TOPIC,
            QUATERNION_TOPIC,
            VELOCITY_TOPIC,
            GPS_TOPIC};

        rosbag::View view(input, rosbag::TopicQuery(topics));

        for (const rosbag::MessageInstance &m : view)
        {
            ros::Time t = m.getTime();

            // CAMERA (CompressedImage)
            if (m.getTopic() == CAMERA_TOPIC)
            {
                sensor_msgs::CompressedImageConstPtr img =
                    m.instantiate<sensor_msgs::CompressedImage>();

                if (img)
                    output.write(CAMERA_TOPIC, t, *img);

                continue;
            }

            // QUATERNION
            if (m.getTopic() == QUATERNION_TOPIC)
            {
                auto q = m.instantiate<geometry_msgs::QuaternionStamped>();
                if (q)
                    output.write(QUATERNION_TOPIC, t, *q);

                continue;
            }

            // VELOCITY
            if (m.getTopic() == VELOCITY_TOPIC)
            {
                auto v = m.instantiate<geometry_msgs::Vector3Stamped>();
                if (v)
                    output.write(VELOCITY_TOPIC, t, *v);

                continue;
            }

            // GPS
            if (m.getTopic() == GPS_TOPIC)
            {
                auto gps = m.instantiate<sensor_msgs::NavSatFix>();
                if (gps)
                    output.write(GPS_TOPIC, t, *gps);

                continue;
            }
        }

        input.close();
        output.close();
    }

private:
    rosbag::Bag input, output;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "rosbag_updater");
    rosbagUpdater upd;
    return 0;
}