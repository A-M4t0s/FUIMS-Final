#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <std_msgs/Float64.h>
#include <fuims/Float64Stamped.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/NavSatFix.h>
#include <geometry_msgs/QuaternionStamped.h>
#include <geometry_msgs/Vector3Stamped.h>

#include <unordered_map>

#define INPUT_BAG_PATH "/home/tony/catkin_ws/bags/bags_Agucadoura/bom/2025-07-03-13-03-16.bag"
#define OUTPUT_BAG_PATH "/home/tony/catkin_ws/bags/bags_fuims/agucadoura.bag"

#define CAMERA_TOPIC "/dji_m350/cameras/main/compressed"
#define QUATERNION_TOPIC "/dji_m350/quaternion"
#define VELOCITY_TOPIC "/dji_m350/velocity"
#define GPS_TOPIC "/dji_m350/gps"
#define ALTITUDE_TOPIC "/dji_m350/altitude"

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
            ALTITUDE_TOPIC,
            GPS_TOPIC};

        rosbag::View view(input, rosbag::TopicQuery(topics));

        std::unordered_map<std::string, uint32_t> topic_seq;

        for (const rosbag::MessageInstance &m : view)
        {
            ros::Time t = m.getTime();
            const std::string &topic = m.getTopic();

            uint32_t &seq = topic_seq[topic]; // auto-inits to 0

            // ===============================
            // CAMERA
            // ===============================
            if (topic == CAMERA_TOPIC)
            {
                auto img = m.instantiate<sensor_msgs::CompressedImage>();
                if (!img)
                    continue;

                sensor_msgs::CompressedImage out = *img;
                out.header.seq = ++seq;
                out.header.stamp = t;
                out.header.frame_id = "dji_m350";

                output.write(CAMERA_TOPIC, t, out);
            }

            // ===============================
            // QUATERNION
            // ===============================
            else if (topic == QUATERNION_TOPIC)
            {
                auto q = m.instantiate<geometry_msgs::QuaternionStamped>();
                if (!q)
                    continue;

                geometry_msgs::QuaternionStamped out = *q;
                out.header.seq = ++seq;
                out.header.stamp = t;
                out.header.frame_id = "dji_m350";

                output.write(QUATERNION_TOPIC, t, out);
            }

            // ===============================
            // VELOCITY
            // ===============================
            else if (topic == VELOCITY_TOPIC)
            {
                auto v = m.instantiate<geometry_msgs::Vector3Stamped>();
                if (!v)
                    continue;

                geometry_msgs::Vector3Stamped out = *v;
                out.header.seq = ++seq;
                out.header.stamp = t;
                out.header.frame_id = "dji_m350";

                output.write(VELOCITY_TOPIC, t, out);
            }

            // ===============================
            // ALTITUDE
            // ===============================
            else if (topic == ALTITUDE_TOPIC)
            {
                auto alt = m.instantiate<std_msgs::Float64>();
                if (!alt)
                    continue;

                fuims::Float64Stamped out;
                out.header.seq = ++seq;
                out.header.stamp = t;
                out.header.frame_id = "dji_m350";
                out.data = alt->data;

                output.write(ALTITUDE_TOPIC, t, out);
            }

            // ===============================
            // GPS
            // ===============================
            else if (topic == GPS_TOPIC)
            {
                auto gps = m.instantiate<sensor_msgs::NavSatFix>();
                if (!gps)
                    continue;

                sensor_msgs::NavSatFix out = *gps;
                out.header.seq = ++seq;
                out.header.stamp = t;
                out.header.frame_id = "dji_m350";

                output.write(GPS_TOPIC, t, out);
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
