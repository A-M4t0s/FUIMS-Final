/**
 * @file vio_state_publisher.cpp
 * @brief Simple node to publish VIO state commands via keyboard input.
 *
 * VIO States:
 *   0 = IDLE
 *   1 = RUNNING
 *   2 = RESET
 *
 * Usage:
 *   rosrun fuims vio_state_publisher
 *
 * Then press:
 *   'i' or '0' -> IDLE
 *   'r' or '1' -> RUNNING
 *   's' or '2' -> RESET (stop)
 *   'q'        -> Quit
 */

#include <ros/ros.h>
#include <std_msgs/UInt8.h>
#include <termios.h>
#include <unistd.h>
#include <iostream>

enum class vioState : uint8_t
{
  IDLE = 0,
  RUNNING = 1,
  RESET = 2
};

/**
 * @brief Get a single character from terminal without waiting for Enter.
 */
char getKey()
{
  struct termios oldt, newt;
  char ch;
  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);
  ch = getchar();
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  return ch;
}

void printMenu()
{
  std::cout << "\n========================================\n";
  std::cout << "       VIO State Publisher\n";
  std::cout << "========================================\n";
  std::cout << "  [i] or [0]  ->  IDLE\n";
  std::cout << "  [r] or [1]  ->  RUNNING\n";
  std::cout << "  [s] or [2]  ->  RESET\n";
  std::cout << "  [q]         ->  Quit\n";
  std::cout << "========================================\n";
  std::cout << "Press a key: " << std::flush;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "vio_state_publisher");
  ros::NodeHandle nh;

  ros::Publisher statePub = nh.advertise<std_msgs::UInt8>("vio/state", 1, true);

  ROS_INFO("VIO State Publisher started. Publishing to 'vio/state'");

  printMenu();

  while (ros::ok())
  {
    char key = getKey();
    std_msgs::UInt8 msg;
    bool publish = true;
    std::string stateName;

    switch (key)
    {
    case 'i':
    case 'I':
    case '0':
      msg.data = static_cast<uint8_t>(vioState::IDLE);
      stateName = "IDLE";
      break;

    case 'r':
    case 'R':
    case '1':
      msg.data = static_cast<uint8_t>(vioState::RUNNING);
      stateName = "RUNNING";
      break;

    case 's':
    case 'S':
    case '2':
      msg.data = static_cast<uint8_t>(vioState::RESET);
      stateName = "RESET";
      break;

    case 'q':
    case 'Q':
      std::cout << "\nExiting VIO State Publisher.\n";
      return 0;

    default:
      publish = false;
      std::cout << "\nInvalid key. Try again.\n";
      printMenu();
      break;
    }

    if (publish)
    {
      statePub.publish(msg);
      std::cout << "\n>> Published state: " << stateName << " (" << static_cast<int>(msg.data) << ")\n";
      printMenu();
    }

    ros::spinOnce();
  }

  return 0;
}
