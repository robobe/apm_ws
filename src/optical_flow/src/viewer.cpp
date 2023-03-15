#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include <image_transport/image_transport.hpp>
#include <opencv2/opencv.hpp> 
#include <cv_bridge/cv_bridge.h>

class Viewer : public rclcpp::Node
{
private:
    const std::string OPENCV_WINDOW = "Image window";
    image_transport::Subscriber sub_;

private:
    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
        }
        catch (cv_bridge::Exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        if (cv_ptr->image.rows > 60 && cv_ptr->image.cols > 60)
            cv::circle(cv_ptr->image, cv::Point(50, 50), 10, CV_RGB(255,0,0));

        cv::imshow(OPENCV_WINDOW, cv_ptr->image);
        cv::waitKey(3);
    }
 
public:
    Viewer() : Node("MyViewer")
    {
        RCLCPP_INFO(this->get_logger(), "hello Viewer node");
        rmw_qos_profile_t custom_qos = rmw_qos_profile_default;
        sub_ = image_transport::create_subscription(this, "/camera/image_raw",
                                                    std::bind(&Viewer::imageCallback, this, std::placeholders::_1), "raw", custom_qos);
    }

    
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<Viewer>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}