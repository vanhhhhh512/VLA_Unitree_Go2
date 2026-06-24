// Copyright (c) 2024, RoboVerse community
// SPDX-License-Identifier: BSD-3-Clause

#include "lidar_processor_cpp/lidar_to_pointcloud_node.hpp"
#include <chrono>

namespace lidar_processor_cpp
{

LidarToPointCloudNode::LidarToPointCloudNode()
  : Node("lidar_to_pointcloud")
{
  declareParameters();
  config_ = loadConfiguration();
  setupSubscriptions();
  setupPublishers();

  if (config_.save_map) {
    save_timer_ = this->create_wall_timer(
      std::chrono::duration<double>(config_.save_interval),
      std::bind(&LidarToPointCloudNode::saveMapCallback, this)
    );
    RCLCPP_INFO(this->get_logger(), "Map saving enabled: %s.ply every %.0fs",
      config_.map_name.c_str(), config_.save_interval);
  }
  logConfiguration();
}

void LidarToPointCloudNode::declareParameters()
{
  this->declare_parameter("robot_ip_lst",  std::vector<std::string>{});
  this->declare_parameter("map_name",      "3d_map");
  this->declare_parameter("map_save",      "false");
  this->declare_parameter("save_interval", 30.0);
  this->declare_parameter("max_points",    500000);
  this->declare_parameter("voxel_size",    0.05);
}

LidarConfig LidarToPointCloudNode::loadConfiguration()
{
  LidarConfig config;
  config.robot_ip_list = this->get_parameter("robot_ip_lst").as_string_array();
  config.map_name      = this->get_parameter("map_name").as_string();
  config.save_map      = (this->get_parameter("map_save").as_string() == "true");
  config.save_interval = this->get_parameter("save_interval").as_double();
  config.max_points    = this->get_parameter("max_points").as_int();
  config.voxel_size    = this->get_parameter("voxel_size").as_double();
  return config;
}

void LidarToPointCloudNode::setupSubscriptions()
{
  auto qos = rclcpp::QoS(1)
    .reliability(rclcpp::ReliabilityPolicy::BestEffort)
    .history(rclcpp::HistoryPolicy::KeepLast);

  if (config_.robot_ip_list.size() <= 1) {
    subscriptions_.push_back(
      this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "robot0/point_cloud2", qos,
        std::bind(&LidarToPointCloudNode::lidarCallback, this, std::placeholders::_1)
      )
    );
  } else {
    for (size_t i = 0; i < config_.robot_ip_list.size(); ++i) {
      subscriptions_.push_back(
        this->create_subscription<sensor_msgs::msg::PointCloud2>(
          "/robot" + std::to_string(i) + "/point_cloud2", qos,
          std::bind(&LidarToPointCloudNode::lidarCallback, this, std::placeholders::_1)
        )
      );
    }
  }
}

void LidarToPointCloudNode::setupPublishers()
{
  auto qos = rclcpp::QoS(1)
    .reliability(rclcpp::ReliabilityPolicy::BestEffort)
    .history(rclcpp::HistoryPolicy::KeepLast);

  pointcloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
    "/point_cloud2", qos);
}

void LidarToPointCloudNode::lidarCallback(
  const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  try {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *cloud);

    if (cloud->points.empty()) return;

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
    if (config_.voxel_size > 0.0) {
      pcl::VoxelGrid<pcl::PointXYZ> vox;
      vox.setInputCloud(cloud);
      vox.setLeafSize(
        static_cast<float>(config_.voxel_size),
        static_cast<float>(config_.voxel_size),
        static_cast<float>(config_.voxel_size));
      vox.filter(*filtered);
    } else {
      filtered = cloud;
    }

    if (filtered->points.empty()) return;

    sensor_msgs::msg::PointCloud2 out;
    pcl::toROSMsg(*filtered, out);
    out.header = msg->header;
    pointcloud_pub_->publish(out);

    if (config_.save_map) {
      std::lock_guard<std::mutex> lock(save_mutex_);
      latest_cloud_ = filtered;
    }

  } catch (const std::exception& e) {
    RCLCPP_ERROR(this->get_logger(), "lidarCallback error: %s", e.what());
  }
}

void LidarToPointCloudNode::publishAggregatedPointcloud(
  const std_msgs::msg::Header&) {}

void LidarToPointCloudNode::saveMapCallback()
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_to_save;
  {
    std::lock_guard<std::mutex> lock(save_mutex_);
    if (!latest_cloud_ || latest_cloud_->points.empty()) return;
    cloud_to_save = latest_cloud_;
  }

  if (static_cast<int>(cloud_to_save->points.size()) > config_.max_points) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr ds(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> vox;
    vox.setInputCloud(cloud_to_save);
    float leaf = static_cast<float>(config_.voxel_size) * 2.0f;
    vox.setLeafSize(leaf, leaf, leaf);
    vox.filter(*ds);
    cloud_to_save = ds;
  }

  const std::string filename = config_.map_name + ".ply";
  if (pcl::io::savePLYFileBinary(filename, *cloud_to_save) == 0) {
    RCLCPP_INFO(this->get_logger(), "Saved map: %s (%zu points)",
      filename.c_str(), cloud_to_save->points.size());
  } else {
    RCLCPP_ERROR(this->get_logger(), "Failed to save map: %s", filename.c_str());
  }
}

void LidarToPointCloudNode::logConfiguration()
{
  RCLCPP_INFO(this->get_logger(), "LiDAR to PointCloud node ready");
  RCLCPP_INFO(this->get_logger(), "  Robots : %zu", config_.robot_ip_list.size());
  RCLCPP_INFO(this->get_logger(), "  Voxel  : %.3fm", config_.voxel_size);
  RCLCPP_INFO(this->get_logger(), "  Save   : %s", config_.save_map ? "true" : "false");
}

}  // namespace lidar_processor_cpp

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<lidar_processor_cpp::LidarToPointCloudNode>();
    rclcpp::spin(node);
  } catch (const std::exception& e) {
    std::cerr << "Fatal: " << e.what() << std::endl;
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}