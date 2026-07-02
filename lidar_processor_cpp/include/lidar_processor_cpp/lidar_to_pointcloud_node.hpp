// Copyright (c) 2024, RoboVerse community
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LIDAR_PROCESSOR_CPP__LIDAR_TO_POINTCLOUD_NODE_HPP_
#define LIDAR_PROCESSOR_CPP__LIDAR_TO_POINTCLOUD_NODE_HPP_

#include <memory>
#include <vector>
#include <string>
#include <mutex>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/header.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl/io/ply_io.h"
#include "pcl/filters/voxel_grid.h"
#include "pcl_conversions/pcl_conversions.h"

namespace lidar_processor_cpp
{

struct LidarConfig
{
  std::vector<std::string> robot_ip_list;
  std::string map_name;
  bool save_map;
  double save_interval;
  int max_points;
  double voxel_size;
};

class LidarToPointCloudNode : public rclcpp::Node
{
public:
  LidarToPointCloudNode();

private:
  void declareParameters();
  LidarConfig loadConfiguration();
  void setupSubscriptions();
  void setupPublishers();
  void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void publishAggregatedPointcloud(const std_msgs::msg::Header& header);
  void saveMapCallback();
  void logConfiguration();

  LidarConfig config_;

  mutable std::mutex save_mutex_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr latest_cloud_;

  std::vector<rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr> subscriptions_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_pub_;
  rclcpp::TimerBase::SharedPtr save_timer_;
};

}  // namespace lidar_processor_cpp

#endif  // LIDAR_PROCESSOR_CPP__LIDAR_TO_POINTCLOUD_NODE_HPP_