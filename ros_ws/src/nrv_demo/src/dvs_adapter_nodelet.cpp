#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include <camera_info_manager/camera_info_manager.h>
#include <dvs_msgs/EventArray.h>
#include <event_camera_codecs/decoder_factory.h>
#include <event_camera_codecs/event_processor.h>
#include <event_camera_msgs/EventPacket.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>


namespace nrv_demo {

class SequenceTracker {
 public:
  void observe(uint64_t sequence) {
    if (have_previous_ && sequence != previous_ + 1U) {
      ++gap_incidents_;
      if (sequence > previous_ + 1U) {
        missing_packets_ += sequence - previous_ - 1U;
      } else {
        ++reordered_or_reset_;
      }
    }
    previous_ = sequence;
    have_previous_ = true;
  }

  uint64_t gapIncidents() const { return gap_incidents_; }
  uint64_t missingPackets() const { return missing_packets_; }
  uint64_t reorderedOrReset() const { return reordered_or_reset_; }

 private:
  bool have_previous_{false};
  uint64_t previous_{0};
  uint64_t gap_incidents_{0};
  uint64_t missing_packets_{0};
  uint64_t reordered_or_reset_{0};
};



class DvsAdapterNodelet;

class AdapterProcessor : public event_camera_codecs::EventProcessor {
 public:
  explicit AdapterProcessor(DvsAdapterNodelet *owner) : owner_(owner) {}
  void eventCD(uint64_t sensor_time, uint16_t x, uint16_t y, uint8_t polarity) override;
  bool eventExtTrigger(uint64_t, uint8_t, uint8_t) override { return true; }
  void finished() override {}
  void rawData(const char *, size_t) override {}

 private:
  DvsAdapterNodelet *owner_;
};

class DvsAdapterNodelet : public nodelet::Nodelet {
 public:
  void append(uint64_t sensor_time, uint16_t x, uint16_t y, uint8_t polarity) {
    if (!have_anchor_) {
      sensor_anchor_ns_ = sensor_time;
      ros_anchor_ns_ = packet_stamp_.isZero() ? ros::Time::now().toNSec()
                                               : packet_stamp_.toNSec();
      have_anchor_ = true;
    }
    if (sensor_time < sensor_anchor_ns_) {
      ++time_reversals_;
      return;
    }
    dvs_msgs::Event event;
    event.x = x;
    event.y = y;
    event.polarity = polarity != 0U;
    event.ts.fromNSec(ros_anchor_ns_ + sensor_time - sensor_anchor_ns_);
    message_.events.push_back(event);
  }

 private:
  friend class AdapterProcessor;

  void onInit() override {
    ros::NodeHandle nh = getNodeHandle();
    ros::NodeHandle private_nh = getPrivateNodeHandle();
    private_nh.param<std::string>("input_topic", input_topic_, "/delta_driver/events");
    private_nh.param<std::string>("output_topic", output_topic_, "events");
    private_nh.param<std::string>("camera_info_topic", camera_info_topic_, "camera_info");
    private_nh.param<std::string>("frame_id", frame_id_, "delta_camera_optical_frame");
    private_nh.param<std::string>("camera_name", camera_name_, "nrv_delta01");
    private_nh.param<std::string>("camera_info_url", camera_info_url_, "");
    private_nh.param("subscriber_queue_size", subscriber_queue_size_, 100);
    private_nh.param("event_reserve", event_reserve_, 1048576);
    private_nh.param("camera_info_rate", camera_info_rate_, 5.0);
    if (event_reserve_ < 1 || subscriber_queue_size_ < 1) {
      throw std::runtime_error("adapter queue and reserve values must be positive");
    }

    message_.events.reserve(static_cast<std::size_t>(event_reserve_));
    publisher_ = nh.advertise<dvs_msgs::EventArray>(output_topic_, 100);
    camera_info_publisher_ = nh.advertise<sensor_msgs::CameraInfo>(camera_info_topic_, 1, true);
    camera_info_manager_.reset(
        new camera_info_manager::CameraInfoManager(nh, camera_name_, camera_info_url_));
    subscriber_ = nh.subscribe(input_topic_, static_cast<uint32_t>(subscriber_queue_size_),
                               &DvsAdapterNodelet::packetCallback, this);
    camera_info_timer_ = nh.createWallTimer(
        ros::WallDuration(1.0 / camera_info_rate_), &DvsAdapterNodelet::publishCameraInfo, this);
    NODELET_INFO("optional adapter: %s -> %s", input_topic_.c_str(),
                 nh.resolveName(output_topic_).c_str());
  }

  void packetCallback(const event_camera_msgs::EventPacket::ConstPtr &packet) {
    sequence_.observe(packet->seq);
    packet_stamp_ = packet->header.stamp;
    message_.events.clear();
    message_.height = static_cast<uint16_t>(packet->height);
    message_.width = static_cast<uint16_t>(packet->width);
    message_.header.frame_id = frame_id_;

    auto *decoder = factory_.getInstance(*packet);
    if (!decoder) {
      NODELET_ERROR_THROTTLE(5.0, "no decoder for EventPacket encoding '%s'",
                             packet->encoding.c_str());
      return;
    }
    decoder->setTimeMultiplier(1000U);
    while (decoder->decode(*packet, &processor_)) {
    }
    message_.header.stamp = message_.events.empty() ? packet_stamp_ : message_.events.back().ts;
    publisher_.publish(message_);
    width_ = packet->width;
    height_ = packet->height;
    ++packets_;
    events_ += message_.events.size();
    if (sequence_.gapIncidents() > reported_gaps_) {
      reported_gaps_ = sequence_.gapIncidents();
      NODELET_ERROR("adapter detected EventPacket sequence gap; total incidents=%llu",
                    static_cast<unsigned long long>(reported_gaps_));
    }
  }

  void publishCameraInfo(const ros::WallTimerEvent &) {
    if (width_ == 0U || height_ == 0U) {
      return;
    }
    sensor_msgs::CameraInfo info = camera_info_manager_->getCameraInfo();
    info.header.stamp = ros::Time::now();
    info.header.frame_id = frame_id_;
    if (info.width == 0U || info.height == 0U) {
      info.width = width_;
      info.height = height_;
    }
    camera_info_publisher_.publish(info);
  }

  std::string input_topic_;
  std::string output_topic_;
  std::string camera_info_topic_;
  std::string frame_id_;
  std::string camera_name_;
  std::string camera_info_url_;
  int subscriber_queue_size_{100};
  int event_reserve_{1048576};
  double camera_info_rate_{5.0};
  ros::Subscriber subscriber_;
  ros::Publisher publisher_;
  ros::Publisher camera_info_publisher_;
  ros::WallTimer camera_info_timer_;
  std::unique_ptr<camera_info_manager::CameraInfoManager> camera_info_manager_;
  event_camera_codecs::DecoderFactory<event_camera_msgs::EventPacket, AdapterProcessor> factory_;
  AdapterProcessor processor_{this};
  dvs_msgs::EventArray message_;
  SequenceTracker sequence_;
  ros::Time packet_stamp_;
  uint64_t sensor_anchor_ns_{0};
  uint64_t ros_anchor_ns_{0};
  uint64_t packets_{0};
  uint64_t events_{0};
  uint64_t time_reversals_{0};
  uint64_t reported_gaps_{0};
  uint32_t width_{0};
  uint32_t height_{0};
  bool have_anchor_{false};
};

void AdapterProcessor::eventCD(uint64_t sensor_time, uint16_t x, uint16_t y,
                               uint8_t polarity) {
  owner_->append(sensor_time, x, y, polarity);
}

}  // namespace nrv_demo

PLUGINLIB_EXPORT_CLASS(nrv_demo::DvsAdapterNodelet, nodelet::Nodelet)
