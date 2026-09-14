#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nrv_demo/performance.hpp>
#include <camera_info_manager/camera_info_manager.h>
#include <dvs_msgs/EventArray.h>
#include <event_camera_codecs/decoder_factory.h>
#include <event_camera_codecs/event_processor.h>
#include <event_camera_msgs/EventPacket.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>


namespace nrv_demo {

using AdapterPerformance = PerformanceRecorder;

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
    if (have_previous_sensor_time_ && sensor_time < previous_sensor_time_ns_) {
      ++time_reversals_;
    }
    previous_sensor_time_ns_ = sensor_time;
    have_previous_sensor_time_ = true;
    // Keep the original epoch anchor, including when the sensor clock resets.
    // Subtract in the correct direction: unsigned wrap would conceal the reset.
    uint64_t mapped_ns;
    if (sensor_time >= sensor_anchor_ns_) {
      const uint64_t offset = sensor_time - sensor_anchor_ns_;
      if (offset > UINT64_MAX - ros_anchor_ns_)
        throw std::overflow_error("sensor-to-ROS timestamp exceeds uint64 nanoseconds");
      mapped_ns = ros_anchor_ns_ + offset;
    } else {
      const uint64_t offset = sensor_anchor_ns_ - sensor_time;
      if (offset > ros_anchor_ns_)
        throw std::underflow_error("sensor-to-ROS timestamp would precede the ROS epoch");
      mapped_ns = ros_anchor_ns_ - offset;
    }
    if (source_width_ == 0U || source_height_ == 0U) {
      throw std::runtime_error("source event dimensions are unavailable");
    }
    const uint16_t scaled_x = static_cast<uint16_t>(std::min<uint32_t>(
        processing_width_ - 1U, static_cast<uint32_t>(x) * processing_width_ / source_width_));
    const uint16_t scaled_y = static_cast<uint16_t>(std::min<uint32_t>(
        processing_height_ - 1U, static_cast<uint32_t>(y) * processing_height_ / source_height_));
    dvs_msgs::Event event;
    event.x = scaled_x;
    event.y = scaled_y;
    event.polarity = polarity != 0U;
    event.ts.fromNSec(mapped_ns);
    message_.events.push_back(event);
    preview_[static_cast<std::size_t>(scaled_y) * processing_width_ + scaled_x] =
        event.polarity ? 255U : 0U;
  }

 private:
  friend class AdapterProcessor;

  void onInit() override {
    ros::NodeHandle nh = getNodeHandle();
    ros::NodeHandle private_nh = getPrivateNodeHandle();
    private_nh.param<std::string>("input_topic", input_topic_, "/delta_driver/events");
    private_nh.param<std::string>("output_topic", output_topic_, "events");
    private_nh.param<std::string>("camera_info_topic", camera_info_topic_, "camera_info");
    private_nh.param<std::string>("image_topic", image_topic_, "/delta_renderer/image");
    private_nh.param<std::string>("frame_id", frame_id_, "delta_camera_optical_frame");
    private_nh.param<std::string>("camera_name", camera_name_, "nrv_delta01");
    private_nh.param<std::string>("camera_info_url", camera_info_url_, "");
    private_nh.param("subscriber_queue_size", subscriber_queue_size_, 100);
    private_nh.param("event_reserve", event_reserve_, 1048576);
    private_nh.param("camera_info_rate", camera_info_rate_, 5.0);
    private_nh.param("image_fps", image_fps_, 10.0);
    int processing_width = 960, processing_height = 720;
    private_nh.param("processing_width", processing_width, 960);
    private_nh.param("processing_height", processing_height, 720);
    const std::vector<std::pair<int, int>> supported{{960, 720}, {640, 480}, {384, 288}};
    if (std::find(supported.begin(), supported.end(),
                  std::make_pair(processing_width, processing_height)) == supported.end()) {
      throw std::runtime_error("unsupported processing resolution");
    }
    processing_width_ = static_cast<uint32_t>(processing_width);
    processing_height_ = static_cast<uint32_t>(processing_height);
    if (event_reserve_ < 1 || subscriber_queue_size_ < 1 ||
        !std::isfinite(image_fps_) || image_fps_ <= 0) {
      throw std::runtime_error("adapter queue, reserve and image rate must be positive");
    }
    bool perf_enabled = false;
    private_nh.param("perf_enabled", perf_enabled, false);
    if (perf_enabled) {
      double interval = 5.0;
      int capacity = 4096;
      std::string output_dir;
      private_nh.param("perf_interval_s", interval, 5.0);
      private_nh.param("perf_capacity", capacity, 4096);
      private_nh.param<std::string>("perf_output_dir", output_dir, "/output");
      if (!std::isfinite(interval) || interval <= 0 || capacity < 1)
        throw std::runtime_error("performance interval and capacity must be positive");
      perf_.reset(new AdapterPerformance(output_dir + "/performance_adapter.jsonl", "adapter", interval, capacity));
    }

    message_.events.reserve(static_cast<std::size_t>(event_reserve_));
    publisher_ = nh.advertise<dvs_msgs::EventArray>(output_topic_, 100);
    camera_info_publisher_ = nh.advertise<sensor_msgs::CameraInfo>(camera_info_topic_, 1, true);
    image_publisher_ = nh.advertise<sensor_msgs::Image>(image_topic_, 1);
    preview_.assign(static_cast<std::size_t>(processing_width_) * processing_height_, 127U);
    image_period_ns_ = static_cast<uint64_t>(1e9 / image_fps_);
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
    const auto tick = perf_ ? AdapterPerformance::Clock::now() : AdapterPerformance::Tick{};
    double packet_age_ms = 0;
    uint64_t callback_ros_ns = 0;
    if (perf_) {
      const auto callback_ros_time = ros::Time::now();
      callback_ros_ns = callback_ros_time.toNSec();
      if (!packet->header.stamp.isZero())
        packet_age_ms = (callback_ros_time - packet->header.stamp).toSec() * 1000;
    }
    sequence_.observe(packet->seq);
    packet_stamp_ = packet->header.stamp;
    source_width_ = packet->width;
    source_height_ = packet->height;
    if (source_width_ == 0U || source_height_ == 0U || processing_width_ > source_width_ ||
        processing_height_ > source_height_) {
      NODELET_ERROR_THROTTLE(5.0, "processing resolution %ux%u is invalid for source %ux%u",
                             processing_width_, processing_height_, source_width_, source_height_);
      return;
    }
    message_.events.clear();
    message_.height = static_cast<uint16_t>(processing_height_);
    message_.width = static_cast<uint16_t>(processing_width_);
    message_.header.frame_id = frame_id_;

    auto *decoder = factory_.getInstance(*packet);
    if (!decoder) {
      NODELET_ERROR_THROTTLE(5.0, "no decoder for EventPacket encoding '%s'",
                             packet->encoding.c_str());
      return;
    }
    decoder->setTimeMultiplier(1000U);
    const auto decode_tick = perf_ ? AdapterPerformance::Clock::now() : AdapterPerformance::Tick{};
    while (decoder->decode(*packet, &processor_)) {
    }
    const auto decoded = perf_ ? AdapterPerformance::Clock::now() : AdapterPerformance::Tick{};
    AdapterPerformance::Tick publish_tick{};
    if (!message_.events.empty()) {
      message_.header.stamp = message_.events.back().ts;
      if (perf_) publish_tick = AdapterPerformance::Clock::now();
      publisher_.publish(message_);
    }
    const uint64_t now_ns = ros::WallTime::now().toNSec();
    if (!last_image_publish_ns_ || now_ns - last_image_publish_ns_ >= image_period_ns_) {
      sensor_msgs::Image image;
      image.header = message_.header;
      image.height = processing_height_;
      image.width = processing_width_;
      image.encoding = "mono8";
      image.is_bigendian = false;
      image.step = processing_width_;
      image.data = preview_;
      image_publisher_.publish(image);
      std::fill(preview_.begin(), preview_.end(), 127U);
      last_image_publish_ns_ = now_ns;
    }
    const auto published = perf_ ? AdapterPerformance::Clock::now() : AdapterPerformance::Tick{};
    width_ = processing_width_;
    height_ = processing_height_;
    ++packets_;
    events_ += message_.events.size();
    if (sequence_.gapIncidents() > reported_gaps_) {
      reported_gaps_ = sequence_.gapIncidents();
      NODELET_ERROR("adapter detected EventPacket sequence gap; total incidents=%llu",
                    static_cast<unsigned long long>(reported_gaps_));
    }
    if (perf_) {
      const auto completed = AdapterPerformance::Clock::now();
      const auto count = message_.events.size();
      perf_->duration("callback", tick, completed, count, packet->seq);
      perf_->duration("decode", decode_tick, decoded, count, packet->seq);
      if (count) perf_->duration("publish", publish_tick, published, count, packet->seq);
      if (!packet->header.stamp.isZero())
        perf_->observe("raw_packet_age", packet_age_ms, count, packet->seq);
      perf_->observe("raw_gap_incidents_cumulative", sequence_.gapIncidents(), 0, packet->seq, "count");
      perf_->observe("raw_missing_packets_cumulative", sequence_.missingPackets(), 0, packet->seq, "count");
      perf_->observe("raw_reordered_or_reset_cumulative", sequence_.reorderedOrReset(), 0, packet->seq, "count");
      perf_->observe("time_reversals_cumulative", time_reversals_, 0, packet->seq, "count");
      const auto callback_ns = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(tick.time_since_epoch()).count());
      // One paired packet per second is sufficient to compare clock progression.
      // All allocation and additional metadata work stay behind the PERF switch.
      if (!last_timestamp_pair_ns_ || callback_ns - last_timestamp_pair_ns_ >= 1000000000ULL) {
        AdapterPerformance::Metadata metadata{
            {"raw_sequence", packet->seq}, {"raw_header_stamp_ns", packet->header.stamp.toNSec()},
            {"callback_monotonic_ns", callback_ns}, {"callback_ros_ns", callback_ros_ns},
            {"event_count", count}, {"sensor_anchor_ns", sensor_anchor_ns_},
            {"ros_anchor_ns", ros_anchor_ns_}};
        if (count) {
          const auto first_ns = message_.events.front().ts.toNSec();
          const auto last_ns = message_.events.back().ts.toNSec();
          // Invert the existing fixed anchor only for diagnostics; event values are untouched.
          const auto sensor_ns = [this](uint64_t mapped_ns) {
            return mapped_ns >= ros_anchor_ns_ ? sensor_anchor_ns_ + (mapped_ns - ros_anchor_ns_)
                                              : sensor_anchor_ns_ - (ros_anchor_ns_ - mapped_ns);
          };
          metadata.insert({{"event_first_ns", first_ns}, {"event_last_ns", last_ns},
                           {"sensor_first_ns", sensor_ns(first_ns)},
                           {"sensor_last_ns", sensor_ns(last_ns)}});
        }
        perf_->observe("timestamp_pair", 1, count, packet->seq, "packet", metadata);
        last_timestamp_pair_ns_ = callback_ns;
      }
    }
  }

  void publishCameraInfo(const ros::WallTimerEvent &) {
    if (width_ == 0U || height_ == 0U) {
      return;
    }
    sensor_msgs::CameraInfo info = camera_info_manager_->getCameraInfo();
    info.header.stamp = ros::Time::now();
    info.header.frame_id = frame_id_;
    const double source_width = info.width ? info.width : source_width_;
    const double source_height = info.height ? info.height : source_height_;
    const double scale_x = processing_width_ / source_width;
    const double scale_y = processing_height_ / source_height;
    info.K[0] *= scale_x;
    info.K[2] *= scale_x;
    info.K[4] *= scale_y;
    info.K[5] *= scale_y;
    info.P[0] *= scale_x;
    info.P[2] *= scale_x;
    info.P[5] *= scale_y;
    info.P[6] *= scale_y;
    info.width = processing_width_;
    info.height = processing_height_;
    info.binning_x = info.binning_y = 0U;
    info.roi.x_offset = info.roi.y_offset = 0U;
    info.roi.width = processing_width_;
    info.roi.height = processing_height_;
    camera_info_publisher_.publish(info);
  }

  std::string input_topic_;
  std::string output_topic_;
  std::string camera_info_topic_;
  std::string image_topic_;
  std::string frame_id_;
  std::string camera_name_;
  std::string camera_info_url_;
  int subscriber_queue_size_{100};
  int event_reserve_{1048576};
  double camera_info_rate_{5.0};
  double image_fps_{10.0};
  std::unique_ptr<AdapterPerformance> perf_;
  ros::Subscriber subscriber_;
  ros::Publisher publisher_;
  ros::Publisher camera_info_publisher_;
  ros::Publisher image_publisher_;
  ros::WallTimer camera_info_timer_;
  std::unique_ptr<camera_info_manager::CameraInfoManager> camera_info_manager_;
  event_camera_codecs::DecoderFactory<event_camera_msgs::EventPacket, AdapterProcessor> factory_;
  AdapterProcessor processor_{this};
  dvs_msgs::EventArray message_;
  SequenceTracker sequence_;
  ros::Time packet_stamp_;
  uint64_t sensor_anchor_ns_{0};
  uint64_t ros_anchor_ns_{0};
  uint64_t previous_sensor_time_ns_{0};
  uint64_t last_timestamp_pair_ns_{0};
  uint64_t packets_{0};
  uint64_t events_{0};
  uint64_t time_reversals_{0};
  uint64_t reported_gaps_{0};
  uint32_t width_{0};
  uint32_t height_{0};
  uint32_t source_width_{0};
  uint32_t source_height_{0};
  uint32_t processing_width_{960};
  uint32_t processing_height_{720};
  uint64_t image_period_ns_{100000000ULL};
  uint64_t last_image_publish_ns_{0};
  std::vector<uint8_t> preview_;
  bool have_anchor_{false};
  bool have_previous_sensor_time_{false};
};

void AdapterProcessor::eventCD(uint64_t sensor_time, uint16_t x, uint16_t y,
                               uint8_t polarity) {
  owner_->append(sensor_time, x, y, polarity);
}

}  // namespace nrv_demo

PLUGINLIB_EXPORT_CLASS(nrv_demo::DvsAdapterNodelet, nodelet::Nodelet)
