#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>
#include <event_camera_codecs/decoder_factory.h>
#include <event_camera_codecs/event_processor.h>
#include <event_camera_codecs/encoder.h>
#include <event_camera_codecs/mono_encoder.h>
#include <event_camera_msgs/EventPacket.h>
#include <nodelet/nodelet.h>
#include <ros/ros.h>
#include <pluginlib/class_list_macros.h>
#include <std_msgs/UInt64MultiArray.h>

extern "C" void nrv_filter(const intptr_t*, const intptr_t*, const double*, size_t,
                          int, int, double*, double*, bool, double, bool, double, uint8_t*);

namespace nrv_demo {
// Decode once at the RAW ingress, filter, then feed both consumers the same packet stream.
class RawNoiseNodelet : public nodelet::Nodelet, public event_camera_codecs::EventProcessor {
 public:
  void eventCD(uint64_t t, uint16_t x, uint16_t y, uint8_t p) override {
    x_.push_back(x); y_.push_back(y); ns_.push_back(t);
    seconds_.push_back(t * 1e-9); polarity_.push_back(p);
  }
  bool eventExtTrigger(uint64_t, uint8_t, uint8_t) override { return true; }
  void finished() override {}
  void rawData(const char*, size_t) override {}

 private:
  void onInit() override {
    auto nh = getNodeHandle();
    pub_ = nh.advertise<event_camera_msgs::EventPacket>("/nrv_noise_filter/events", 100);
    stats_pub_ = nh.advertise<std_msgs::UInt64MultiArray>("/nrv_noise_filter/counts", 1, true);
    sub_ = nh.subscribe("/delta_driver/events", 100, &RawNoiseNodelet::packet, this);
    timer_ = nh.createWallTimer(ros::WallDuration(1), &RawNoiseNodelet::stats, this);
    NODELET_INFO("RAW event filtering: /delta_driver/events -> /nrv_noise_filter/events (mono)");
  }

  void stats(const ros::WallTimerEvent&) {
    std_msgs::UInt64MultiArray counts;
    counts.data = {input_count_, kept_count_};
    stats_pub_.publish(counts);
  }

  void packet(const event_camera_msgs::EventPacket::ConstPtr& input) {
    bool background = false, refractory = false;
    double window_ms = 5, interval_ms = 1;
    XmlRpc::XmlRpcValue config;
    if (getNodeHandle().getParamCached("/nrv_noise", config)) {
      background = static_cast<bool>(config["background"]);
      refractory = static_cast<bool>(config["refractory"]);
      window_ms = static_cast<double>(config["window_ms"]);
      interval_ms = static_cast<double>(config["interval_ms"]);
    }
    const bool reset = width_ != input->width || height_ != input->height ||
        background != background_ || refractory != refractory_ ||
        window_ms != window_ms_ || interval_ms != interval_ms_;
    width_ = input->width; height_ = input->height;
    background_ = background; refractory_ = refractory;
    window_ms_ = window_ms; interval_ms_ = interval_ms;
    x_.clear(); y_.clear(); ns_.clear(); seconds_.clear(); polarity_.clear();
    auto* decoder = factory_.getInstance(*input);
    if (!decoder) {
      NODELET_ERROR_THROTTLE(5, "Unsupported input encoding: %s", input->encoding.c_str());
      return;
    }
    decoder->setTimeMultiplier(1000);
    while (decoder->decode(*input, this)) {}
    if (reset || (!ns_.empty() && have_time_ && ns_.front() < last_ns_)) {
      seen_.assign(static_cast<size_t>(width_) * height_, -std::numeric_limits<double>::infinity());
      accepted_ = seen_;
    }
    if (!ns_.empty()) { last_ns_ = ns_.back(); have_time_ = true; }
    input_count_ += x_.size();
    keep_.resize(x_.size());
    nrv_filter(x_.data(), y_.data(), seconds_.data(), x_.size(), width_, height_,
               seen_.data(), accepted_.data(), background_, window_ms_ / 1000.,
               refractory_, interval_ms_ / 1000., keep_.data());

    event_camera_msgs::EventPacket output;
    output.header = input->header;
    output.width = width_; output.height = height_;
    output.encoding = "mono";
    output.is_bigendian = false;  // This container targets little-endian linux/amd64.
    event_camera_codecs::mono::Encoder encoder;
    encoder.setBuffer(&output.events);
    for (size_t i = 0; i < x_.size(); ++i) {
      if (!keep_[i]) continue;
      // mono stores a signed 32-bit nanosecond offset. Start a fresh packet if needed.
      if (!output.events.empty() && (ns_[i] < output.time_base ||
          ns_[i] - output.time_base > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()))) {
        output.seq = seq_++;
        pub_.publish(output);
        output.events.clear();
      }
      if (output.events.empty()) output.time_base = ns_[i];
      encoder.encodeCD(static_cast<int32_t>(ns_[i] - output.time_base), x_[i], y_[i], polarity_[i]);
      ++kept_count_;
    }
    // Empty packets also mark progress when the entire input batch is filtered out.
    output.seq = seq_++;
    pub_.publish(output);
  }

  ros::Publisher pub_, stats_pub_;
  ros::Subscriber sub_;
  ros::WallTimer timer_;
  event_camera_codecs::DecoderFactory<event_camera_msgs::EventPacket, RawNoiseNodelet> factory_;
  std::vector<intptr_t> x_, y_;
  std::vector<uint64_t> ns_;
  std::vector<double> seconds_, seen_, accepted_;
  std::vector<uint8_t> polarity_, keep_;
  uint32_t width_{0}, height_{0};
  bool background_{false}, refractory_{false}, have_time_{false};
  double window_ms_{5}, interval_ms_{1};
  uint64_t input_count_{0}, kept_count_{0}, seq_{0}, last_ns_{0};
};
}
PLUGINLIB_EXPORT_CLASS(nrv_demo::RawNoiseNodelet, nodelet::Nodelet)
