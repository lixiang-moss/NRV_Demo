#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

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

// Byte fields keep the wire layout little-endian and padding-free on any host.
struct CompactEvent {
  uint8_t x_lo, x_hi, y_lo, y_hi, polarity;
};
static_assert(sizeof(CompactEvent) == 5, "NRV2 events must occupy five bytes");

class EventTcpSender {
 public:
  EventTcpSender(std::string host, int port)
      : host_(std::move(host)), port_(port), thread_(&EventTcpSender::run, this) {}

  ~EventTcpSender() { stop(); }

  void stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
      shutdown_discarded_ += queue_.size();
      queue_.clear();
      queued_bytes_ = 0;
    }
    ready_.notify_all();
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  void submit(const dvs_msgs::EventArray &message) {
    if (message.events.empty() || message.events.size() > UINT32_MAX) {
      return;
    }
    const uint32_t sequence = sequence_.fetch_add(1);
    std::vector<uint8_t> bytes;
    bytes.reserve(20U + 13U * message.events.size());
    bytes.insert(bytes.end(), {'N', 'R', 'V', '1'});
    appendBe32(bytes, message.width);
    appendBe32(bytes, message.height);
    appendBe32(bytes, static_cast<uint32_t>(message.events.size()));
    appendBe32(bytes, sequence);
    for (const auto &event : message.events) {
      appendLe16(bytes, event.x);
      appendLe16(bytes, event.y);
      appendLe64(bytes, event.ts.toNSec());
      bytes.push_back(event.polarity ? 1U : 0U);
    }
    enqueue(std::move(bytes));
  }

  void submitCompact(const std::vector<CompactEvent> &events, uint32_t width,
                     uint32_t height, uint64_t start_ns, uint64_t end_ns,
                     uint64_t raw_missing, uint64_t raw_resets) {
    if (events.empty() || events.size() > UINT32_MAX) return;
    std::vector<uint8_t> bytes;
    bytes.reserve(52U + sizeof(CompactEvent) * events.size());
    bytes.insert(bytes.end(), {'N', 'R', 'V', '2'});
    appendBe32(bytes, width);
    appendBe32(bytes, height);
    appendBe32(bytes, static_cast<uint32_t>(events.size()));
    appendBe32(bytes, sequence_.fetch_add(1));
    appendBe32(bytes, static_cast<uint32_t>(start_ns >> 32));
    appendBe32(bytes, static_cast<uint32_t>(start_ns));
    appendBe32(bytes, static_cast<uint32_t>(end_ns >> 32));
    appendBe32(bytes, static_cast<uint32_t>(end_ns));
    appendBe32(bytes, static_cast<uint32_t>(raw_missing >> 32));
    appendBe32(bytes, static_cast<uint32_t>(raw_missing));
    appendBe32(bytes, static_cast<uint32_t>(raw_resets >> 32));
    appendBe32(bytes, static_cast<uint32_t>(raw_resets));
    const auto *data = reinterpret_cast<const uint8_t *>(events.data());
    bytes.insert(bytes.end(), data, data + sizeof(CompactEvent) * events.size());
    enqueue(std::move(bytes));
  }

  uint64_t sent() const { return sent_.load(); }
  uint64_t dropped() const { return overflow_dropped_.load() + send_failed_.load(); }
  uint64_t overflowDropped() const { return overflow_dropped_.load(); }
  uint64_t sendFailed() const { return send_failed_.load(); }
  uint64_t shutdownDiscarded() const { return shutdown_discarded_.load(); }
  uint64_t sentBytes() const { return sent_bytes_.load(); }
  size_t queuePeak() const { return queue_peak_.load(); }
  size_t queueBytesPeak() const { return queue_bytes_peak_.load(); }

 private:
  void enqueue(std::vector<uint8_t> bytes) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stopping_) {
        ++shutdown_discarded_;
        return;
      }
      if (bytes.size() > max_queue_bytes_) {
        ++overflow_dropped_;
        return;
      }
      while (!queue_.empty() && (queue_.size() >= max_queue_ ||
                                queued_bytes_ + bytes.size() > max_queue_bytes_)) {
        queued_bytes_ -= queue_.front().size();
        queue_.pop_front();
        ++overflow_dropped_;
      }
      queued_bytes_ += bytes.size();
      queue_.push_back(std::move(bytes));
      queue_peak_ = std::max(queue_peak_.load(), queue_.size());
      queue_bytes_peak_ = std::max(queue_bytes_peak_.load(), queued_bytes_);
    }
    ready_.notify_one();
  }

  static void appendBe32(std::vector<uint8_t> &out, uint32_t value) {
    const uint32_t encoded = htonl(value);
    const auto *p = reinterpret_cast<const uint8_t *>(&encoded);
    out.insert(out.end(), p, p + sizeof(encoded));
  }

  static void appendLe16(std::vector<uint8_t> &out, uint16_t value) {
    out.push_back(static_cast<uint8_t>(value));
    out.push_back(static_cast<uint8_t>(value >> 8));
  }

  static void appendLe64(std::vector<uint8_t> &out, uint64_t value) {
    for (unsigned int shift = 0; shift < 64; shift += 8) {
      out.push_back(static_cast<uint8_t>(value >> shift));
    }
  }

  int connectSocket() const {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
      return -1;
    }
    struct timeval timeout;
    timeout.tv_sec = 1;
    timeout.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    struct sockaddr_in address;
    std::memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<uint16_t>(port_));
    if (inet_pton(AF_INET, host_.c_str(), &address.sin_addr) != 1 ||
        connect(fd, reinterpret_cast<struct sockaddr *>(&address), sizeof(address)) != 0) {
      close(fd);
      return -1;
    }
    return fd;
  }

  bool sendAll(int fd, const std::vector<uint8_t> &bytes) {
    size_t offset = 0;
    while (offset < bytes.size()) {
      if (stopping_) return false;
      const ssize_t count = send(
          fd, bytes.data() + offset, bytes.size() - offset, MSG_NOSIGNAL);
      if (count < 0 && errno == EINTR) continue;
      if (count <= 0) {
        return false;
      }
      offset += static_cast<size_t>(count);
    }
    return true;
  }

  void run() {
    int fd = -1;
    while (true) {
      std::vector<uint8_t> item;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait_for(lock, std::chrono::milliseconds(200), [this] {
          return stopping_ || !queue_.empty();
        });
        if (stopping_) {
          break;
        }
        if (queue_.empty()) {
          continue;
        }
        item = std::move(queue_.front());
        queued_bytes_ -= item.size();
        queue_.pop_front();
      }
      if (fd < 0) {
        fd = connectSocket();
      }
      if (fd < 0 || !sendAll(fd, item)) {
        if (fd >= 0) {
          close(fd);
          fd = -1;
        }
        if (stopping_) {
          ++shutdown_discarded_;
        } else {
          ++send_failed_;
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        continue;
      }
      ++sent_;
      sent_bytes_ += item.size();
    }
    if (fd >= 0) {
      close(fd);
    }
  }

  const std::string host_;
  const int port_;
  const size_t max_queue_{64};
  const size_t max_queue_bytes_{256U * 1024U * 1024U};
  std::mutex mutex_;
  std::condition_variable ready_;
  std::deque<std::vector<uint8_t>> queue_;
  size_t queued_bytes_{0};
  std::atomic<bool> stopping_{false};
  std::atomic<uint32_t> sequence_{0};
  std::atomic<uint64_t> sent_{0};
  std::atomic<uint64_t> overflow_dropped_{0}, send_failed_{0}, shutdown_discarded_{0};
  std::atomic<uint64_t> sent_bytes_{0};
  std::atomic<size_t> queue_peak_{0};
  std::atomic<size_t> queue_bytes_peak_{0};
  // Start only after every field accessed by run() has been initialized.
  std::thread thread_;
};

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
  ~DvsAdapterNodelet() override {
    subscriber_.shutdown();
    camera_info_timer_.stop();
    if (bridge_) bridge_->stop();
    if (summary_path_.empty()) return;
    std::ofstream out(summary_path_);
    if (!out) {
      NODELET_ERROR("Cannot write adapter summary: %s", summary_path_.c_str());
      return;
    }
    out << "{\n"
        << "  \"bridge_protocol\": \"" << (bridge_compact_ ? "NRV2" : "NRV1") << "\",\n"
        << "  \"raw_packets\": " << packets_ << ",\n"
        << "  \"decoded_events\": " << events_ << ",\n"
        << "  \"raw_missing_packets\": " << sequence_.missingPackets() << ",\n"
        << "  \"raw_gap_incidents\": " << sequence_.gapIncidents() << ",\n"
        << "  \"raw_reordered_or_reset\": " << sequence_.reorderedOrReset() << ",\n"
        << "  \"mean_decode_ms\": " << (packets_ ? decode_ms_ / packets_ : 0) << ",\n"
        << "  \"mean_retime_ms\": " << (packets_ ? retime_ms_ / packets_ : 0) << ",\n"
        << "  \"mean_pack_ms\": " << (packets_ ? pack_ms_ / packets_ : 0) << ",\n"
        << "  \"mean_publish_ms\": " << (packets_ ? publish_ms_ / packets_ : 0) << ",\n"
        << "  \"max_callback_ms\": " << max_callback_ms_ << ",\n"
        << "  \"max_packet_age_ms\": " << max_packet_age_ms_ << ",\n"
        << "  \"bridge_sent_batches\": " << (bridge_ ? bridge_->sent() : 0) << ",\n"
        << "  \"bridge_dropped_batches\": " << (bridge_ ? bridge_->dropped() : 0) << ",\n"
        << "  \"bridge_overflow_batches\": " << (bridge_ ? bridge_->overflowDropped() : 0) << ",\n"
        << "  \"bridge_send_failed_batches\": " << (bridge_ ? bridge_->sendFailed() : 0) << ",\n"
        << "  \"bridge_shutdown_discarded_batches\": " << (bridge_ ? bridge_->shutdownDiscarded() : 0) << ",\n"
        << "  \"bridge_sent_bytes\": " << (bridge_ ? bridge_->sentBytes() : 0) << ",\n"
        << "  \"bridge_queue_peak\": " << (bridge_ ? bridge_->queuePeak() : 0) << ",\n"
        << "  \"bridge_queue_bytes_peak\": " << (bridge_ ? bridge_->queueBytesPeak() : 0) << "\n}\n";
  }

  void append(uint64_t, uint16_t x, uint16_t y, uint8_t polarity) {
    if (bridge_compact_) {
      compact_events_.push_back({static_cast<uint8_t>(x), static_cast<uint8_t>(x >> 8),
                                static_cast<uint8_t>(y), static_cast<uint8_t>(y >> 8),
                                static_cast<uint8_t>(polarity != 0U)});
      if (!publish_events_) return;
    }
    dvs_msgs::Event event;
    event.x = x;
    event.y = y;
    event.polarity = polarity != 0U;
    event.ts = packet_stamp_;
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
    private_nh.param<std::string>("bridge_host", bridge_host_, "127.0.0.1");
    private_nh.param("bridge_port", bridge_port_, 0);
    private_nh.param("bridge_compact", bridge_compact_, false);
    bridge_compact_ = bridge_compact_ && bridge_port_ > 0;
    private_nh.param<std::string>("summary_path", summary_path_, "");
    if (event_reserve_ < 1 || subscriber_queue_size_ < 1) {
      throw std::runtime_error("adapter queue and reserve values must be positive");
    }

    if (bridge_compact_) compact_events_.reserve(static_cast<std::size_t>(event_reserve_));
    else message_.events.reserve(static_cast<std::size_t>(event_reserve_));
    if (bridge_port_ > 0) {
      bridge_.reset(new EventTcpSender(bridge_host_, bridge_port_));
    }
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
    const auto tick = std::chrono::steady_clock::now();
    max_packet_age_ms_ = std::max(max_packet_age_ms_,
                                 (ros::Time::now() - packet->header.stamp).toSec() * 1000);
    sequence_.observe(packet->seq);
    packet_stamp_ = packet->header.stamp;
    message_.events.clear();
    compact_events_.clear();
    publish_events_ = publisher_.getNumSubscribers() > 0;
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
    const auto decoded = std::chrono::steady_clock::now();
    decode_ms_ += milliseconds(decoded - tick);
    const size_t event_count = bridge_compact_ ? compact_events_.size() : message_.events.size();
    if (event_count) {
      const auto interval = packetInterval(packet->header.stamp);
      if (!message_.events.empty()) retimeMessage(interval.first, interval.second);
      message_.header.stamp.fromNSec(interval.second);
      const auto retimed = std::chrono::steady_clock::now();
      retime_ms_ += milliseconds(retimed - decoded);
      if (bridge_) {
        if (bridge_compact_) {
          bridge_->submitCompact(compact_events_, packet->width, packet->height,
                                 interval.first, interval.second,
                                 sequence_.missingPackets(), sequence_.reorderedOrReset());
        } else {
          bridge_->submit(message_);
        }
      }
      const auto packed = std::chrono::steady_clock::now();
      pack_ms_ += milliseconds(packed - retimed);
      if (publish_events_) publisher_.publish(message_);
      publish_ms_ += milliseconds(std::chrono::steady_clock::now() - packed);
    }
    width_ = packet->width;
    height_ = packet->height;
    ++packets_;
    events_ += event_count;
    max_callback_ms_ = std::max(max_callback_ms_, milliseconds(std::chrono::steady_clock::now() - tick));
    if (sequence_.gapIncidents() > reported_gaps_) {
      reported_gaps_ = sequence_.gapIncidents();
      NODELET_ERROR_THROTTLE(1.0, "adapter RAW missing=%llu, gap incidents=%llu",
                            static_cast<unsigned long long>(sequence_.missingPackets()),
                            static_cast<unsigned long long>(reported_gaps_));
    }
    NODELET_INFO_THROTTLE(5.0,
      "adapter: packets=%llu events=%llu RAW missing=%llu | decode %.2f ms, retime %.2f ms, pack %.2f ms/packet | TCP overflow=%llu failed=%llu",
      static_cast<unsigned long long>(packets_), static_cast<unsigned long long>(events_),
      static_cast<unsigned long long>(sequence_.missingPackets()),
      decode_ms_ / packets_, retime_ms_ / packets_, pack_ms_ / packets_,
      static_cast<unsigned long long>(bridge_ ? bridge_->overflowDropped() : 0),
      static_cast<unsigned long long>(bridge_ ? bridge_->sendFailed() : 0));
  }

  template <typename Duration>
  static double milliseconds(Duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
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

  std::pair<uint64_t, uint64_t> packetInterval(const ros::Time &host_stamp) {
    static constexpr uint64_t kDefaultPacketNs = 10000000ULL;
    static constexpr uint64_t kMaxPacketNs = 100000000ULL;
    const uint64_t end_ns = host_stamp.isZero() ? ros::Time::now().toNSec()
                                                : host_stamp.toNSec();
    uint64_t span_ns = kDefaultPacketNs;
    if (last_packet_host_ns_ > 0 && end_ns > last_packet_host_ns_) {
      span_ns = std::min(end_ns - last_packet_host_ns_, kMaxPacketNs);
    }
    const uint64_t start_ns = end_ns > span_ns ? end_ns - span_ns : 0;
    last_packet_host_ns_ = end_ns;
    return {start_ns, end_ns};
  }

  void retimeMessage(uint64_t start_ns, uint64_t end_ns) {
    const uint64_t span_ns = end_ns - start_ns;
    const size_t last = message_.events.size() - 1;
    for (size_t index = 0; index < message_.events.size(); ++index) {
      const uint64_t offset_ns = last > 0
        ? span_ns * index / last
        : 0;
      message_.events[index].ts.fromNSec(start_ns + offset_ns);
    }
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
  std::string bridge_host_{"127.0.0.1"};
  std::string summary_path_;
  double decode_ms_{0}, retime_ms_{0}, pack_ms_{0}, publish_ms_{0};
  double max_callback_ms_{0}, max_packet_age_ms_{0};
  int bridge_port_{0};
  bool bridge_compact_{false}, publish_events_{false};
  std::vector<CompactEvent> compact_events_;
  ros::Subscriber subscriber_;
  ros::Publisher publisher_;
  ros::Publisher camera_info_publisher_;
  ros::WallTimer camera_info_timer_;
  std::unique_ptr<camera_info_manager::CameraInfoManager> camera_info_manager_;
  std::unique_ptr<EventTcpSender> bridge_;
  event_camera_codecs::DecoderFactory<event_camera_msgs::EventPacket, AdapterProcessor> factory_;
  AdapterProcessor processor_{this};
  dvs_msgs::EventArray message_;
  SequenceTracker sequence_;
  ros::Time packet_stamp_;
  uint64_t packets_{0};
  uint64_t events_{0};
  uint64_t last_packet_host_ns_{0};
  uint64_t reported_gaps_{0};
  uint32_t width_{0};
  uint32_t height_{0};
};

void AdapterProcessor::eventCD(uint64_t sensor_time, uint16_t x, uint16_t y,
                               uint8_t polarity) {
  owner_->append(sensor_time, x, y, polarity);
}

}  // namespace nrv_demo

PLUGINLIB_EXPORT_CLASS(nrv_demo::DvsAdapterNodelet, nodelet::Nodelet)
