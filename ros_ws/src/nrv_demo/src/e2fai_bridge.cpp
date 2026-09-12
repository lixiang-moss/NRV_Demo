#include <arpa/inet.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <dvs_msgs/EventArray.h>
#include <nlohmann/json.hpp>
#include <nrv_demo/E2faiResult.h>
#include <nrv_demo/performance.hpp>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/String.h>

namespace nrv_demo {
namespace {
using Json = nlohmann::json;
constexpr size_t kMaxPayload = 256U * 1024U * 1024U;
constexpr size_t kMaxMetadata = 65536U;
constexpr size_t kMaxBatches = 32U;

uint64_t monotonicNs() {
  timespec value{};
  if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
    throw std::runtime_error("Cannot read monotonic clock");
  }
  return static_cast<uint64_t>(value.tv_sec) * 1000000000ULL + value.tv_nsec;
}

std::string newSessionId() {
  std::random_device random;
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (unsigned int i = 0; i < 4; ++i) out << std::setw(8) << random();
  return out.str();
}

void makeDirectories(const std::string &directory) {
  if (directory.empty()) throw std::runtime_error("output_dir must not be empty");
  for (size_t end = 1; end <= directory.size(); ++end) {
    if (end != directory.size() && directory[end] != '/') continue;
    const std::string part = directory.substr(0, end);
    if (::mkdir(part.c_str(), 0775) != 0 && errno != EEXIST) {
      throw std::runtime_error("Cannot create output directory: " + part);
    }
  }
}

void storeBe(uint8_t *destination, uint64_t value, size_t bytes) {
  for (size_t i = 0; i < bytes; ++i) {
    destination[bytes - i - 1] = static_cast<uint8_t>(value >> (i * 8));
  }
}

uint64_t loadBe(const uint8_t *source, size_t bytes) {
  uint64_t value = 0;
  for (size_t i = 0; i < bytes; ++i) value = (value << 8) | source[i];
  return value;
}

void storeLe(uint8_t *destination, uint64_t value, size_t bytes) {
  for (size_t i = 0; i < bytes; ++i) destination[i] = static_cast<uint8_t>(value >> (i * 8));
}

uint64_t unsignedField(const Json &metadata, const char *name,
                       uint64_t maximum = std::numeric_limits<uint64_t>::max()) {
  const auto &value = metadata.at(name);
  if (!value.is_number_unsigned() && !(value.is_number_integer() && value.get<int64_t>() >= 0)) {
    throw std::runtime_error(std::string("Expected unsigned integer metadata: ") + name);
  }
  const uint64_t result = value.get<uint64_t>();
  if (result > maximum) throw std::runtime_error(std::string("Metadata exceeds range: ") + name);
  return result;
}
}  // namespace

// One ROS callback thread preserves subscription order. Network sending and
// receiving are independent threads; neither inference nor socket I/O runs on it.
class E2faiBridge {
 public:
  E2faiBridge() : private_nh_("~") {
    private_nh_.param<std::string>("session_id", session_id_, "");
    if (session_id_.empty()) session_id_ = newSessionId();
    private_nh_.param<std::string>("host", host_, "127.0.0.1");
    private_nh_.param("port", port_, 8765);
    if (!private_nh_.getParam("output_dir", output_dir_)) {
      throw std::runtime_error("Missing output_dir parameter");
    }
    if (host_ != "127.0.0.1" || port_ < 1 || port_ > 65535) {
      throw std::runtime_error("The host GPU bridge requires 127.0.0.1 and a valid port");
    }
    makeDirectories(output_dir_);
    bool enabled = false;
    double interval = 5.0;
    int capacity = 4096;
    private_nh_.param("perf_enabled", enabled, false);
    private_nh_.param("perf_interval_s", interval, 5.0);
    private_nh_.param("perf_capacity", capacity, 4096);
    if (enabled) {
      if (interval <= 0 || capacity < 1) throw std::runtime_error("Invalid performance configuration");
      perf_.reset(new PerformanceRecorder(output_dir_ + "/performance_ros_bridge.jsonl",
                                          "ros_bridge", interval, static_cast<size_t>(capacity)));
    }
    status_publisher_ = private_nh_.advertise<std_msgs::String>("status", 1, true);
    result_publisher_ = private_nh_.advertise<E2faiResult>("result", 1);
    subscriber_ = nh_.subscribe("/dvs/events", 10, &E2faiBridge::onEvents, this);
    timer_ = nh_.createWallTimer(ros::WallDuration(1), &E2faiBridge::onTimer, this);
    publishStatus();
    sender_ = std::thread(&E2faiBridge::sendLoop, this);
  }

  ~E2faiBridge() { close(); }

  void close() {
    if (closed_.exchange(true)) return;
    stopped_ = true;
    subscriber_.shutdown();
    timer_.stop();
    closeQueue();
    shutdownSocket();
    if (sender_.joinable()) sender_.join();
    if (reader_.joinable()) reader_.join();
    {
      std::lock_guard<std::mutex> guard(connection_mutex_);
      if (socket_ >= 0) ::close(socket_);
      socket_ = -1;
    }
    perf_.reset();
    Json summary;
    {
      std::lock_guard<std::mutex> guard(state_mutex_);
      const bool failed = state_ == "paused";
      summary = {{"session_id", session_id_}, {"state", failed ? state_ : "stopped"},
                 {"state_before_stop", state_}, {"detail", detail_},
                 {"stop_reason", failed ? "failure" : "acquisition_stopped"},
                 {"local_errors", local_errors_}, {"worker_errors", worker_errors_},
                 {"error_code", error_code_}, {"batches", batches_.load()},
                 {"results", results_.load()}, {"bridge_implementation", "roscpp"}};
    }
    {
      std::lock_guard<std::mutex> guard(queue_mutex_);
      summary.update({{"queue_peak_batches", queue_peak_batches_}, {"queue_peak_bytes", queue_peak_bytes_},
                      {"discarded_batches", discarded_batches_}, {"discarded_events", discarded_events_},
                      {"discarded_bytes", discarded_bytes_}});
    }
    std::ofstream out(output_dir_ + "/bridge_summary.json");
    if (out) out << summary.dump(2) << '\n';
    else ROS_ERROR("Cannot write bridge_summary.json");
  }

 private:
  using Clock = PerformanceRecorder::Clock;
  using Tick = PerformanceRecorder::Tick;
  struct Batch {
    Json metadata;
    std::vector<uint8_t> payload;
    uint64_t count{0}, sequence{0};
    Tick queued{};
  };

  void publishStatus() {
    Json value;
    {
      std::lock_guard<std::mutex> guard(state_mutex_);
      value = {{"session_id", session_id_}, {"state", state_}, {"detail", detail_},
               {"batches", batches_.load()}, {"results", results_.load()},
               {"local_errors", local_errors_}, {"worker_errors", worker_errors_},
               {"error_code", error_code_}};
    }
    {
      std::lock_guard<std::mutex> guard(queue_mutex_);
      value.update({{"queue_batches", queue_.size()}, {"queue_bytes", queue_bytes_},
                    {"queue_peak_batches", queue_peak_batches_}, {"queue_peak_bytes", queue_peak_bytes_}});
    }
    std_msgs::String message;
    message.data = value.dump();
    status_publisher_.publish(message);
  }

  void onTimer(const ros::WallTimerEvent &) { publishStatus(); }

  void closeQueue() {
    {
      std::lock_guard<std::mutex> guard(queue_mutex_);
      discarded_batches_ += queue_.size();
      discarded_bytes_ += queue_bytes_;
      for (const auto &batch : queue_) discarded_events_ += batch.count;
      queue_.clear();
      queue_bytes_ = 0;
    }
    queue_ready_.notify_all();
  }

  void shutdownSocket() {
    std::lock_guard<std::mutex> guard(connection_mutex_);
    if (socket_ >= 0) ::shutdown(socket_, SHUT_RDWR);
  }

  void pause(const std::string &detail, const std::string &code = "bridge_error", bool worker = false) {
    if (stopped_.exchange(true)) return;
    {
      std::lock_guard<std::mutex> guard(state_mutex_);
      state_ = "paused";
      detail_ = detail;
      error_code_ = code;
      if (worker) ++worker_errors_;
      else ++local_errors_;
    }
    closeQueue();
    shutdownSocket();
    if (perf_) perf_->observe("session_errors", 1, 0, batches_.load(), "count");
    ROS_ERROR_STREAM("E2FAI session paused: " << detail << ". Original camera and GUI continue.");
    publishStatus();
  }

  void onEvents(const dvs_msgs::EventArray::ConstPtr &message) {
    if (stopped_ || message->events.empty()) return;
    const uint64_t callback_ns = monotonicNs();
    const Tick begin = perf_ ? Clock::now() : Tick{};
    try {
      if (message->events.size() > kMaxPayload / 13U) {
        pause("Input event payload exceeds 256 MiB", "input_fifo_overflow");
        return;
      }
      Batch batch;
      batch.count = message->events.size();
      batch.payload.resize(batch.count * 13U);
      uint8_t *record = batch.payload.data();
      for (const auto &event : message->events) {
        storeLe(record, event.x, 2);
        storeLe(record + 2, event.y, 2);
        storeLe(record + 4, event.ts.toNSec(), 8);
        record[12] = event.polarity ? 1 : 0;
        record += 13;
      }
      batch.sequence = batches_.fetch_add(1);
      batch.metadata = {{"session_id", session_id_}, {"batch_seq", batch.sequence},
                        {"ros_header_seq", message->header.seq}, {"ros_sequence_valid", true},
                        {"header_stamp_ns", message->header.stamp.toNSec()},
                        {"width", message->width}, {"height", message->height},
                        {"count", batch.count}, {"callback_ns", callback_ns}};
      const uint64_t sequence = batch.sequence, count = batch.count;
      bool overflow = false;
      {
        std::lock_guard<std::mutex> guard(queue_mutex_);
        if (stopped_) return;
        overflow = queue_.size() >= kMaxBatches || queue_bytes_ + batch.payload.size() > kMaxPayload;
        if (!overflow) {
          batch.queued = perf_ ? Clock::now() : Tick{};
          queue_bytes_ += batch.payload.size();
          queue_.push_back(std::move(batch));
          queue_peak_batches_ = std::max(queue_peak_batches_, queue_.size());
          queue_peak_bytes_ = std::max(queue_peak_bytes_, queue_bytes_);
        }
      }
      if (overflow) {
        pause("Input FIFO exceeded 32 batches or 256 MiB; restart acquisition to resume", "input_fifo_overflow");
        return;
      }
      queue_ready_.notify_one();
      if (perf_) perf_->duration("object_to_wire", begin, Clock::now(), count, sequence);
    } catch (const std::exception &error) {
      pause(error.what(), "event_conversion_error");
    }
  }

  int connectSocket() {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    const int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
      ::close(fd);
      return -1;
    }
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<uint16_t>(port_));
    inet_pton(AF_INET, host_.c_str(), &address.sin_addr);
    bool connected = ::connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0;
    if (!connected && errno == EINPROGRESS) {
      for (int attempt = 0; attempt < 10 && !stopped_; ++attempt) {
        pollfd target{fd, POLLOUT, 0};
        const int ready = ::poll(&target, 1, 100);
        if (ready < 0 && errno == EINTR) continue;
        if (ready > 0) {
          int error = 0;
          socklen_t size = sizeof(error);
          connected = getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &size) == 0 && error == 0;
          break;
        }
      }
    }
    if (!connected || stopped_ || fcntl(fd, F_SETFL, flags) < 0) {
      ::close(fd);
      return -1;
    }
    return fd;
  }

  void sendAll(int fd, const uint8_t *data, size_t size) {
    size_t offset = 0;
    while (offset < size && !stopped_) {
      const ssize_t sent = ::send(fd, data + offset, size - offset, MSG_NOSIGNAL);
      if (sent < 0 && errno == EINTR) continue;
      if (sent <= 0) throw std::runtime_error("GPU event connection send failed");
      offset += static_cast<size_t>(sent);
    }
    if (offset != size) throw std::runtime_error("Event send interrupted by session shutdown");
  }

  void sendPacket(int fd, const Batch &batch) {
    const std::string metadata = Json({{"kind", "events"}, {"meta", batch.metadata}}).dump();
    if (metadata.size() > kMaxMetadata) throw std::runtime_error("Event metadata exceeds protocol limit");
    std::array<uint8_t, 16> header{{'N', 'E', 'F', '1'}};
    storeBe(header.data() + 4, metadata.size(), 4);
    storeBe(header.data() + 8, batch.payload.size(), 8);
    sendAll(fd, header.data(), header.size());
    sendAll(fd, reinterpret_cast<const uint8_t *>(metadata.data()), metadata.size());
    sendAll(fd, batch.payload.data(), batch.payload.size());
  }

  void sendLoop() {
    try {
      int fd = -1;
      while (!stopped_) {
        fd = connectSocket();
        if (fd >= 0) break;
        std::unique_lock<std::mutex> guard(queue_mutex_);
        queue_ready_.wait_for(guard, std::chrono::milliseconds(500), [this] { return stopped_.load(); });
      }
      if (fd < 0) return;
      {
        std::lock_guard<std::mutex> guard(connection_mutex_);
        if (stopped_) { ::close(fd); return; }
        socket_ = fd;
      }
      {
        std::lock_guard<std::mutex> guard(state_mutex_);
        if (!stopped_) state_ = "connected";
      }
      reader_ = std::thread(&E2faiBridge::receiveLoop, this, fd);
      publishStatus();
      while (!stopped_) {
        Batch batch;
        {
          std::unique_lock<std::mutex> guard(queue_mutex_);
          queue_ready_.wait(guard, [this] { return stopped_ || !queue_.empty(); });
          if (stopped_) break;
          batch = std::move(queue_.front());
          queue_bytes_ -= batch.payload.size();
          queue_.pop_front();
        }
        if (perf_) perf_->duration("input_queue_wait", batch.queued, Clock::now(), batch.count, batch.sequence);
        const Tick begin = perf_ ? Clock::now() : Tick{};
        sendPacket(fd, batch);
        if (perf_) perf_->duration("event_send", begin, Clock::now(), batch.count, batch.sequence);
      }
    } catch (const std::exception &error) {
      if (!stopped_) pause(error.what());
    }
  }

  bool receiveAll(int fd, uint8_t *data, size_t size) {
    size_t offset = 0;
    while (offset < size && !stopped_) {
      const ssize_t count = ::recv(fd, data + offset, size - offset, 0);
      if (count < 0 && errno == EINTR) continue;
      if (count <= 0) return false;
      offset += static_cast<size_t>(count);
    }
    return offset == size;
  }

  void fillImage(sensor_msgs::Image &image, const E2faiResult &result, const std::string &encoding,
                 uint32_t bytes_per_pixel, const std::vector<uint8_t> &payload, size_t offset, size_t size) {
    image.header = result.header;
    image.width = result.width;
    image.height = result.height;
    image.encoding = encoding;
    image.is_bigendian = false;
    image.step = result.width * bytes_per_pixel;
    image.data.assign(payload.begin() + offset, payload.begin() + offset + size);
  }

  void publishResult(const Json &metadata, const std::vector<uint8_t> &payload) {
    const Tick begin = perf_ ? Clock::now() : Tick{};
    E2faiResult result;
    result.session_id = metadata.at("session_id").get<std::string>();
    result.width = unsignedField(metadata, "width", 10000);
    result.height = unsignedField(metadata, "height", 10000);
    const size_t pixels = static_cast<size_t>(result.width) * result.height;
    if (!pixels || payload.size() != pixels * 12U) {
      throw std::runtime_error("Result payload does not match mono8 + rgb8 + 32FC2 dimensions");
    }
    result.window_id = unsignedField(metadata, "window_id");
    result.source_batch_first = unsignedField(metadata, "source_batch_first");
    result.source_batch_last = unsignedField(metadata, "source_batch_last");
    result.window_start_ns = unsignedField(metadata, "window_start_ns");
    result.window_end_ns = unsignedField(metadata, "window_end_ns");
    result.event_first_ns = unsignedField(metadata, "event_first_ns");
    result.event_last_ns = unsignedField(metadata, "event_last_ns");
    result.event_count = unsignedField(metadata, "event_count");
    result.reset_reason = metadata.at("reset_reason").get<std::string>();
    result.reset_count = unsignedField(metadata, "reset_count");
    result.source_callback_ns = unsignedField(metadata, "source_callback_ns");
    result.completed_ns = unsignedField(metadata, "completed_ns");
    result.header.seq = static_cast<uint32_t>(result.window_id);
    result.header.stamp.fromNSec(result.window_end_ns);
    result.header.frame_id = "delta_camera_optical_frame";
    fillImage(result.gray, result, "mono8", 1, payload, 0, pixels);
    fillImage(result.flow_preview, result, "rgb8", 3, payload, pixels, pixels * 3U);
    fillImage(result.flow, result, "32FC2", 8, payload, pixels * 4U, pixels * 8U);
    if (stopped_) return;
    result_publisher_.publish(result);
    if (perf_) {
      perf_->duration("result_construct_publish", begin, Clock::now(), 0, result.window_id);
      const double age_ms = static_cast<double>(static_cast<int64_t>(monotonicNs()) -
                                                static_cast<int64_t>(result.source_callback_ns)) / 1e6;
      perf_->observe("callback_to_result_receive", age_ms, 0, result.window_id);
    }
    ++results_;
  }

  void receiveLoop(int fd) {
    try {
      while (!stopped_) {
        std::array<uint8_t, 16> header{};
        if (!receiveAll(fd, header.data(), header.size())) {
          if (!stopped_) pause("GPU worker disconnected");
          return;
        }
        const uint64_t metadata_size = loadBe(header.data() + 4, 4);
        const uint64_t payload_size = loadBe(header.data() + 8, 8);
        if (std::memcmp(header.data(), "NEF1", 4) || !metadata_size || metadata_size > kMaxMetadata ||
            payload_size > kMaxPayload) throw std::runtime_error("Invalid NEF1 result header");
        std::string encoded(metadata_size, '\0');
        std::vector<uint8_t> payload(payload_size);
        if (!receiveAll(fd, reinterpret_cast<uint8_t *>(&encoded[0]), encoded.size()) ||
            !receiveAll(fd, payload.data(), payload.size())) {
          if (!stopped_) pause("GPU worker disconnected inside result packet");
          return;
        }
        const Json envelope = Json::parse(encoded);
        const std::string kind = envelope.at("kind").get<std::string>();
        const Json &metadata = envelope.at("meta");
        if (!metadata.is_object()) throw std::runtime_error("Result metadata must be an object");
        if (metadata.value("session_id", std::string()) != session_id_) {
          if (perf_) perf_->observe("stale_session_results", 1, 0, results_.load(), "count");
          continue;
        }
        if (kind == "error") {
          pause(metadata.value("message", std::string("GPU worker error")),
                metadata.value("code", std::string("worker_error")), true);
          return;
        }
        if (kind != "result") throw std::runtime_error("Unexpected GPU packet: " + kind);
        publishResult(metadata, payload);
      }
    } catch (const std::exception &error) {
      if (!stopped_) pause(error.what());
    }
  }

  ros::NodeHandle nh_, private_nh_;
  ros::Publisher status_publisher_, result_publisher_;
  ros::Subscriber subscriber_;
  ros::WallTimer timer_;
  std::string session_id_, host_, output_dir_;
  int port_{8765};
  std::unique_ptr<PerformanceRecorder> perf_;
  std::atomic<bool> stopped_{false}, closed_{false};
  std::atomic<uint64_t> batches_{0}, results_{0};
  std::mutex state_mutex_;
  std::string state_{"connecting"}, detail_, error_code_;
  uint64_t local_errors_{0}, worker_errors_{0};
  std::mutex queue_mutex_;
  std::condition_variable queue_ready_;
  std::deque<Batch> queue_;
  size_t queue_bytes_{0}, queue_peak_batches_{0}, queue_peak_bytes_{0};
  uint64_t discarded_batches_{0}, discarded_events_{0}, discarded_bytes_{0};
  std::mutex connection_mutex_;
  int socket_{-1};
  std::thread sender_, reader_;
};
}  // namespace nrv_demo

int main(int argc, char **argv) {
  ros::init(argc, argv, "nrv_e2fai");
  try {
    nrv_demo::E2faiBridge bridge;
    ros::spin();
    bridge.close();
  } catch (const std::exception &error) {
    ROS_FATAL_STREAM("Cannot run E2FAI bridge: " << error.what());
    return 1;
  }
  return 0;
}
