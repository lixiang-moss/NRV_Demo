#pragma once
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

#include <ros/ros.h>

namespace nrv_demo {
// Optional diagnostics only. Sorting and disk I/O never run on a RAW callback.
class PerformanceRecorder {
 public:
  using Clock = std::chrono::steady_clock;
  using Tick = Clock::time_point;
  using Metadata = std::map<std::string, uint64_t>;

  PerformanceRecorder(std::string path, std::string component, double interval, size_t capacity)
      : path_(std::move(path)), component_(std::move(component)), interval_(interval), capacity_(capacity), start_(Clock::now()),
        thread_([this] { run(); }) {}

  ~PerformanceRecorder() {
    {
      std::lock_guard<std::mutex> guard(mutex_);
      stopping_ = true;
    }
    condition_.notify_all();
    thread_.join();
    flush(true);
  }

  void observe(const std::string &name, double value, uint64_t events, uint64_t sequence,
               const std::string &unit = "ms", const Metadata &metadata = Metadata{}) {
    if (!std::isfinite(value)) return;
    const auto sample_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now().time_since_epoch()).count();
    std::lock_guard<std::mutex> guard(mutex_);
    auto &metric = metrics_[name];
    metric.unit = unit;
    if (!metric.count) metric.minimum = metric.maximum = value;
    ++metric.count;
    metric.events += events;
    metric.total += value;
    metric.minimum = std::min(metric.minimum, value);
    metric.maximum = std::max(metric.maximum, value);
    metric.sequence = sequence;
    if (metric.samples.size() == capacity_) metric.samples.pop_front();
    metric.samples.push_back({value, events, sample_ns,
        metadata.empty() ? nullptr : std::make_shared<const Metadata>(metadata)});
  }

  void duration(const std::string &name, Tick begin, Tick end, uint64_t events, uint64_t sequence) {
    observe(name, std::chrono::duration<double, std::milli>(end - begin).count(), events, sequence);
  }

 private:
  struct Sample {
    double value;
    uint64_t events;
    int64_t ns;
    std::shared_ptr<const Metadata> metadata;
  };

  struct Metric {
    std::string unit;
    std::deque<Sample> samples;
    uint64_t count{0}, events{0}, sequence{0};
    double total{0}, minimum{0}, maximum{0};
  };

  static double percentile(const std::vector<double> &values, double fraction) {
    const double position = (values.size() - 1) * fraction;
    const size_t lower = static_cast<size_t>(position);
    const size_t upper = std::min(lower + 1, values.size() - 1);
    return values[lower] + (values[upper] - values[lower]) * (position - lower);
  }

  void run() {
    std::unique_lock<std::mutex> guard(mutex_);
    while (!condition_.wait_for(guard, std::chrono::duration<double>(interval_),
                                [this] { return stopping_; })) {
      guard.unlock();
      flush(false);
      guard.lock();
    }
  }

  void flush(bool final) {
    std::map<std::string, Metric> metrics;
    const auto end = Clock::now();
    Tick start;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      metrics.swap(metrics_);
      start = start_;
      start_ = end;
    }
    std::ofstream out(path_, std::ios::app);
    if (!out) {
      ROS_ERROR_STREAM_THROTTLE(5.0, "Cannot write performance: " << path_);
      return;
    }
    const auto ns = [](Tick tick) {
      return std::chrono::duration_cast<std::chrono::nanoseconds>(tick.time_since_epoch()).count();
    };
    out << std::setprecision(15)
        << "{\"schema_version\":1,\"component\":\"" << component_ << "\",\"clock\":\"steady_clock\","
        << "\"interval_start_ns\":" << ns(start) << ",\"interval_end_ns\":" << ns(end)
        << ",\"interval_s\":" << std::chrono::duration<double>(end - start).count()
        << ",\"final\":" << (final ? "true" : "false")
        << ",\"capacity_per_metric\":" << capacity_ << ",\"counters\":{},\"metrics\":{";
    bool first = true;
    for (const auto &entry : metrics) {
      const auto &metric = entry.second;
      std::vector<double> values;
      values.reserve(metric.samples.size());
      uint64_t sample_events = 0;
      for (const auto &sample : metric.samples) {
        values.push_back(sample.value);
        sample_events += sample.events;
      }
      std::sort(values.begin(), values.end());
      if (!first) out << ',';
      first = false;
      out << '"' << entry.first << "\":{\"unit\":\"" << metric.unit
          << "\",\"count\":" << metric.count << ",\"events\":" << metric.events
          << ",\"sample_count\":" << values.size() << ",\"sample_events\":" << sample_events
          << ",\"overflow\":" << metric.count - values.size() << ",\"sum\":" << metric.total
          << ",\"mean\":" << metric.total / metric.count << ",\"min\":" << metric.minimum
          << ",\"max\":" << metric.maximum << ",\"p50\":" << percentile(values, .5)
          << ",\"p95\":" << percentile(values, .95) << ",\"p99\":" << percentile(values, .99)
          << ",\"percentile_scope\":\"recent_bounded_samples\",\"metadata_last\":{\""
          << (component_ == "adapter" ? "raw_sequence" : "sequence") << "\":"
          << metric.sequence << "},\"samples\":[";
      bool first_sample = true;
      for (const auto &sample : metric.samples) {
        if (!first_sample) out << ',';
        first_sample = false;
        out << '[' << sample.value << ',' << sample.events << ',' << sample.ns << ']';
      }
      out << ']';
      const bool have_metadata = std::any_of(metric.samples.begin(), metric.samples.end(),
          [](const Sample &sample) { return static_cast<bool>(sample.metadata); });
      if (have_metadata) {
        out << ",\"sample_metadata\":[";
        bool first_metadata = true;
        for (const auto &sample : metric.samples) {
          if (!first_metadata) out << ',';
          first_metadata = false;
          out << '{';
          bool first_field = true;
          if (sample.metadata) {
            for (const auto &field : *sample.metadata) {
              if (!first_field) out << ',';
              first_field = false;
              // Metadata keys are fixed metric field names; values retain integer ns precision.
              out << std::quoted(field.first) << ':' << field.second;
            }
          }
          out << '}';
        }
        out << ']';
      }
      out << '}';
    }
    out << "}}\n";
  }

  const std::string path_, component_;
  const double interval_;
  const size_t capacity_;
  Tick start_;
  std::mutex mutex_;
  std::condition_variable condition_;
  std::map<std::string, Metric> metrics_;
  bool stopping_{false};
  std::thread thread_;
};

}
