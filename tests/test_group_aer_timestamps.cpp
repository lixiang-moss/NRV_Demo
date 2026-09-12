// Focused tests for the pinned group_aer header and its build-time patch.
// No ROS master, camera, GUI or GPU is required.
// Compile with g++ -std=c++14 -O2 -I/opt/ros/noetic/include and this source file.
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <event_camera_codecs/group_aer_decoder.h>

struct Message { std::vector<uint8_t> events; uint64_t time_base{0}; };
struct Event { uint64_t t; uint16_t x, y; uint8_t p; };
struct Processor {
  bool retain{true};
  std::vector<Event> events;
  uint64_t n{0}, negative{0}, repeated{0}, positive_over_300ms{0};
  uint64_t first{0}, previous{0}, max_forward{0}, max_backward{0};
  uint64_t xyp_hash{14695981039346656037ULL};
  void eventCD(uint64_t t, uint16_t x, uint16_t y, uint8_t p) {
    if (n) {
      if (t < previous) { ++negative; max_backward = std::max(max_backward, previous-t); }
      else {
        const auto delta = t-previous;
        repeated += delta == 0;
        positive_over_300ms += delta > 300000000ULL;
        max_forward = std::max(max_forward, delta);
      }
    } else first = t;
    previous = t;
    ++n;
    for (const uint8_t byte : {uint8_t(x >> 8), uint8_t(x), uint8_t(y >> 8), uint8_t(y), p})
      xyp_hash = (xyp_hash ^ byte) * 1099511628211ULL;
    if (retain) events.push_back({t, x, y, p});
  }
  void finished() {}
};
using Decoder = event_camera_codecs::group_aer::Decoder<Message, Processor>;
constexpr uint32_t kFrame = 0x0c000000, kOneEvent = 0x80000001;
constexpr uint32_t kMaxRef = 0x3fffff;
constexpr uint64_t kPeriodUs = (uint64_t{kMaxRef}+1)*1000;
uint32_t ref(uint32_t value) { return 0x08000000 | value; }
uint32_t sub(uint32_t value) { return 0x08800000 | value; }
uint32_t col(uint32_t value) { return 0x04000000 | value; }
void require(bool value, const char *what) { if (!value) throw std::runtime_error(what); }
void feed(Decoder &d, Processor &p, std::initializer_list<uint32_t> words) {
  std::vector<uint8_t> bytes;
  for (const auto word : words)
    for (int shift=24; shift>=0; shift-=8) bytes.push_back(uint8_t(word >> shift));
  require(d.decode(bytes.data(), bytes.size(), &p) == bytes.size(), "incomplete word consumption");
}
void init(Decoder &d, Processor &p, uint32_t r, uint32_t s) {
  d.setGeometry(960, 720);
  d.setTimeMultiplier(1000);
  feed(d, p, {kFrame, ref(r), sub(s), col(7), kOneEvent});
}
void times(const Processor &p, std::initializer_list<uint64_t> expected_us) {
  require(p.events.size() == expected_us.size(), "event count changed");
  size_t i=0;
  for (const auto t : expected_us) {
    const auto &event=p.events[i++];
    require(event.t == t*1000, "unexpected decoded timestamp");
    require(event.x == 7 && event.y == 0 && event.p == 1, "x/y/p/order changed");
  }
}
void selfTest() {
  {
    Decoder d; Processor p; init(d,p,12,123);
    feed(d,p,{sub(123),kOneEvent,ref(12),sub(123),kOneEvent});
    times(p,{12123,12123,12123});
  }
  {
    Decoder d; Processor p; init(d,p,12,999);
    feed(d,p,{sub(2),kOneEvent});  // The reference word was omitted at this tick.
    feed(d,p,{sub(3),kOneEvent});  // State also survives a RAW packet boundary.
    times(p,{12999,13002,13003});
  }
  {
    Decoder d; Processor p; init(d,p,12,100);
    feed(d,p,{ref(1012),sub(200),kOneEvent});
    times(p,{12100,1012200});  // Preserve a normal one-second input gap.
  }
  {
    Decoder d; Processor p; init(d,p,10955,783);
    feed(d,p,{ref(3),sub(738),kOneEvent,sub(800),kOneEvent});
    times(p,{10955783,3738,3800});  // Actual recorded startup reset stays visible.
    require(p.negative == 1 && p.positive_over_300ms == 0, "reset hidden by a false rollover");
  }
  {
    Decoder d; Processor p; init(d,p,824,999);
    feed(d,p,{sub(194),kOneEvent,ref(824),sub(967),kOneEvent});
    times(p,{824999,825194,824967});  // Explicit reference overrides an inferred tick.
  }
  {
    Decoder d; Processor p; init(d,p,12,500);
    feed(d,p,{ref(12),sub(400),kOneEvent});
    times(p,{12500,12400});  // An explicit reference also disambiguates a lower sub.
  }
  {
    Decoder d; Processor p; init(d,p,kMaxRef,999);
    feed(d,p,{ref(0),sub(2),kOneEvent});
    times(p,{kPeriodUs-1,kPeriodUs+2});
  }
  {
    Decoder d; Processor p; init(d,p,kMaxRef,999);
    feed(d,p,{sub(2),kOneEvent,ref(0),sub(3),kOneEvent});
    times(p,{kPeriodUs-1,kPeriodUs+2,kPeriodUs+3});  // No double count of implicit wrap.
    feed(d,p,{ref(2),sub(4),kOneEvent,ref(1),sub(5),kOneEvent});
    times(p,{kPeriodUs-1,kPeriodUs+2,kPeriodUs+3,kPeriodUs+2004,1005});
  }
  {
    Decoder d; Processor p; init(d,p,kMaxRef,1023);
    times(p,{uint64_t{kMaxRef}*1000+1023});  // Integer composition includes all 10 bits.
  }
  {
    Decoder d; Processor p; init(d,p,12,123); p.events.clear();
    feed(d,p,{0x84020205,col(9),0x80010001});
    require(p.events.size()==4, "event masks changed");
    const uint16_t ys[]={0,2,9,0}; const uint16_t xs[]={7,7,7,9};
    const uint8_t ps[]={1,1,0,0};
    for (size_t i=0;i<4;++i)
      require(p.events[i].x==xs[i] && p.events[i].y==ys[i] && p.events[i].p==ps[i]
              && p.events[i].t==12123000, "event polarity/position/order changed");
  }
  std::cout << "PASS: 10 focused group_aer timestamp and event-order cases\n";
}
void audit(const char *path) {
  // Diagnostic input: consecutive [little-endian uint32 byte length, RAW bytes].
  // Produced directly from the bag without ROS transport or EventArray conversion.
  std::ifstream input(path, std::ios::binary);
  require(input.good(), "cannot open framed RAW input");
  Decoder d; Processor p; p.retain=false; d.setGeometry(960,720); d.setTimeMultiplier(1000);
  uint64_t batches=0;
  while (true) {
    uint8_t length[4]; input.read(reinterpret_cast<char *>(length), 4);
    if (input.gcount()==0 && input.eof()) break;
    require(input.gcount()==4, "truncated length");
    const uint32_t size=uint32_t(length[0]) | (uint32_t(length[1])<<8)
      | (uint32_t(length[2])<<16) | (uint32_t(length[3])<<24);
    std::vector<uint8_t> bytes(size);
    input.read(reinterpret_cast<char *>(bytes.data()),size);
    require(input.gcount()==size, "truncated RAW packet");
    require(d.decode(bytes.data(),bytes.size(),&p)==bytes.size(), "incomplete RAW decode");
    ++batches;
  }
  std::cout << "{\"raw_batches\":" << batches << ",\"events\":" << p.n
    << ",\"negative_deltas\":" << p.negative << ",\"repeated_timestamps\":" << p.repeated
    << ",\"positive_jumps_over_300ms\":" << p.positive_over_300ms
    << ",\"max_forward_ns\":" << p.max_forward << ",\"max_backward_ns\":" << p.max_backward
    << ",\"first_ns\":" << p.first << ",\"last_ns\":" << p.previous
    << ",\"xyp_order_hash\":\"" << p.xyp_hash << "\"}\n";
}
int main(int argc, char **argv) {
  try {
    if (argc==3 && std::string(argv[1])=="--raw") audit(argv[2]);
    else selfTest();
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n'; return 1;
  }
}
