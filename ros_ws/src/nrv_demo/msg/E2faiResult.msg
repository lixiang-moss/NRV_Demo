# Event times are adapter-mapped timestamps, not calibrated physical latency.
std_msgs/Header header
string session_id
uint64 window_id
uint64 source_batch_first
uint64 source_batch_last
uint64 window_start_ns
uint64 window_end_ns
uint64 event_first_ns
uint64 event_last_ns
uint64 event_count
uint32 width
uint32 height
string reset_reason
uint64 reset_count
# Catch-up generation within this acquisition session; older results are obsolete.
uint64 processing_generation
# Same-host monotonic times, never ROS or sensor timestamps.
uint64 source_callback_ns
uint64 completed_ns
sensor_msgs/Image gray
sensor_msgs/Image flow_preview
# Two channels: x,y; inference-grid pixels per input window.
sensor_msgs/Image flow
