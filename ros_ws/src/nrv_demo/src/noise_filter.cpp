#include <cstddef>
#include <cstdint>

// C ABI for NumPy buffers: preserve event order and history across ROS batches.
extern "C" void nrv_filter(const intptr_t* x, const intptr_t* y, const double* ts,
                          size_t count, int width, int height, double* seen,
                          double* accepted, bool background, double window,
                          bool refractory, double interval, uint8_t* keep) {
  for (size_t i = 0; i < count; ++i) {
    const int px = x[i], py = y[i];
    const size_t index = static_cast<size_t>(py) * width + px;
    bool supported = !background;
    if (background) {
      for (int dy = -1; dy <= 1 && !supported; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          const int nx = px + dx, ny = py + dy;
          if ((!dx && !dy) || nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
          const double age = ts[i] - seen[static_cast<size_t>(ny) * width + nx];
          if (age >= 0 && age <= window) { supported = true; break; }
        }
      }
    }
    keep[i] = supported && (!refractory || ts[i] - accepted[index] >= interval);
    seen[index] = ts[i];
    if (keep[i]) accepted[index] = ts[i];
  }
}
