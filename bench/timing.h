// The two timing loops the benchmarks share: a fixed number of repetitions, or a time budget when a
// call is too slow to repeat often.
#pragma once

#include <chrono>
#include <cstdio>
#include <functional>
#include <string>

namespace bench
{

  /**
   * @brief Runs `call` once to fill the caches it fills, then `rounds` times, and prints the mean.
   * @param name What the line is about.
   * @param rounds How many measured calls.
   * @param call The measured call; its return value is discarded.
   */
  inline void repeat(const std::string &name, int64_t rounds, const std::function<void()> &call)
  {
    call();
    const auto start = std::chrono::steady_clock::now();
    for (int64_t at = 0; at < rounds; ++at) call();
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("  %-44s %11.3f us  (%lld calls)\n", name.c_str(),
                seconds * 1e6 / static_cast<double>(rounds), static_cast<long long>(rounds));
  }

  /**
   * @brief Runs `call` until at least `seconds` have passed, and prints the mean.
   * @param name What the line is about.
   * @param seconds The time the measurement takes at least, which has to be positive.
   * @param call The measured call; its return value is discarded.
   */
  inline void run_for(const std::string &name, double seconds, const std::function<void()> &call)
  {
    call();
    int64_t rounds = 0;
    const auto start = std::chrono::steady_clock::now();
    double spent = 0.0;
    do
    {
      call();
      ++rounds;
      spent = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    } while (spent < seconds);
    std::printf("  %-44s %11.3f us  (%lld calls)\n", name.c_str(), spent * 1e6 / static_cast<double>(rounds),
                static_cast<long long>(rounds));
  }

} // namespace bench
