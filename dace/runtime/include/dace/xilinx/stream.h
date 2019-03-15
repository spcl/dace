#pragma once

#include "hlslib/xilinx/Stream.h"
#include "dace/xilinx/vec.h"
#ifndef DACE_SYNTHESIS
#include <string>  // std::to_string
#endif

namespace dace {

/// Proxy class that wraps hlslib::Stream in a dace::Stream-compatible
/// interface.
template <typename T, unsigned vector_length, unsigned capacity>
class FIFO {
 public:
  FIFO() : stream_(capacity) {
    #pragma HLS INLINE
    #pragma HLS STREAM variable=stream_ depth=capacity
  }
  FIFO(char const *const name) : stream_(name, capacity) {
    #pragma HLS INLINE
    #pragma HLS STREAM variable=stream_ depth=capacity
  }
  FIFO(FIFO const&) = delete;
  FIFO(FIFO&&) = delete;
  FIFO& operator=(FIFO const&) = delete;
  FIFO& operator=(FIFO&&) = delete;
  ~FIFO() = default;

  using Data_t = dace::vec<T, vector_length>;

  Data_t pop_blocking() {
    #pragma HLS INLINE
    return stream_.ReadBlocking();
  }

  Data_t pop() {
    #pragma HLS INLINE
    return pop_blocking();
  }

  bool pop_try(Data_t& output) {
    #pragma HLS INLINE
    return stream_.ReadNonBlocking(output);
  }

  template <typename U>
  void push_blocking(U&& val) {
    #pragma HLS INLINE
    return stream_.WriteBlocking(std::forward<U>(val));
  }

  template <typename U>
  void push(U&& val) {
    #pragma HLS INLINE
    return push_blocking(val);
  }

  // ArrayView-compatible interface

  template <typename U>
  void write(U&& val) {
    #pragma HLS INLINE
    return push(std::forward<U>(val));
  }

  template <typename U>
  void operator=(U&& val) {
    #pragma HLS INLINE
    push(std::forward<U>(val));
  }

  operator Data_t() { 
    #pragma HLS INLINE
    return pop_blocking();
  }

#ifndef DACE_SYNTHESIS
  void SetName(std::string const &str) {
    stream_.set_name(str.c_str());
  }
#endif

 private:
  hlslib::Stream<Data_t, capacity> stream_;
};

// DataView interface for streams
template <typename T, unsigned vector_length, unsigned capacity>
class StreamView {
 public:
  StreamView(FIFO<T, vector_length, capacity>& stream) : stream_(stream) {}
  StreamView(StreamView<T, vector_length, capacity> const &) = default;
  StreamView(StreamView<T, vector_length, capacity> &&) = default;
  StreamView() = delete;
  ~StreamView() = default;

  using Data_t = dace::vec<T, vector_length>;

  template <typename U>
  void write(U&& val) {
    #pragma HLS INLINE
    stream_.push_blocking(std::forward<U>(val));
  }

  template <typename U>
  void push(U&& val) {
    #pragma HLS INLINE
    write(std::forward<U>(val));
  }

  template <typename U>
  void operator=(U&& val) {
    #pragma HLS INLINE
    return write(std::forward<U>(val));
  }

  operator Data_t() {
    #pragma HLS INLINE
    return stream_.pop_blocking();
  }

  Data_t pop() {
    #pragma HLS INLINE
    return Data_t(*this);
  }

private:
  FIFO<T, vector_length, capacity> &stream_;
};

template <typename T, unsigned vector_length, unsigned capacity>
StreamView<T, vector_length, capacity> make_streamview(
    FIFO<T, vector_length, capacity>& stream) {
  #pragma HLS INLINE
  return StreamView<T, vector_length, capacity>(stream);
}

template <typename T, unsigned vector_length, unsigned capacity>
StreamView<T, vector_length, capacity>& make_streamview(
    StreamView<T, vector_length, capacity>& view) {
  #pragma HLS INLINE
  return view;
}

template <typename T, unsigned vector_length, unsigned capacity>
void SetNames(FIFO<T, vector_length, capacity> fifos[], char const *const str,
              const unsigned num) {
  #pragma HLS INLINE
#ifndef DACE_SYNTHESIS
  for (unsigned i = 0; i < num; ++i) {
    fifos[i].SetName(std::string(str) + "[" + std::to_string(i) + "]");
  }
#endif
}

}  // End namespace dace
