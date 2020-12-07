// Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_INTSET_H
#define __DACE_INTSET_H

// Iterable integer sets for compiler inference and automatic unrolling

#include <array>
#include <tuple>

#include "types.h"

namespace dace
{

    template <int... ArgRanges>
    struct const_int_range;

    template <int RangeBegin, int RangeEnd, int RangeSkip>
    struct const_int_range<RangeBegin, RangeEnd, RangeSkip> {
        static constexpr size_t dims = 1;
        static constexpr int size = (RangeEnd - RangeBegin + RangeSkip - 1) / RangeSkip;

        static DACE_CONSTEXPR DACE_HDFI size_t len(size_t dim) {
            return (RangeEnd - RangeBegin + RangeSkip - 1) / RangeSkip;
        }
        static DACE_CONSTEXPR DACE_HDFI int index_value(const size_t range_value,
                                                        const size_t /*dimension*/) {
            return RangeBegin + (range_value % size) * RangeSkip;
        }
    };

    template <int RangeBegin, int RangeEnd, int RangeSkip, int... ArgRanges>
    struct const_int_range<RangeBegin, RangeEnd, RangeSkip, ArgRanges...> {
        static constexpr size_t dims = sizeof...(ArgRanges) / 3 + 1;
        static constexpr int size = const_int_range<ArgRanges...>::size *
            const_int_range<RangeBegin, RangeEnd, RangeSkip>::size;

        static DACE_CONSTEXPR DACE_HDFI size_t len(size_t dim) {
            const int _ranges[] = { RangeBegin, RangeEnd, RangeSkip, ArgRanges... };
            return (_ranges[3 * dim + 1] - _ranges[3 * dim] + _ranges[3 * dim + 2] - 1) / _ranges[3 * dim + 2];
        }

        static DACE_CONSTEXPR DACE_HDFI int index_value(const size_t range_value, const size_t dimension) {
            const int _ranges[] = { RangeBegin, RangeEnd, RangeSkip, ArgRanges... };
            if (dimension == dims - 1)
                return _ranges[3 * dimension] +
                (range_value % len(dimension)) * _ranges[3 * dimension + 2];
            auto value = range_value;
            for (auto dim = dimension + 1; dim < dims; ++dim) {
                value /= len(dim);
            }
            return _ranges[3 * dimension] +
                (value % len(dimension)) * _ranges[3 * dimension + 2];
        }

        static DACE_CONSTEXPR DACE_HDFI std::array<int, dims> index_values(const size_t range_value) {
            std::array<int, dims> values{};
            const int _ranges[] = { RangeBegin, RangeEnd, RangeSkip, ArgRanges... };
            auto value = range_value;
            for (int dim = dims - 1; dim >= 0; --dim) {
                values[dim] = _ranges[3 * dim] +
                    (value % len(dim)) * _ranges[3 * dim + 2];
                value /= len(dim);
            }
            return values;
        }
    };

    template <class... ArgRanges>
    class int_range {
        static constexpr size_t kDims = sizeof...(ArgRanges);

    private:
        const std::array<std::tuple<int, int, int>, kDims> _ranges;
        std::array<size_t, kDims> _range_lengths;
        const int _total_length;

        // For some reason constexpr works even when passing a runtime size
        DACE_HDFI int _calc_length() {
            // Hopefully the compiler vectorizes this
            size_t total_length = 1;
            for (size_t i = 0; i < kDims; ++i) {
                auto length = (std::get<1>(_ranges[i]) - std::get<0>(_ranges[i]) +
                               std::get<2>(_ranges[i]) - 1) /
                    std::get<2>(_ranges[i]);
                _range_lengths[i] = length;
                total_length *= length;
            }
            return total_length;
        }

    public:
        DACE_HDFI int_range(ArgRanges &&... ranges)
            : _ranges({ ranges... }), _total_length(_calc_length()) {
            // -std=c++1z
            // (_ranges.push_back(ranges), ...);
        }

        DACE_HDFI int size() const { return _total_length; }

        DACE_HDFI int index_value(const size_t range_value, 
                                  const size_t dimension) const {
            if (dimension == kDims - 1)
                return std::get<0>(_ranges[dimension]) +
                (range_value % _range_lengths[dimension]) *
                std::get<2>(_ranges[dimension]);
            auto value = range_value;
            for (auto dim = dimension + 1; dim < kDims; ++dim) {
                value /= _range_lengths[dim];
            }
            return std::get<0>(_ranges[dimension]) +
                (value % _range_lengths[dimension]) *
                std::get<2>(_ranges[dimension]);
        }
        DACE_HDFI std::array<int, kDims> index_values(
                const size_t range_value) const {
            std::array<int, kDims> values;
            auto value = range_value;
            for (int dim = kDims - 1; dim >= 0; --dim) {
                values[dim] = std::get<0>(_ranges[dim]) +
                    (value % _range_lengths[dim]) *
                    std::get<2>(_ranges[dim]);
                value /= _range_lengths[dim];
            }
            return values;
        }

    };

    template <class... ArgRanges>
    DACE_HDFI int_range<ArgRanges...> make_range(ArgRanges &&... ranges) {
        return int_range<ArgRanges...>(std::forward<ArgRanges>(ranges)...);
    }



}  // namespace dace

#endif  // __DACE_INTSET_H
