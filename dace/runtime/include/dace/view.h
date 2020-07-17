#ifndef __DACE_VIEW_H
#define __DACE_VIEW_H

#include <cstdint>

#include "types.h"
#include "vector.h"
#include "reduction.h"

void __dace_materialize(const char* arrayname, int start, int end,
                        void* outarray);
void __dace_serialize(const char* arrayname, int start, int end,
                      const void* outarray);

// ADVICE:
// Be aware that there are specialized versions of ArrayView and ArrayView
// Immaterial _below_.

namespace dace {

    template <typename T, uint8_t DIMS, int VECTOR_LEN = 1,
        int NUM_ACCESSES = static_cast<int>(NA_RUNTIME), bool ALIGNED = false,
        typename OffsetT = int32_t>
    class ArrayViewIn
    {
    protected:

     template <int VECTOR_LEN_OTHER>
     using vec_other_t = typename std::conditional<ALIGNED, vec<T, VECTOR_LEN>,
                                                   vecu<T, VECTOR_LEN>>::type;

     template <int VECTOR_LEN_OTHER>
     using vecu_other_t = typename std::conditional<false, vec<T, VECTOR_LEN>,
                                                    vecu<T, VECTOR_LEN>>::type;

     // The internal pointer type relies on the alignment of the original array
     using vec_t = typename std::conditional<ALIGNED, vec<T, VECTOR_LEN>,
                                             vecu<T, VECTOR_LEN>>::type;

     T const* m_ptr;
     OffsetT m_stride[DIMS];

    public:
        template <typename... Dim>
        explicit DACE_HDFI ArrayViewIn(T const* ptr, const Dim&... strides) : 
            m_ptr(ptr) {
            static_assert(sizeof...(strides) == static_cast<int>(DIMS),
                          "Dimension mismatch");
            OffsetT stridearr[] = { static_cast<OffsetT>(strides)... };
            for (int i = 0; i < DIMS; ++i)
                m_stride[i] = stridearr[i];
        }

        template <typename... Dim>
        DACE_HDFI const vec_t& operator()(const Dim&... indices) const {
            static_assert(sizeof...(indices) == DIMS, "Dimension mismatch");
            OffsetT index_array[] = { static_cast<OffsetT>(indices)... };

            return get_element(index_array);
        }

        template <int VECTOR_LEN_OTHER = VECTOR_LEN>
        DACE_HDFI vecu_other_t<VECTOR_LEN_OTHER> const* ptr(
            T const* _ptr) const {
            return reinterpret_cast<vecu_other_t<VECTOR_LEN_OTHER> const*>(
                _ptr);
        }

        template <int VECTOR_LEN_OTHER = VECTOR_LEN>
        DACE_HDFI vecu_other_t<VECTOR_LEN_OTHER> const* ptr() const {
            return ptr<VECTOR_LEN_OTHER>(m_ptr);
        }

        template <int VECTOR_LEN_OTHER = VECTOR_LEN>
        DACE_HDFI vec_other_t<VECTOR_LEN_OTHER> const& ref() const {
            return *ptr<VECTOR_LEN_OTHER>();
        }

        template <int VECTOR_LEN_OTHER = VECTOR_LEN>
        DACE_HDFI vec_other_t<VECTOR_LEN_OTHER> val() const {
            return *ptr<VECTOR_LEN_OTHER>();
        }

        // template <typename... Dim>
        // DACE_HDFI T const* ptr_at(const Dim&... indices) const {
        //     static_assert(sizeof...(indices) == DIMS, "Dimension mismatch");
        //     OffsetT index_array[] = { static_cast<OffsetT>(indices)... };
        //     OffsetT offset;
        //     get_offset(index_array, offset);
        //     return m_ptr + offset;
        // }

    protected:
        DACE_HDFI void get_offset(OffsetT(&index_array)[DIMS],
                                  OffsetT& offset) const {
            offset = 0;
            for (int i = 0; i < DIMS - 1; ++i) {
                offset += index_array[i] * m_stride[i];
            }
            offset += index_array[DIMS - 1] * m_stride[DIMS - 1] * VECTOR_LEN;
        }

        DACE_HDFI const vec_t& get_element(OffsetT(&index_array)[DIMS]) const {
            OffsetT offset;
            get_offset(index_array, offset);
            return *ptr(m_ptr + offset);
        }
    };
    
    template <typename T, uint8_t DIMS, int VECTOR_LEN = 1,
        int NUM_ACCESSES = static_cast<int>(NA_RUNTIME), bool ALIGNED = false,
        typename OffsetT = int32_t>
    class ArrayViewOut
    {
    protected:
        // The internal pointer type relies on the alignment of the original array
        using vec_t = typename std::conditional<ALIGNED, vec<T, VECTOR_LEN>,
            vecu<T, VECTOR_LEN>>::type;

        template <int VECTOR_LEN_OTHER>
        using vecu_other_t = vecu<T, VECTOR_LEN_OTHER>;

        T* m_ptr;
        OffsetT m_stride[DIMS];

    public:
        template <typename... Dim>
        explicit DACE_HDFI ArrayViewOut(T* ptr, const Dim&... strides) : 
            m_ptr(ptr) {
            static_assert(sizeof...(strides) == static_cast<int>(DIMS),
                          "Dimension mismatch");
            OffsetT stridearr[] = { static_cast<OffsetT>(strides)... };
            for (int i = 0; i < DIMS; ++i)
                m_stride[i] = stridearr[i];
        }

        template <typename... Dim>
        DACE_HDFI const vec_t& operator()(const Dim&... indices) const {
            static_assert(sizeof...(indices) == DIMS, "Dimension mismatch");
            OffsetT index_array[] = { static_cast<OffsetT>(indices)... };

            return get_element(index_array);
        }

        template <int VECTOR_LEN_OTHER = VECTOR_LEN>
        DACE_HDFI vecu_other_t<VECTOR_LEN_OTHER> const* ptr(
            T const* _ptr) const {
          if (VECTOR_LEN_OTHER == VECTOR_LEN) {
            return reinterpret_cast<vecu_other_t<VECTOR_LEN_OTHER> const*>(
                _ptr);
          }
        }

        template <int VECTOR_LEN_OTHER = VECTOR_LEN>
        DACE_HDFI vecu_other_t<VECTOR_LEN_OTHER>* ptr(T *_ptr) const {
          if (VECTOR_LEN_OTHER == VECTOR_LEN) {
            return reinterpret_cast<vecu_other_t<VECTOR_LEN_OTHER>*>(_ptr);
          }
        }

        template <int VECTOR_LEN_OTHER = VECTOR_LEN>
        DACE_HDFI vecu_other_t<VECTOR_LEN_OTHER> const* ptr() const {
          if (VECTOR_LEN_OTHER == VECTOR_LEN) {
              return ptr<VECTOR_LEN_OTHER>(m_ptr);
          }
        }

        template <int VECTOR_LEN_OTHER = VECTOR_LEN>
        DACE_HDFI vecu_other_t<VECTOR_LEN_OTHER>* ptr() {
          if (VECTOR_LEN_OTHER == VECTOR_LEN) {
            return ptr<VECTOR_LEN_OTHER>(m_ptr);
          }
        }

        template <int VECTOR_LEN_OTHER = VECTOR_LEN>
        DACE_HDFI vecu_other_t<VECTOR_LEN_OTHER> const& ref() const {
            return *ptr<VECTOR_LEN_OTHER>();
        }

        template <int VECTOR_LEN_OTHER = VECTOR_LEN>
        DACE_HDFI vecu_other_t<VECTOR_LEN_OTHER> & ref() {
            return *ptr<VECTOR_LEN_OTHER>();
        }

        template <int VECTOR_LEN_OTHER = VECTOR_LEN>
        DACE_HDFI vecu_other_t<VECTOR_LEN_OTHER> val() const {
            return *ptr<VECTOR_LEN_OTHER>();
        }

        // template <typename... Dim>
        // DACE_HDFI T const* ptr_at(const Dim&... indices) const {
        //     static_assert(sizeof...(indices) == DIMS, "Dimension mismatch");
        //     OffsetT index_array[] = { static_cast<OffsetT>(indices)... };
        //     OffsetT offset;
        //     get_offset(index_array, offset);
        //     return m_ptr + offset;
        // }

        // template <typename... Dim>
        // DACE_HDFI T* ptr_at(const Dim&... indices) {
        //     static_assert(sizeof...(indices) == DIMS, "Dimension mismatch");
        //     OffsetT index_array[] = { static_cast<OffsetT>(indices)... };
        //     OffsetT offset;
        //     get_offset(index_array, offset);
        //     return m_ptr + offset;
        // }

        template <typename... Dim>
        DACE_HDFI void write(const vec_t& value, const Dim&... indices) {
            static_assert(sizeof...(indices) == DIMS, "Dimension mismatch");
            OffsetT index_array[] = { static_cast<OffsetT>(indices)... };

            set_element(value, index_array);
        }

        template <typename CONFLICT_RESOLUTION, typename... Dim>
        DACE_HDFI vec_t write_and_resolve(const vec_t& value, CONFLICT_RESOLUTION wcr,
                                         const Dim&... indices) {
            static_assert(sizeof...(indices) == DIMS, "Dimension mismatch");
            OffsetT index_array[] = { static_cast<OffsetT>(indices)... };

            return set_element_wcr(value, index_array, wcr);
        }

        template <typename CONFLICT_RESOLUTION, typename... Dim>
        DACE_HDFI vec_t write_and_resolve_nc(const vec_t& value,
                                            CONFLICT_RESOLUTION wcr,
                                            const Dim&... indices) {
            static_assert(sizeof...(indices) == DIMS, "Dimension mismatch");
            OffsetT index_array[] = { static_cast<OffsetT>(indices)... };

            return set_element_wcr_nc(value, index_array, wcr);
        }

        template <ReductionType REDT, typename... Dim>
        DACE_HDFI vec_t write_and_resolve(const vec_t& value,
                                         const Dim&... indices) {
            static_assert(sizeof...(indices) == DIMS, "Dimension mismatch");
            OffsetT index_array[] = { static_cast<OffsetT>(indices)... };

            return set_element_wcr<REDT>(value, index_array);
        }

        template <ReductionType REDT, typename... Dim>
        DACE_HDFI vec_t write_and_resolve_nc(const vec_t& value,
                                            const Dim&... indices) {
            static_assert(sizeof...(indices) == DIMS, "Dimension mismatch");
            OffsetT index_array[] = { static_cast<OffsetT>(indices)... };

            return set_element_wcr_nc<REDT>(value, index_array);
        }

    protected:
        DACE_HDFI void get_offset(OffsetT(&index_array)[DIMS],
                                  OffsetT& offset) const {
            offset = 0;
            for (int i = 0; i < DIMS - 1; ++i) {
                offset += index_array[i] * m_stride[i];
            }
            offset += index_array[DIMS - 1] * m_stride[DIMS - 1] * VECTOR_LEN;
        }

        DACE_HDFI const vec_t& get_element(OffsetT(&index_array)[DIMS]) const {
            OffsetT offset;
            get_offset(index_array, offset);
            return *ptr(m_ptr + offset);
        }

        DACE_HDFI void set_element(const vec_t& value, OffsetT(&index_array)[DIMS]) {
            OffsetT offset;
            get_offset(index_array, offset);
            *ptr(m_ptr + offset) = value;
        }

        template <ReductionType REDT>
        DACE_HDFI vec_t set_element_wcr(const vec_t& value,
                                       OffsetT(&index_array)[DIMS]) {
            OffsetT offset;
            get_offset(index_array, offset);

            return wcr_fixed<REDT, vec_t>::reduce_atomic(ptr<VECTOR_LEN>(
                m_ptr + offset), value);
        }

        template <ReductionType REDT>
        DACE_HDFI vec_t set_element_wcr_nc(const vec_t& value,
                                          OffsetT(&index_array)[DIMS]) {
            OffsetT offset;
            get_offset(index_array, offset);

            return wcr_fixed<REDT, vec_t>::reduce(ptr(m_ptr + offset), value);
        }

        template <typename WCR_T>
        DACE_HDFI vec_t set_element_wcr(const vec_t& value,
                                       OffsetT(&index_array)[DIMS], WCR_T wcr) {
            OffsetT offset;
            get_offset(index_array, offset);

            return wcr_custom<vec_t>::template reduce_atomic(
                wcr, ptr(m_ptr + offset), value);
        }

        template <typename WCR_T>
        DACE_HDFI vec_t set_element_wcr_nc(const vec_t& value,
                                          OffsetT(&index_array)[DIMS], WCR_T wcr) {
            OffsetT offset;
            get_offset(index_array, offset);

            return wcr_custom<vec_t>::template reduce(
                wcr, ptr(m_ptr + offset), value);
        }
    };

    // Scalar version
    template <typename T, int VECTOR_LEN, int NUM_ACCESSES, bool ALIGNED,
        typename OffsetT>
        class ArrayViewIn<T, 0, VECTOR_LEN, NUM_ACCESSES, ALIGNED, OffsetT> {
        protected:
            // The internal pointer type relies on the alignment of the original array
            // using vec_t = vec<T, VECTOR_LEN>;
            using vec_t = vecu<T, VECTOR_LEN>;

            T const *m_ptr;

        public:

            explicit DACE_HDFI ArrayViewIn(T const *ptr) : m_ptr(ptr) {}

            // Template on int to conform to the same interface as non-scalar,
            // but only accept the native vector length
            template <int VECTOR_LEN_OTHER = VECTOR_LEN>
            DACE_HDFI vec_t const* ptr(T const *_ptr) const {
                return reinterpret_cast<vec_t const*>(_ptr);
            }

            template <int VECTOR_LEN_OTHER = VECTOR_LEN>
            DACE_HDFI vec_t const* ptr() const {
                return ptr<VECTOR_LEN_OTHER>(m_ptr);
            }

            template <int VECTOR_LEN_OTHER = VECTOR_LEN>
            DACE_HDFI vec_t const& ref() const {
                return *ptr<VECTOR_LEN_OTHER>();
            }

            template <int VECTOR_LEN_OTHER = VECTOR_LEN>
            DACE_HDFI vec_t val() const {
                return *ptr<VECTOR_LEN_OTHER>();
            }

            DACE_HDFI operator vec_t() const {
                return val<VECTOR_LEN>();
            }
    };

    // Scalar version
    template <typename T, int VECTOR_LEN, int NUM_ACCESSES, bool ALIGNED,
        typename OffsetT>
        class ArrayViewOut<T, 0, VECTOR_LEN, NUM_ACCESSES, ALIGNED, OffsetT> {
        protected:
            // The internal pointer type relies on the alignment of the original array
            // using vec_t = vec<T, VECTOR_LEN>;
            using vec_t = vecu<T, VECTOR_LEN>;

            T* m_ptr;

        public:
            explicit DACE_HDFI ArrayViewOut(T* ptr) : m_ptr(ptr) {}

            // Template on int to conform to the same interface as non-scalar,
            // but only accept the native vector length
            template <int VECTOR_LEN_OTHER = VECTOR_LEN>
            DACE_HDFI vec_t const* ptr(T const *_ptr) const {
                return reinterpret_cast<vec_t const*>(_ptr);
            }

            template <int VECTOR_LEN_OTHER = VECTOR_LEN>
            DACE_HDFI vec_t *ptr(T *_ptr) const {
                return reinterpret_cast<vec_t *>(_ptr);
            }

            template <int VECTOR_LEN_OTHER = VECTOR_LEN>
            DACE_HDFI vec_t const* ptr() const {
                return ptr<VECTOR_LEN_OTHER>(m_ptr);
            }

            template <int VECTOR_LEN_OTHER = VECTOR_LEN>
            DACE_HDFI vec_t *ptr() {
                return ptr<VECTOR_LEN_OTHER>(m_ptr);
            }

            template <int VECTOR_LEN_OTHER = VECTOR_LEN>
            DACE_HDFI vec_t const& ref() const {
                return *ptr<VECTOR_LEN_OTHER>();
            }

            template <int VECTOR_LEN_OTHER = VECTOR_LEN>
            DACE_HDFI vec_t &ref() {
                return *ptr<VECTOR_LEN_OTHER>();
            }

            template <int VECTOR_LEN_OTHER = VECTOR_LEN>
            DACE_HDFI vec_t val() const {
                return *ptr<VECTOR_LEN_OTHER>();
            }

            DACE_HDFI operator vec_t() const {
                return val<VECTOR_LEN>();
            }

            DACE_HDFI void write(const vec_t& value) {
                *ptr<VECTOR_LEN>() = value;
            }

            template <typename CONFLICT_RESOLUTION>
            DACE_HDFI vec_t write_and_resolve(const vec_t& value,
                                             CONFLICT_RESOLUTION wcr) {
                return wcr_custom<vec_t>::reduce_atomic(
                    wcr, ptr<VECTOR_LEN>(), value);
            }

            template <typename CONFLICT_RESOLUTION>
            DACE_HDFI vec_t write_and_resolve_nc(const vec_t& value,
                                             CONFLICT_RESOLUTION wcr) {
                return wcr_custom<vec_t>::template reduce(
                    wcr, ptr<VECTOR_LEN>(), value);
            }

            template <ReductionType REDT>
            DACE_HDFI vec_t write_and_resolve(const vec_t& value) {
                return wcr_fixed<REDT, vec_t>::reduce_atomic(
                    ptr<VECTOR_LEN>(), value);
            }

            template <ReductionType REDT>
            DACE_HDFI vec_t write_and_resolve_nc(const vec_t& value) {
                return wcr_fixed<REDT, vec_t>::reduce(ptr<VECTOR_LEN>(), value);
            }

            // Special case for vector conditionals
#define VECTOR_CONDITIONAL_WRITE_AND_RESOLVE(N)                             \
            template <typename CONFLICT_RESOLUTION>                         \
            DACE_HDFI vec<int, N> write_and_resolve(const vec<int, N>& value,      \
                                             CONFLICT_RESOLUTION wcr) {     \
                int ppcnt = 0;                                              \
                for (int v = 0; v < N; ++v) ppcnt += value[v] ? 1 : 0;      \
                return write_and_resolve(ppcnt)                             \
            }                                                               \
            template <ReductionType REDT>                                 \
            DACE_HDFI vec<int, N> write_and_resolve(const vec<int, N>& value) {    \
                int ppcnt = 0;                                              \
                for (int v = 0; v < N; ++v) ppcnt += value[v] ? 1 : 0;      \
                return write_and_resolve<REDT>(ppcnt)                       \
            }                                                               \
            template <typename CONFLICT_RESOLUTION>                         \
            DACE_HDFI vec<int, N> write_and_resolve_nc(const vec<int, N>& value,   \
                                                CONFLICT_RESOLUTION wcr) {  \
                int ppcnt = 0;                                              \
                for (int v = 0; v < N; ++v) ppcnt += value[v] ? 1 : 0;      \
                return write_and_resolve_nc(ppcnt)                          \
            }                                                               \
            template <ReductionType REDT>                                 \
            DACE_HDFI vec<int, N> write_and_resolve_nc(const vec<int, N>& value) { \
                int ppcnt = 0;                                              \
                for (int v = 0; v < N; ++v) ppcnt += value[v] ? 1 : 0;      \
                return write_and_resolve_nc<REDT>(ppcnt)                    \
            }

            //VECTOR_CONDITIONAL_WRITE_AND_RESOLVE(2)
            //VECTOR_CONDITIONAL_WRITE_AND_RESOLVE(4)
            //VECTOR_CONDITIONAL_WRITE_AND_RESOLVE(8)
    };

    template <typename T, int VECTOR_LEN, typename OffsetT, int... DIMS>
    class ArrayViewImmaterialIn  // No skip version
    {
    protected:
        enum {
            NDIMS = sizeof...(DIMS),
            TOTAL_SIZE = TotalNDSize<DIMS...>::value,
        };

        // The internal pointer type relies on the alignment of the original array
        using vec_t = vecu<T, VECTOR_LEN>;

        T m_local_data[TOTAL_SIZE];
        // OffsetT m_stride[NDIMS];

    public:
        template <typename... Dim>
        explicit DACE_HDFI ArrayViewImmaterialIn(const char* name,
                                                 const Dim&... dims) {
            static_assert(sizeof...(dims) == 2 * static_cast<int>(NDIMS),
                          "Dimension mismatch");
            OffsetT dim_array[] = { static_cast<OffsetT>(dims)... };
            OffsetT global_offset = 0;
            OffsetT cur_stride = 1;

            for (int i = 0; i < NDIMS; ++i) {
                global_offset += dim_array[2 * i] * cur_stride;
                // m_stride[i] = dim_array[2*i+1];
                cur_stride *= dim_array[2 * i + 1];
            }
            // TODO: Assuming contiguous memory regions for now.
            // Change to ND regions later
            __dace_materialize(name, global_offset, global_offset + TOTAL_SIZE,
                               m_local_data);
        }

        template <typename... Dim>
        DACE_HDFI const vec_t& operator()(const Dim&... indices) const {
            static_assert(sizeof...(indices) == NDIMS, "Dimension mismatch");
            OffsetT index_array[] = { static_cast<OffsetT>(indices)... };

            return get_element(index_array);
        }

    protected:
        // TODO: Only supports 1D, implement ND later
        /*
            DACE_HDFI void get_offset(OffsetT(&index_array)[NDIMS], OffsetT& offset)
           const
            {
                OffsetT mult = 1;//m_stride[NDIMS - 1];
                offset = 0;
                for (int8_t i = NDIMS - 2; i >= 0; --i)
                {

                }
                offset += (m_off[NDIMS - 1] + index_array[NDIMS - 1]) * VECTOR_LEN;
            }*/

        DACE_HDFI const vec_t& get_element(OffsetT(&index_array)[NDIMS]) const {
            OffsetT offset = index_array[0];
            // get_offset(index_array, offset);

            return *reinterpret_cast<vec_t const*>(m_local_data + offset);
        }
    };

    template <typename T, int VECTOR_LEN, typename OffsetT, int... DIMS>
    class ArrayViewImmaterialOut  // No skip version
    {
    protected:
        enum {
            NDIMS = sizeof...(DIMS),
            TOTAL_SIZE = TotalNDSize<DIMS...>::value,
        };

        // The internal pointer type relies on the alignment of the original array
        using vec_t = vecu<T, VECTOR_LEN>;

        T m_local_data[TOTAL_SIZE];
        // OffsetT m_stride[NDIMS];
        const char* m_name;
        OffsetT m_global_offset;

    public:
        template <typename... Dim>
        explicit DACE_HDFI ArrayViewImmaterialOut(const char* name,
                                                  const Dim&... dims)
            : m_name(name) {
            static_assert(sizeof...(dims) == 2 * static_cast<int>(NDIMS),
                          "Dimension mismatch");
            OffsetT dim_array[] = { static_cast<OffsetT>(dims)... };
            OffsetT global_offset = 0;
            OffsetT cur_stride = 1;

            for (int i = 0; i < NDIMS; ++i) {
                global_offset += dim_array[2 * i] * cur_stride;
                // m_stride[i] = dim_array[2*i+1];
                cur_stride *= dim_array[2 * i + 1];
            }
            m_global_offset = global_offset;
        }

        DACE_HDFI ~ArrayViewImmaterialOut() {
            __dace_serialize(m_name, m_global_offset, m_global_offset + TOTAL_SIZE,
                             m_local_data);
        }

        template <typename... Dim>
        DACE_HDFI void write(const vec_t& value, const Dim&... indices) {
            static_assert(sizeof...(indices) == NDIMS, "Dimension mismatch");
            OffsetT index_array[] = { static_cast<OffsetT>(indices)... };

            set_element(value, index_array);
        }

    protected:
        // TODO: Only supports 1D, implement ND later
        /*
            DACE_HDFI void get_offset(OffsetT(&index_array)[NDIMS], OffsetT& offset)
           const
            {
                OffsetT mult = 1;//m_stride[NDIMS - 1];
                offset = 0;
                for (int8_t i = NDIMS - 2; i >= 0; --i)
                {

                }
                offset += (m_off[NDIMS - 1] + index_array[NDIMS - 1]) * VECTOR_LEN;
            }*/

        DACE_HDFI void set_element(const vec_t& value,
                                   OffsetT(&index_array)[NDIMS]) {
            OffsetT offset = index_array[0];
            // get_offset(index_array, offset);

            *reinterpret_cast<vec_t*>(m_local_data + offset) = value;
        }
    };

    // Scalar version
    template <typename T, int VECTOR_LEN, typename OffsetT>
    class ArrayViewImmaterialIn<T, VECTOR_LEN, OffsetT> {
    protected:
        using vec_t = vecu<T, VECTOR_LEN>;

        const char* m_name;
        OffsetT m_offset;

    public:
        explicit DACE_HDFI ArrayViewImmaterialIn(const char* name, OffsetT offset)
            : m_name(name), m_offset(offset) {}

        DACE_HDFI operator vec_t() {
            vec_t tmp;
            __dace_materialize(m_name, m_offset, m_offset + 1, &tmp);
            return tmp;
        }
    };

    // Scalar version
    template <typename T, int VECTOR_LEN, typename OffsetT>
    class ArrayViewImmaterialOut<T, VECTOR_LEN, OffsetT> {
    protected:
        using vec_t = vecu<T, VECTOR_LEN>;

        const char* m_name;
        OffsetT m_offset;

    public:
        explicit DACE_HDFI ArrayViewImmaterialOut(const char* name, OffsetT offset)
            : m_name(name), m_offset(offset) {}

        DACE_HDFI void write(const vec_t& value) {
            __dace_serialize(m_name, m_offset, m_offset + 1, &value);
        }

        DACE_HDFI void write(const vec_t& value, OffsetT offset) {
            __dace_serialize(m_name, m_offset + offset, m_offset + 1 + offset, &value);
        }
    };

}  // namespace dace

#endif  // __DACE_VIEW_H
