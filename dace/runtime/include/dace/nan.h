// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_NAN_H
#define __DACE_NAN_H

// Class to define a stateless NAN and related operators.
#include <limits>

namespace dace
{
    namespace math
    {
        //////////////////////////////////////////////////////
        // Defines a typeless Pi
        struct typeless_nan
        {
            DACE_CONSTEXPR DACE_HDFI typeless_nan() noexcept = default;
            DACE_CONSTEXPR DACE_HDFI typeless_nan(const typeless_nan&) noexcept = default;
            DACE_CONSTEXPR DACE_HDFI typeless_nan(typeless_nan&&) noexcept = default;
            DACE_HDFI ~typeless_nan() noexcept = default;

            DACE_CONSTEXPR DACE_HDFI typeless_nan& operator=(const typeless_nan&) noexcept = default;
            DACE_CONSTEXPR DACE_HDFI typeless_nan& operator=(typeless_nan&&) noexcept = default;

            operator int() const = delete;
            DACE_CONSTEXPR DACE_HDFI operator float() const
            {
                return std::numeric_limits<float>::quiet_NaN();
            }
            DACE_CONSTEXPR DACE_HDFI operator double() const
            {
                return std::numeric_limits<double>::quiet_NaN();
            }

#if !( defined(__CUDACC__) || defined(__HIPCC__) )
            //There is no long double on the GPU
            DACE_CONSTEXPR DACE_HDFI operator long double() const
            {
                return std::numeric_limits<long double>::quiet_NaN();
            }
#endif
            DACE_CONSTEXPR DACE_HDFI typeless_nan operator+() const
            {
                return typeless_nan{};
            }
            DACE_CONSTEXPR DACE_HDFI typeless_nan operator-() const
            {
                return typeless_nan{};
            }
        };

        template<typename T>
        DACE_CONSTEXPR DACE_HDFI std::enable_if_t<std::is_floating_point<T>::value || std::is_integral<T>::value, typeless_nan>
        operator*(const T&,  const typeless_nan&) noexcept { return typeless_nan{}; }

        template<typename T>
        DACE_CONSTEXPR DACE_HDFI std::enable_if_t<std::is_floating_point<T>::value || std::is_integral<T>::value, typeless_nan>
        operator*(const typeless_nan&,  const T&) noexcept { return typeless_nan{}; }

        DACE_CONSTEXPR DACE_HDFI typeless_nan
        operator*(const typeless_nan&,  const typeless_nan&) noexcept { return typeless_nan{}; }


        template<typename T>
        DACE_CONSTEXPR DACE_HDFI std::enable_if_t<std::is_floating_point<T>::value || std::is_integral<T>::value, typeless_nan>
        operator+(const T&,  const typeless_nan&) noexcept { return typeless_nan{}; }

        template<typename T>
        DACE_CONSTEXPR DACE_HDFI std::enable_if_t<std::is_floating_point<T>::value || std::is_integral<T>::value, typeless_nan>
        operator+(const typeless_nan&,  const T&) noexcept { return typeless_nan{}; }

        DACE_CONSTEXPR DACE_HDFI typeless_nan
        operator+(const typeless_nan&,  const typeless_nan&) noexcept { return typeless_nan{}; }


        template<typename T>
        DACE_CONSTEXPR DACE_HDFI std::enable_if_t<std::is_floating_point<T>::value || std::is_integral<T>::value, typeless_nan>
        operator-(const T&,  const typeless_nan&) noexcept { return typeless_nan{}; }

        template<typename T>
        DACE_CONSTEXPR DACE_HDFI std::enable_if_t<std::is_floating_point<T>::value || std::is_integral<T>::value, typeless_nan>
        operator-(const typeless_nan&,  const T&) noexcept { return typeless_nan{}; }

        DACE_CONSTEXPR DACE_HDFI typeless_nan
        operator-(const typeless_nan&,  const typeless_nan&) noexcept { return typeless_nan{}; }


        template<typename T>
        DACE_CONSTEXPR DACE_HDFI std::enable_if_t<std::is_floating_point<T>::value || std::is_integral<T>::value, typeless_nan>
        operator/(const T&,  const typeless_nan&) noexcept { return typeless_nan{}; }

        template<typename T>
        DACE_CONSTEXPR DACE_HDFI std::enable_if_t<std::is_floating_point<T>::value || std::is_integral<T>::value, typeless_nan>
        operator/(const typeless_nan&,  const T&) noexcept { return typeless_nan{}; }

        DACE_CONSTEXPR DACE_HDFI typeless_nan
        operator/(const typeless_nan&,  const typeless_nan&) noexcept { return typeless_nan{}; }


        template<typename T>
        DACE_CONSTEXPR DACE_HDFI std::enable_if_t<std::is_floating_point<T>::value || std::is_integral<T>::value, typeless_nan>
        operator%(const T&,  const typeless_nan&) noexcept { return typeless_nan{}; }

        template<typename T>
        DACE_CONSTEXPR DACE_HDFI std::enable_if_t<std::is_floating_point<T>::value || std::is_integral<T>::value, typeless_nan>
        operator%(const typeless_nan&,  const T&) noexcept { return typeless_nan{}; }

        DACE_CONSTEXPR DACE_HDFI typeless_nan
        operator%(const typeless_nan&,  const typeless_nan&) noexcept { return typeless_nan{}; }

        DACE_HDFI typeless_nan ipow(const typeless_nan&, const unsigned int&) {
            return typeless_nan{};
        }

	//These functions allows to perfrom operations with `typeless_nan` instances.
#	define FADAPT(F) DACE_CONSTEXPR DACE_HDFI typeless_nan F (const typeless_nan&) noexcept { return typeless_nan{}; }
#	define FADAPT2(F) template<typename T1> DACE_CONSTEXPR DACE_HDFI typeless_nan F (T1&&, dace::math::typeless_nan) noexcept { return typeless_nan{}; }; \
			  template<typename T2> DACE_CONSTEXPR DACE_HDFI typeless_nan F (const typeless_nan&, T2&&) noexcept { return typeless_nan{}; }; \
			  DACE_CONSTEXPR DACE_HDFI typeless_nan F (const typeless_nan&, const typeless_nan&) noexcept { return typeless_nan{}; }
        FADAPT(tanh); FADAPT(cos); FADAPT(sin); FADAPT(sqrt); FADAPT(tan);
        FADAPT(acos); FADAPT(asin); FADAPT(atan); FADAPT(log); FADAPT(exp);
        FADAPT(floor); FADAPT(ceil); FADAPT(round); FADAPT(abs);
        FADAPT2(max); FADAPT2(min);
#       undef FADAPT2
#	undef FADAPT
    }
}


#endif  // __DACE_NAN_H
