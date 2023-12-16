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
            operator int() const = delete;
            operator float() const
            {
                return std::numeric_limits<float>::quiet_NaN();
            }
            operator double() const
            {
                return std::numeric_limits<double>::quiet_NaN();
            }
            operator long double() const
            {
                return std::numeric_limits<long double>::quiet_NaN();
            }
            typeless_nan operator+() const
            {
                return typeless_nan{};
            }
            typeless_nan operator-() const
            {
                return typeless_nan{};
            }
        };

        template<typename T>
        DACE_CONSTEXPR typename std::enable_if<std::is_floating_point<T>::value, typeless_nan>::type
        operator*(const T&,  const typeless_nan&) { return typeless_nan{}; }

        template<typename T>
        DACE_CONSTEXPR typename std::enable_if<std::is_floating_point<T>::value, typeless_nan>::type
        operator*(const typeless_nan&,  const T&) { return typeless_nan{}; }

        inline typeless_nan
        operator*(const typeless_nan&,  const typeless_nan&) { return typeless_nan{}; }


        template<typename T>
        DACE_CONSTEXPR typename std::enable_if<std::is_floating_point<T>::value, typeless_nan>::type
        operator+(const T&,  const typeless_nan&) { return typeless_nan{}; }

        template<typename T>
        DACE_CONSTEXPR typename std::enable_if<std::is_floating_point<T>::value, typeless_nan>::type
        operator+(const typeless_nan&,  const T&) { return typeless_nan{}; }

        inline typeless_nan
        operator+(const typeless_nan&,  const typeless_nan&) { return typeless_nan{}; }


        template<typename T>
        DACE_CONSTEXPR typename std::enable_if<std::is_floating_point<T>::value, typeless_nan>::type
        operator-(const T&,  const typeless_nan&) { return typeless_nan{}; }

        template<typename T>
        DACE_CONSTEXPR typename std::enable_if<std::is_floating_point<T>::value, typeless_nan>::type
        operator-(const typeless_nan&,  const T&) { return typeless_nan{}; }

        inline typeless_nan
        operator-(const typeless_nan&,  const typeless_nan&) { return typeless_nan{}; }


        template<typename T>
        DACE_CONSTEXPR typename std::enable_if<std::is_floating_point<T>::value, typeless_nan>::type
        operator/(const T&,  const typeless_nan&) { return typeless_nan{}; }

        template<typename T>
        DACE_CONSTEXPR typename std::enable_if<std::is_floating_point<T>::value, typeless_nan>::type
        operator/(const typeless_nan&,  const T&) { return typeless_nan{}; }

        inline typeless_nan
        operator/(const typeless_nan&,  const typeless_nan&) { return typeless_nan{}; }


        template<typename T>
        DACE_CONSTEXPR typename std::enable_if<std::is_floating_point<T>::value, typeless_nan>::type
        operator%(const T&,  const typeless_nan&) { return typeless_nan{}; }

        template<typename T>
        DACE_CONSTEXPR typename std::enable_if<std::is_floating_point<T>::value, typeless_nan>::type
        operator%(const typeless_nan&,  const T&) { return typeless_nan{}; }

        inline typeless_nan
        operator%(const typeless_nan&,  const typeless_nan&) { return typeless_nan{}; }

    }
}

	//These functions allows to perfrom operations with `typeless_nan` instances.
#	define FADAPT(F) DACE_CONSTEXPR ::dace::math::typeless_nan F (::dace::math::typeless_nan) { return ::dace::math::typeless_nan{}; }
#	define FADAPT2(F) template<typename T1> DACE_CONSTEXPR dace::math::typeless_nan F (T1&&, dace::math::typeless_nan) { return ::dace::math::typeless_nan{}; }; \
			  template<typename T2> DACE_CONSTEXPR dace::math::typeless_nan F (dace::math::typeless_nan, T2&&) { return ::dace::math::typeless_nan{}; }; \
			  DACE_CONSTEXPR ::dace::math::typeless_nan F (dace::math::typeless_nan, dace::math::typeless_nan) { return ::dace::math::typeless_nan{}; }
        FADAPT(tanh); FADAPT(cos); FADAPT(sin); FADAPT(sqrt); FADAPT(tan);
        FADAPT(acos); FADAPT(asin); FADAPT(atan); FADAPT(log); FADAPT(exp);
        FADAPT(floor); FADAPT(ceil); FADAPT(round); FADAPT(abs);
        FADAPT2(max); FADAPT2(min);
#       undef FADAPT2
#	undef FADAPT

#endif  // __DACE_NAN_H
