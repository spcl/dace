// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_NAN_H
#define __DACE_NAN_H

// Class to define a stateless NAN and related operators.

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
                return std::nanf("");
            }
            operator double() const
            {
                return std::nan("");
            }
            operator long double() const
            {
                return std::nanl("");
            }
            typeless_nan operator+() const
            {
                *this;
            }
            typeless_nan operator-() const
            {
                *this;
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

	//These functions allows to perfrom operations with `typeless_nan` instances.
#	define FADAPT(F) typeless_nan DACE_CONSTEXPR F (const typeless_nan&) { return typeless_nan{}; }
#	define FADAPT2(F) typeless_nan DACE_CONSTEXPR F (const typeless_nan&, const typeless_nan&) { return typeless_nan{}; }
#	define FADAPT3(F) typeless_nan DACE_CONSTEXPR F (const typeless_nan&, const typeless_nan&, const typeless_nan&) { return typeless_nan{}; }
#	define FADAPT4(F) typeless_nan DACE_CONSTEXPR F (const typeless_nan&, const typeless_nan&, const typeless_nan&, const typeless_nan&) { return typeless_nan{}; }
        FADAPT(tanh); FADAPT(cos); FADAPT(sin); FADAPT(sqrt); FADAPT(tan);
        FADAPT(acos); FADAPT(asin); FADAPT(atan); FADAPT(log); FADAPT(exp);
        FADAPT(floor); FADAPT(ceil); FADAPT(round); FADAPT(abs);
        FADAPT2(max); FADAPT2(min);
#       undef FADAPT4
#       undef FADAPT3
#       undef FADAPT2
#	undef FADAPT
    }
}


#endif  // __DACE_NAN_H
