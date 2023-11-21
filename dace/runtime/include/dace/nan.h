// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_NAN_H
#define __DACE_NAN_H

// Class to define a stateless NAN and related operators.

#include <cmath>
#include <stdexcept>

namespace dace
{
    namespace math
    {
        //////////////////////////////////////////////////////
        // Defines a typeless Pi
        struct typeless_nan
        {
            operator int() const
            {
                throw std::logic_error("Tried to convert a `NAN` into an `int`.");
            }
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
        std::enable_if_t<std::is_floating_point<T>::value, typeless_nan>
        operator*(const T& lhs,  const typeless_nan& rhs) { return rhs; }

        template<typename T>
        std::enable_if_t<std::is_floating_point<T>::value, typeless_nan>
        operator*(const typeless_nan& rhs,  const T& lhs) { return rhs; }

        template<typename T>
        std::enable_if_t<std::is_floating_point<T>::value, typeless_nan>
        operator+(const T& lhs,  const typeless_nan& rhs) { return rhs; }

        template<typename T>
        std::enable_if_t<std::is_floating_point<T>::value, typeless_nan>
        operator+(const typeless_nan& rhs,  const T& lhs) { return rhs; }

        template<typename T>
        std::enable_if_t<std::is_floating_point<T>::value, typeless_nan>
        operator-(const T& lhs,  const typeless_nan& rhs) { return rhs; }

        template<typename T>
        std::enable_if_t<std::is_floating_point<T>::value, typeless_nan>
        operator-(const typeless_nan& rhs,  const T& lhs) { return rhs; }

        template<typename T>
        std::enable_if_t<std::is_floating_point<T>::value, typeless_nan>
        operator/(const T& lhs,  const typeless_nan& rhs) { return rhs; }

        template<typename T>
        std::enable_if_t<std::is_floating_point<T>::value, typeless_nan>
        operator/(const typeless_nan& rhs,  const T& lhs) { return rhs; }

        template<typename T>
        std::enable_if_t<std::is_floating_point<T>::value, typeless_nan>
        operator%(const T& lhs,  const typeless_nan& rhs) { return rhs; }

        template<typename T>
        std::enable_if_t<std::is_floating_point<T>::value, typeless_nan>
        operator%(const typeless_nan& rhs,  const T& lhs) { return rhs; }
    }
}


#endif  // __DACE_NAN_H
