// Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_PI_H
#define __DACE_PI_H

// Classes that are used to define a typeless Pi

//#define _USE_MATH_DEFINES
//#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace dace
{
    namespace math
    {
        //////////////////////////////////////////////////////
        // Defines a typeless Pi
        struct typeless_pi
        {
            double value() const { return M_PI; }
            operator int() const
            {
                return int(this->value());
            }
            operator float() const
            {
                return float(this->value());
            }
            operator double() const
            {
                return double(this->value());
            }
        };
        struct typeless_pi_mult : typeless_pi
        {
            int mult; typeless_pi_mult(int m = 1) : mult(m) {}
            double value() const { return mult * M_PI; }

            operator int() const
            {
                return int(this->value());
            }
            operator float() const
            {
                return float(this->value());
            }
            operator double() const
            {
                return double(this->value());
            }
        };
        struct typeless_pi_exp : typeless_pi_mult
        {
            int mult, exp; typeless_pi_exp(int m = 1, int e = 1) : mult(m), exp(e) {}
            double value() const { return mult * std::pow(M_PI, exp); }
            operator int() const
            {
                return int(this->value());
            }
            operator float() const
            {
                return float(this->value());
            }
            operator double() const
            {
                return double(this->value());
            }
        };
        inline typeless_pi_mult operator*(const typeless_pi&, const int& num)
        {
            return typeless_pi_mult(num);
        }
        inline typeless_pi_mult operator*(const typeless_pi_mult& p, const int& num)
        {
            return typeless_pi_mult(p.mult * num);
        }
        inline typeless_pi_exp operator*(const typeless_pi_exp& p, const int& num)
        {
            return typeless_pi_exp(p.mult * num, p.exp);
        }
        inline typeless_pi_mult operator*(const int& num, const typeless_pi&)
        {
            return typeless_pi_mult(num);
        }
        inline typeless_pi_mult operator*(const int& num, const typeless_pi_mult& p)
        {
            return typeless_pi_mult(num * p.mult);
        }
        inline typeless_pi_exp operator*(const int& num, const typeless_pi_exp& p)
        {
            return typeless_pi_exp(num * p.mult, p.exp);
        }
        template <typename T>
        T operator+(const typeless_pi& p, const T& num)
        {
            return T(p.value()) + num;
        }
        template <typename T>
        T operator-(const typeless_pi& p, const T& num)
        {
            return T(p.value()) - num;
        }

        template <typename T>
        T operator*(const typeless_pi& p, const T& num)
        {
            return T(p.value()) * num;
        }
        template <typename T>
        T operator/(const typeless_pi& p, const T& num)
        {
            return T(p.value()) / num;
        }
        template <typename T>
        T operator+(const T& num, const typeless_pi& p)
        {
            return num + T(p.value());
        }
        template <typename T>
        T operator-(const T& num, const typeless_pi& p)
        {
            return num - T(p.value());
        }
        template <typename T>
        T operator*(const T& num, const typeless_pi& p)
        {
            return num * T(p.value());
        }
        template <typename T>
        T operator/(const T& num, const typeless_pi& p)
        {
            return num / T(p.value());
        }
        template <typename T>
        T operator+(const typeless_pi_mult& p, const T& num)
        {
            return T(p.value()) + num;
        }
        template <typename T>
        T operator-(const typeless_pi_mult& p, const T& num)
        {
            return T(p.value()) - num;
        }

        template <typename T>
        T operator*(const typeless_pi_mult& p, const T& num)
        {
            return T(p.value()) * num;
        }
        template <typename T>
        T operator/(const typeless_pi_mult& p, const T& num)
        {
            return T(p.value()) / num;
        }
        template <typename T>
        T operator+(const T& num, const typeless_pi_mult& p)
        {
            return num + T(p.value());
        }
        template <typename T>
        T operator-(const T& num, const typeless_pi_mult& p)
        {
            return num - T(p.value());
        }
        template <typename T>
        T operator*(const T& num, const typeless_pi_mult& p)
        {
            return num * T(p.value());
        }
        template <typename T>
        T operator/(const T& num, const typeless_pi_mult& p)
        {
            return num / T(p.value());
        }
        template <typename T>
        T operator+(const typeless_pi_exp& p, const T& num)
        {
            return T(p.value()) + num;
        }
        template <typename T>
        T operator-(const typeless_pi_exp& p, const T& num)
        {
            return T(p.value()) - num;
        }

        template <typename T>
        T operator*(const typeless_pi_exp& p, const T& num)
        {
            return T(p.value()) * num;
        }
        template <typename T>
        T operator/(const typeless_pi_exp& p, const T& num)
        {
            return T(p.value()) / num;
        }
        template <typename T>
        T operator+(const T& num, const typeless_pi_exp& p)
        {
            return num + T(p.value());
        }
        template <typename T>
        T operator-(const T& num, const typeless_pi_exp& p)
        {
            return num - T(p.value());
        }
        template <typename T>
        T operator*(const T& num, const typeless_pi_exp& p)
        {
            return num * T(p.value());
        }
        template <typename T>
        T operator/(const T& num, const typeless_pi_exp& p)
        {
            return num / T(p.value());
        }
        inline typeless_pi_mult operator-(const typeless_pi&)
        {
            return typeless_pi_mult(-1);
        }
        template <typename T>
        typeless_pi_mult operator+(const typeless_pi&, const typeless_pi&)
        {
            return typeless_pi_mult(2);
        }
        template <typename T>
        typeless_pi_mult operator+(const typeless_pi_mult& p1, const typeless_pi_mult& p2)
        {
            return typeless_pi_mult(p1.mult + p2.mult);
        }
        template <typename T>
        typeless_pi_exp operator*(const typeless_pi_mult& p1, const typeless_pi_mult& p2)
        {
            return typeless_pi_exp(p1.mult * p2.mult, 2);
        }
        template <typename T>
        typeless_pi_exp operator*(const typeless_pi&, const typeless_pi&)
        {
            return typeless_pi_exp(1, 2);
        }
        template <typename T>
        typeless_pi_exp operator*(const typeless_pi_exp& p1, const typeless_pi_exp& p2)
        {
            return typeless_pi_exp(p1.mult * p2.mult, p1.exp + p2.exp);
        }
    }
}


#endif  // __DACE_PI_H
