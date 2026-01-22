// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_PI_H
#define __DACE_PI_H

#include <type_traits>

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

        template<typename T>
        struct is_typeless_pi { static constexpr bool value = false; };
        #define MAKE_TYPELESS_PI(type) template<> struct is_typeless_pi<type> { static constexpr bool value = true; }

        struct typeless_pi;

        /* Represents $m * \pi$. */
        struct typeless_pi_mult
        {
            int mult;

            DACE_CONSTEXPR DACE_HDFI typeless_pi_mult(int m): mult(m) {}
            DACE_CONSTEXPR DACE_HDFI typeless_pi_mult() noexcept: typeless_pi_mult(1) {};

            DACE_CONSTEXPR DACE_HDFI typeless_pi_mult(const typeless_pi&) noexcept: typeless_pi_mult(1) {};
            DACE_CONSTEXPR DACE_HDFI typeless_pi_mult(const typeless_pi_mult&) noexcept = default;
            DACE_CONSTEXPR DACE_HDFI typeless_pi_mult(typeless_pi_mult&&) noexcept = default;
            DACE_HDFI ~typeless_pi_mult() noexcept = default;

            DACE_CONSTEXPR DACE_HDFI typeless_pi_mult& operator=(const typeless_pi_mult&) noexcept = default;
            DACE_CONSTEXPR DACE_HDFI typeless_pi_mult& operator=(typeless_pi_mult&&) noexcept = default;

            template<
                typename T,
                typename = std::enable_if_t<std::is_integral<T>::value>
            >
            DACE_CONSTEXPR DACE_HDFI operator T() const noexcept
            { return T(mult * M_PI); }

            DACE_CONSTEXPR DACE_HDFI operator float() const noexcept
            { return float(mult * M_PI); }

            DACE_CONSTEXPR DACE_HDFI operator double() const noexcept
            { return mult * M_PI; }

            DACE_CONSTEXPR DACE_HDFI operator long double() const noexcept
            { return (long double)(mult * M_PI); }

            DACE_CONSTEXPR DACE_HDFI typeless_pi_mult operator+() const noexcept
            { return *this; }

            DACE_CONSTEXPR DACE_HDFI typeless_pi_mult operator-() const noexcept
            { return typeless_pi_mult(-this->mult); }
        };
        MAKE_TYPELESS_PI(typeless_pi_mult);

        /* Represents $\pi$ */
        struct typeless_pi
        {
            DACE_CONSTEXPR DACE_HDFI typeless_pi() noexcept = default;
            DACE_CONSTEXPR DACE_HDFI typeless_pi(const typeless_pi&) noexcept = default;
            DACE_CONSTEXPR DACE_HDFI typeless_pi(typeless_pi&&) noexcept = default;
            DACE_HDFI ~typeless_pi() noexcept = default;
            DACE_CONSTEXPR DACE_HDFI typeless_pi& operator=(const typeless_pi&) noexcept = default;
            DACE_CONSTEXPR DACE_HDFI typeless_pi& operator=(typeless_pi&&) noexcept = default;

            template<
                typename T,
                typename = std::enable_if_t<std::is_integral<T>::value>
            >
            DACE_CONSTEXPR DACE_HDFI operator T() const noexcept
            { return T(M_PI); }

            DACE_CONSTEXPR DACE_HDFI operator float() const noexcept
            { return float(M_PI); }

            DACE_CONSTEXPR DACE_HDFI operator double() const noexcept
            { return M_PI; }

            DACE_CONSTEXPR DACE_HDFI operator long double() const noexcept
            { return (long double)(M_PI); }

            DACE_CONSTEXPR DACE_HDFI typeless_pi operator+() const noexcept
            { return *this; }

            DACE_CONSTEXPR DACE_HDFI typeless_pi_mult operator-() const noexcept
            { return typeless_pi_mult(-1); }
        };
        MAKE_TYPELESS_PI(typeless_pi);

        /* Represents $m * \pi^{e}$ */
        struct typeless_pi_exp
        {
            int mult, exp;

            DACE_CONSTEXPR DACE_HDFI typeless_pi_exp(int m, int e): mult(m), exp(e) {}
            DACE_CONSTEXPR DACE_HDFI typeless_pi_exp() noexcept: typeless_pi_exp(1, 1) {};

            DACE_CONSTEXPR DACE_HDFI typeless_pi_exp(const typeless_pi_exp&) noexcept = default;
            DACE_CONSTEXPR DACE_HDFI typeless_pi_exp(typeless_pi_exp&&) noexcept = default;
            DACE_HDFI ~typeless_pi_exp() noexcept = default;

            DACE_CONSTEXPR DACE_HDFI typeless_pi_exp& operator=(const typeless_pi_exp&) noexcept = default;
            DACE_CONSTEXPR DACE_HDFI typeless_pi_exp& operator=(typeless_pi_exp&&) noexcept = default;

            template<
                typename T,
                typename = std::enable_if_t<std::is_integral<T>::value>
            >
            DACE_CONSTEXPR DACE_HDFI operator T() const noexcept
            { return T(mult * std::pow(static_cast<T>(M_PI), exp)); }


            /* We have to do the selection this way, because it seems as nvidia does
             *  not provide `powl` and `powf` in the std namespace */
            DACE_CONSTEXPR DACE_HDFI operator float() const
            { using std::pow; return mult * pow(static_cast<float>(M_PI), exp); }

            DACE_CONSTEXPR DACE_HDFI operator double() const
            { using std::pow; return mult * std::pow(static_cast<double>(M_PI), exp); }

#if !( defined(__CUDACC__) || defined(__HIPCC__) )
            //There is no long double on the GPU
            DACE_CONSTEXPR DACE_HDFI operator long double() const
            { using std::pow; return mult * std::pow(static_cast<long double>(M_PI), exp); }
#endif

            DACE_CONSTEXPR DACE_HDFI typeless_pi_exp operator+() const
            { return *this; }

            DACE_CONSTEXPR DACE_HDFI typeless_pi_exp operator-() const
            { return typeless_pi_exp(-this->mult, this->exp); }
        };
        MAKE_TYPELESS_PI(typeless_pi_exp);


        DACE_CONSTEXPR DACE_HDFI int operator/(const typeless_pi&, const typeless_pi&) noexcept
        { return 1; }

        DACE_CONSTEXPR DACE_HDFI int operator-(const typeless_pi&, const typeless_pi&) noexcept
        { return 0; }


        DACE_CONSTEXPR DACE_HDFI typeless_pi_mult operator*(const typeless_pi&, const int& num) noexcept
        { return typeless_pi_mult(num); }

        DACE_CONSTEXPR DACE_HDFI typeless_pi_mult operator*(const int& num, const typeless_pi&) noexcept
        { return typeless_pi_mult(num); }

        DACE_CONSTEXPR DACE_HDFI typeless_pi_mult operator*(const typeless_pi_mult& p, const int& num) noexcept
        { return typeless_pi_mult(p.mult * num); }

        DACE_CONSTEXPR DACE_HDFI typeless_pi_mult operator*(const int& num, const typeless_pi_mult& p) noexcept
        { return typeless_pi_mult(p.mult * num); }

        DACE_CONSTEXPR DACE_HDFI typeless_pi_mult operator+(const typeless_pi&, const typeless_pi&) noexcept
        { return typeless_pi_mult(2); }

        DACE_CONSTEXPR DACE_HDFI typeless_pi_mult operator+(const typeless_pi&, const typeless_pi_mult& pi) noexcept
        { return typeless_pi_mult(pi.mult + 1); }

        DACE_CONSTEXPR DACE_HDFI typeless_pi_mult operator+(const typeless_pi_mult& pi, const typeless_pi&) noexcept
        { return typeless_pi_mult(pi.mult + 1); }

        DACE_CONSTEXPR DACE_HDFI typeless_pi_mult operator+(const typeless_pi_mult& pl, const typeless_pi_mult& pr) noexcept
        { return typeless_pi_mult(pl.mult + pr.mult); }

        DACE_CONSTEXPR DACE_HDFI typeless_pi_mult operator-(const typeless_pi_mult& pl, const typeless_pi_mult& pr) noexcept
        { return typeless_pi_mult(pl.mult - pr.mult); }

        DACE_CONSTEXPR DACE_HDFI int operator/(const typeless_pi_mult& pl, const typeless_pi&) noexcept
        { return pl.mult; }

        DACE_CONSTEXPR DACE_HDFI double operator/(const typeless_pi& pl, const typeless_pi_mult& pr) noexcept
        { return 1.0 / pr.mult; }



        DACE_CONSTEXPR DACE_HDFI typeless_pi_exp operator*(const typeless_pi&, const typeless_pi&) noexcept
        { return typeless_pi_exp(1, 2); }

        DACE_CONSTEXPR DACE_HDFI typeless_pi_exp operator*(const typeless_pi_mult& pl, const typeless_pi_mult& pr) noexcept
        { return typeless_pi_exp(pl.mult * pr.mult, 2); }

        DACE_CONSTEXPR DACE_HDFI typeless_pi_exp operator*(const typeless_pi_mult& pl, const typeless_pi&) noexcept
        { return typeless_pi_exp(pl.mult, 2); }

        DACE_CONSTEXPR DACE_HDFI typeless_pi_exp operator*(const typeless_pi& pl, const typeless_pi_mult& pr) noexcept
        { return typeless_pi_exp(pr.mult, 2); }

        DACE_CONSTEXPR DACE_HDFI typeless_pi_exp operator*(const typeless_pi_exp& pl, const typeless_pi_mult& pr) noexcept
        { return typeless_pi_exp(pl.mult * pr.mult, pl.exp + 1); }

        DACE_CONSTEXPR DACE_HDFI typeless_pi_exp operator*(const typeless_pi_mult& pl, const typeless_pi_exp& pr) noexcept
        { return pr * pl; }

        DACE_CONSTEXPR DACE_HDFI typeless_pi_exp operator*(const typeless_pi_exp& pl, const typeless_pi_exp& pr) noexcept
        { return typeless_pi_exp(pl.mult * pr.mult, pr.exp + pl.exp); }

        DACE_CONSTEXPR DACE_HDFI typeless_pi_exp operator*(const typeless_pi_exp& pl, const int& num) noexcept
        { return typeless_pi_exp(pl.mult * num, pl.exp); }

        DACE_CONSTEXPR DACE_HDFI typeless_pi_exp operator*(const int& num, const typeless_pi_exp& pr) noexcept
        { return typeless_pi_exp(pr.mult * num, pr.exp); }

        DACE_CONSTEXPR DACE_HDFI typeless_pi_exp operator/(const typeless_pi_exp& pl, const typeless_pi&) noexcept
        { return typeless_pi_exp(pl.mult, pl.exp - 1); }


        // The code generator guarantees us that `b > 0`.
        DACE_HDFI typeless_pi_exp ipow(const typeless_pi_mult& pi, const unsigned int& b) {
            return typeless_pi_exp(pow(pi.mult, b), b);
        }
        DACE_HDFI typeless_pi_exp ipow(const typeless_pi& pi, const unsigned int& b) {
            return typeless_pi_exp(1, b);
        }
        DACE_HDFI typeless_pi_exp ipow(const typeless_pi_exp& pi, const unsigned int& b) {
            return typeless_pi_exp(pow(pi.mult, b), pi.exp * b);
        }

#       define DEF_PI_OPS(op) 										\
	template<typename T, typename PI, typename = std::enable_if_t<is_typeless_pi<PI>::value && (!is_typeless_pi<T>::value)> >	\
	DACE_CONSTEXPR DACE_HDFI T operator op (const T& lhs, const PI& pi) noexcept			\
	{ return lhs op (static_cast<T>(pi)); }								\
	template<typename PI, typename T, typename = std::enable_if_t<is_typeless_pi<PI>::value && (!is_typeless_pi<T>::value)> >	\
	DACE_CONSTEXPR DACE_HDFI T operator op (const PI& pi, const T& rhs) noexcept			\
	{ return (static_cast<T>(pi)) op rhs; }

	DEF_PI_OPS(+);
	DEF_PI_OPS(-);
	DEF_PI_OPS(/);
	DEF_PI_OPS(*);

        DACE_CONSTEXPR DACE_HDFI int sin(const typeless_pi&) noexcept
        { return 0; }

        DACE_CONSTEXPR DACE_HDFI int sin(const typeless_pi_mult& pi) noexcept
        { return 0; }

	DACE_HDFI double sin(const typeless_pi_exp& pi) noexcept
	{ return std::sin(static_cast<double>(pi)); }

        DACE_CONSTEXPR DACE_HDFI int cos(const typeless_pi&) noexcept
        { return 1; }

        DACE_CONSTEXPR DACE_HDFI int cos(const typeless_pi_mult& pi) noexcept
        { return (pi.mult % 2 == 0) ? 1 : (-1); }

	DACE_HDFI double cos(const typeless_pi_exp& pi) noexcept
	{ return std::cos(static_cast<double>(pi)); }


#       define DEF_PI_TRIGO(F)						\
	DACE_HDFI double F (const typeless_pi& pi) noexcept		\
	{ return std:: F( static_cast<double>(pi) ); }			\
	DACE_HDFI double F (const typeless_pi_mult& pi) noexcept	\
	{ return std:: F( static_cast<double>(pi) ); }			\
	DACE_HDFI double F (const typeless_pi_exp& pi) noexcept		\
	{ return std:: F( static_cast<double>(pi) ); }

        DEF_PI_TRIGO(asin);
        DEF_PI_TRIGO(acos);
        DEF_PI_TRIGO(tan);
        DEF_PI_TRIGO(atan);
        DEF_PI_TRIGO(exp);
        DEF_PI_TRIGO(log);


#       undef DEF_PI_TRIGO
#       undef DEF_PI_OPS
#	undef MAKE_TYPELESS_PI
    }
}


#endif  // __DACE_PI_H
