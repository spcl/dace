// Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_INTEROP_H
#define __DACE_INTEROP_H

#include "types.h"

// Various classes to simplify interoperability with python in code converted to C++

class range 
{
public:
    class iterator
    {
        friend class range;
    public:
        DACE_HDFI int operator *() const { return i_; }
        DACE_HDFI const iterator &operator ++() { i_ += s_; return *this; }
        DACE_HDFI iterator operator ++(int) { iterator copy(*this); i_ += s_; return copy; }

        DACE_HDFI bool operator ==(const iterator &other) const { return i_ == other.i_; }
        DACE_HDFI bool operator !=(const iterator &other) const { return i_ != other.i_; }

    protected:
        DACE_HDFI iterator(int start, int skip = 1) : i_(start), s_(skip) { }

    private:
        int i_, s_;
    };

    DACE_HDFI iterator begin() const { return begin_; }
    DACE_HDFI iterator end() const { return end_; }
    DACE_HDFI range(int end) : begin_(0), end_(end) {}
    DACE_HDFI range(int begin, int end) : begin_(begin), end_(end) {}
    DACE_HDFI range(int begin, int end, int skip) : begin_(begin, skip), end_(end, skip) {}
private:
    iterator begin_;
    iterator end_;
};

#endif  // __DACE_INTEROP_H
