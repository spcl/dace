// Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include <cstdlib>
#include <stdexcept>
#include <string>

#ifdef _MSC_VER
inline int setenv(const char *name, const char *value, int overwrite)
{
    int errcode = 0;
    if (!overwrite) {
        size_t envsize = 0;
        errcode = getenv_s(&envsize, NULL, 0, name);
        if (errcode || envsize) return errcode;
    }
    return _putenv_s(name, value);
}
inline int unsetenv(const char *name)
{
    return _putenv_s(name, "");
}
#endif // _MSC_VER

namespace dace {



inline void set_environment_variable(std::string const &key,
                                     std::string const &val) {
  const auto ret = setenv(key.c_str(), val.c_str(), 1);
  if (ret != 0) {
    throw std::runtime_error("Failed to set environment variable " + key);
  }
}

inline void unset_environment_variable(std::string const &key) {
  unsetenv(key.c_str());
}

} // End namespace dace
