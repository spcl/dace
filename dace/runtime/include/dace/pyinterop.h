// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_INTEROP_H
#define __DACE_INTEROP_H

#include <Python.h>

#include <cstring>
#include <stdexcept>
#include <type_traits>

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

typedef void *pyobject;

template <typename T>
inline const char* dace_numpy_dtype_name() {
  if constexpr (std::is_same_v<T, double>) {
    return "float64";
  } else if constexpr (std::is_same_v<T, float>) {
    return "float32";
  } else if constexpr (std::is_same_v<T, int8_t>) {
    return "int8";
  } else if constexpr (std::is_same_v<T, int16_t>) {
    return "int16";
  } else if constexpr (std::is_same_v<T, int32_t>) {
    return "int32";
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return "int64";
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    return "uint8";
  } else if constexpr (std::is_same_v<T, uint16_t>) {
    return "uint16";
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    return "uint32";
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    return "uint64";
  } else if constexpr (std::is_same_v<T, bool>) {
    return "bool_";
  } else {
    throw std::runtime_error("Unsupported NumPy dtype conversion");
  }
}

template <typename T>
inline const char* dace_ctypes_scalar_name() {
  if constexpr (std::is_same_v<T, double>) {
    return "c_double";
  } else if constexpr (std::is_same_v<T, float>) {
    return "c_float";
  } else if constexpr (std::is_same_v<T, int8_t>) {
    return "c_int8";
  } else if constexpr (std::is_same_v<T, int16_t>) {
    return "c_int16";
  } else if constexpr (std::is_same_v<T, int32_t>) {
    return "c_int32";
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return "c_int64";
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    return "c_uint8";
  } else if constexpr (std::is_same_v<T, uint16_t>) {
    return "c_uint16";
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    return "c_uint32";
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    return "c_uint64";
  } else if constexpr (std::is_same_v<T, bool>) {
    return "c_bool";
  } else {
    throw std::runtime_error("Unsupported ctypes scalar conversion");
  }
}

template <typename T>
inline PyObject* dace_make_pyarray(T* ptr, const Py_ssize_t* shape,
                                   const Py_ssize_t* strides, size_t ndim) {
  PyObject* numpy_module = PyImport_ImportModule("numpy");
  if (numpy_module == nullptr) {
    return nullptr;
  }
  PyObject* ctypes_module = PyImport_ImportModule("ctypes");
  if (ctypes_module == nullptr) {
    Py_DecRef(numpy_module);
    return nullptr;
  }

  PyObject* numpy_dtype =
      PyObject_GetAttrString(numpy_module, dace_numpy_dtype_name<T>());
  PyObject* ctypes_scalar =
      PyObject_GetAttrString(ctypes_module, dace_ctypes_scalar_name<T>());
  if (numpy_dtype == nullptr || ctypes_scalar == nullptr) {
    Py_DecRef(numpy_dtype);
    Py_DecRef(ctypes_scalar);
    Py_DecRef(ctypes_module);
    Py_DecRef(numpy_module);
    return nullptr;
  }

  Py_ssize_t total_size = 1;
  for (size_t i = 0; i < ndim; ++i) {
    total_size *= shape[i];
  }

  PyObject* array_len = PyLong_FromSsize_t(total_size);
  PyObject* array_type = PyNumber_Multiply(ctypes_scalar, array_len);
  PyObject* pointer_fn = PyObject_GetAttrString(ctypes_module, "POINTER");
  PyObject* pointer_type =
      pointer_fn ? PyObject_CallFunctionObjArgs(pointer_fn, array_type, nullptr)
                 : nullptr;
  PyObject* voidp_type = PyObject_GetAttrString(ctypes_module, "c_void_p");
  PyObject* cast_fn = PyObject_GetAttrString(ctypes_module, "cast");
  PyObject* address =
      PyLong_FromUnsignedLongLong(reinterpret_cast<unsigned long long>(ptr));
  PyObject* voidp =
      voidp_type ? PyObject_CallFunctionObjArgs(voidp_type, address, nullptr)
                 : nullptr;
  PyObject* casted =
      (cast_fn && voidp && pointer_type)
          ? PyObject_CallFunctionObjArgs(cast_fn, voidp, pointer_type, nullptr)
          : nullptr;
  PyObject* contents =
      casted ? PyObject_GetAttrString(casted, "contents") : nullptr;

  PyObject* shape_tuple = PyTuple_New(ndim);
  PyObject* strides_tuple = PyTuple_New(ndim);
  if (shape_tuple == nullptr || strides_tuple == nullptr) {
    Py_DecRef(strides_tuple);
    Py_DecRef(shape_tuple);
    Py_DecRef(contents);
    Py_DecRef(casted);
    Py_DecRef(voidp);
    Py_DecRef(address);
    Py_DecRef(cast_fn);
    Py_DecRef(voidp_type);
    Py_DecRef(pointer_type);
    Py_DecRef(pointer_fn);
    Py_DecRef(array_type);
    Py_DecRef(array_len);
    Py_DecRef(ctypes_scalar);
    Py_DecRef(numpy_dtype);
    Py_DecRef(ctypes_module);
    Py_DecRef(numpy_module);
    return nullptr;
  }
  for (size_t i = 0; i < ndim; ++i) {
    PyObject* shape_value = PyLong_FromSsize_t(shape[i]);
    PyObject* stride_value = PyLong_FromSsize_t(strides[i]);
    if (shape_value == nullptr || stride_value == nullptr ||
        PyTuple_SetItem(shape_tuple, i, shape_value) != 0 ||
        PyTuple_SetItem(strides_tuple, i, stride_value) != 0) {
      Py_DecRef(shape_value);
      Py_DecRef(stride_value);
      Py_DecRef(strides_tuple);
      Py_DecRef(shape_tuple);
      Py_DecRef(contents);
      Py_DecRef(casted);
      Py_DecRef(voidp);
      Py_DecRef(address);
      Py_DecRef(cast_fn);
      Py_DecRef(voidp_type);
      Py_DecRef(pointer_type);
      Py_DecRef(pointer_fn);
      Py_DecRef(array_type);
      Py_DecRef(array_len);
      Py_DecRef(ctypes_scalar);
      Py_DecRef(numpy_dtype);
      Py_DecRef(ctypes_module);
      Py_DecRef(numpy_module);
      return nullptr;
    }
  }

  PyObject* ndarray_ctor = PyObject_GetAttrString(numpy_module, "ndarray");
  PyObject* args = PyTuple_New(0);
  PyObject* kwargs = PyDict_New();
  if (shape_tuple != nullptr) {
    PyDict_SetItemString(kwargs, "shape", shape_tuple);
  }
  if (numpy_dtype != nullptr) {
    PyDict_SetItemString(kwargs, "dtype", numpy_dtype);
  }
  if (contents != nullptr) {
    PyDict_SetItemString(kwargs, "buffer", contents);
  }
  if (strides_tuple != nullptr) {
    PyDict_SetItemString(kwargs, "strides", strides_tuple);
  }

  PyObject* result =
      ndarray_ctor ? PyObject_Call(ndarray_ctor, args, kwargs) : nullptr;

  Py_DecRef(kwargs);
  Py_DecRef(args);
  Py_DecRef(ndarray_ctor);
  Py_DecRef(strides_tuple);
  Py_DecRef(shape_tuple);
  Py_DecRef(contents);
  Py_DecRef(casted);
  Py_DecRef(voidp);
  Py_DecRef(address);
  Py_DecRef(cast_fn);
  Py_DecRef(voidp_type);
  Py_DecRef(pointer_type);
  Py_DecRef(pointer_fn);
  Py_DecRef(array_type);
  Py_DecRef(array_len);
  Py_DecRef(ctypes_scalar);
  Py_DecRef(numpy_dtype);
  Py_DecRef(ctypes_module);
  Py_DecRef(numpy_module);
  return result;
}

inline PyObject* dace_make_pyobject(pyobject value) {
  PyObject* result = reinterpret_cast<PyObject*>(value);
  Py_IncRef(result);
  return result;
}

inline PyObject* dace_make_pyobject(bool value) {
  return PyBool_FromLong(value ? 1 : 0);
}

template <typename T>
inline std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>,
                        PyObject*>
dace_make_pyobject(T value) {
  return PyLong_FromLongLong(static_cast<long long>(value));
}

template <typename T>
inline std::enable_if_t<std::is_floating_point_v<T>, PyObject*>
dace_make_pyobject(T value) {
  return PyFloat_FromDouble(static_cast<double>(value));
}

template <typename T>
inline void dace_set_pyobject_attr(pyobject obj, const char* attr, T value) {
  PyGILState_STATE gil_state = PyGILState_Ensure();
  PyObject* pyobj = reinterpret_cast<PyObject*>(obj);
  PyObject* pyvalue = dace_make_pyobject(value);
  if (PyObject_SetAttrString(pyobj, attr, pyvalue) != 0) {
    Py_DecRef(pyvalue);
    PyGILState_Release(gil_state);
    throw std::runtime_error("Failed to set Python attribute");
  }
  Py_DecRef(pyvalue);
  PyGILState_Release(gil_state);
}

template <typename T>
inline void dace_set_pyobject_attr_array(pyobject obj, const char* attr, T* ptr,
                                         const Py_ssize_t* shape,
                                         const Py_ssize_t* strides,
                                         size_t ndim) {
  PyGILState_STATE gil_state = PyGILState_Ensure();
  PyObject* pyobj = reinterpret_cast<PyObject*>(obj);
  PyObject* pyvalue = dace_make_pyarray(ptr, shape, strides, ndim);
  if (pyvalue == nullptr) {
    PyGILState_Release(gil_state);
    throw std::runtime_error("Failed to materialize Python array attribute");
  }
  if (PyObject_SetAttrString(pyobj, attr, pyvalue) != 0) {
    Py_DecRef(pyvalue);
    PyGILState_Release(gil_state);
    throw std::runtime_error("Failed to set Python array attribute");
  }
  Py_DecRef(pyvalue);
  PyGILState_Release(gil_state);
}

inline PyObject* dace_resolve_pyobject_attr_path(pyobject obj,
                                                 const char* attr_path) {
  PyObject* current = reinterpret_cast<PyObject*>(obj);
  Py_IncRef(current);
  const char* cursor = attr_path;

  while (current != nullptr) {
    const char* separator = std::strchr(cursor, '.');
    PyObject* next = nullptr;
    if (separator == nullptr) {
      next = PyObject_GetAttrString(current, cursor);
      Py_DecRef(current);
      return next;
    }

    const Py_ssize_t token_size = separator - cursor;
    PyObject* token = PyUnicode_FromStringAndSize(cursor, token_size);
    if (token == nullptr) {
      Py_DecRef(current);
      return nullptr;
    }
    next = PyObject_GetAttr(current, token);
    Py_DecRef(token);
    Py_DecRef(current);
    if (next == nullptr) {
      return nullptr;
    }
    current = next;
    cursor = separator + 1;
  }

  return nullptr;
}

template <typename T>
inline T dace_get_pyobject_attr(pyobject obj, const char* attr) {
  PyGILState_STATE gil_state = PyGILState_Ensure();
  PyObject* pyvalue = dace_resolve_pyobject_attr_path(obj, attr);
  if (pyvalue == nullptr) {
    PyGILState_Release(gil_state);
    throw std::runtime_error("Failed to read Python attribute");
  }

  T result;
  if constexpr (std::is_same_v<T, bool>) {
    result = PyObject_IsTrue(pyvalue) != 0;
  } else if constexpr (std::is_integral_v<T>) {
    result = static_cast<T>(PyLong_AsLongLong(pyvalue));
  } else if constexpr (std::is_floating_point_v<T>) {
    result = static_cast<T>(PyFloat_AsDouble(pyvalue));
  } else {
    Py_DecRef(pyvalue);
    PyGILState_Release(gil_state);
    throw std::runtime_error("Unsupported Python attribute conversion");
  }

  if (PyErr_Occurred()) {
    Py_DecRef(pyvalue);
    PyGILState_Release(gil_state);
    throw std::runtime_error("Failed to convert Python attribute");
  }

  Py_DecRef(pyvalue);
  PyGILState_Release(gil_state);
  return result;
}

template <typename T>
inline T* dace_get_pyobject_attr_ptr(pyobject obj, const char* attr) {
  PyGILState_STATE gil_state = PyGILState_Ensure();
  PyObject* pyvalue = dace_resolve_pyobject_attr_path(obj, attr);
  if (pyvalue == nullptr) {
    PyGILState_Release(gil_state);
    throw std::runtime_error("Failed to read Python attribute");
  }

  Py_buffer view;
  if (PyObject_GetBuffer(pyvalue, &view, PyBUF_STRIDES) != 0) {
    Py_DecRef(pyvalue);
    PyGILState_Release(gil_state);
    throw std::runtime_error("Python attribute does not expose a buffer");
  }

  if (view.itemsize != sizeof(T)) {
    PyBuffer_Release(&view);
    Py_DecRef(pyvalue);
    PyGILState_Release(gil_state);
    throw std::runtime_error("Python attribute buffer itemsize mismatch");
  }

  T* result = reinterpret_cast<T*>(view.buf);
  PyBuffer_Release(&view);
  Py_DecRef(pyvalue);
  PyGILState_Release(gil_state);
  return result;
}

// Sympy functions
template <typename U, typename... T>
static DACE_HDFI U Min(U val, T... vals) {
    return min(val, vals...);
}
template <typename U, typename... T>
static DACE_HDFI U Max(U val, T... vals) {
    return max(val, vals...);
}
template <typename T>
static DACE_HDFI T Abs(T val) {
    return abs(val);
}
template <typename T, typename U>
DACE_CONSTEXPR DACE_HDFI typename std::common_type<T, U>::type IfExpr(bool condition, const T& iftrue, const U& iffalse)
{
    return condition ? iftrue : iffalse;
}

#endif  // __DACE_INTEROP_H
