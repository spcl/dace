// Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_SERIALIZATION_H
#define __DACE_SERIALIZATION_H

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#undef __in
#undef __inout
#undef __out
#else
#include <sys/stat.h>
#endif

namespace dace {


static bool create_directory(const char *dirpath) {
#if defined(_WIN32) || defined(_WIN64)
    // TODO: Use CreateDirectoryW for unicode character paths
    BOOL res = ::CreateDirectoryA(dirpath, NULL);
    if (!res) {
        DWORD err = ::GetLastError();
        if (err == ERROR_ALREADY_EXISTS)
            return true;
        return false;
    }
    return true;
#else
    // The second argument emulates chmod 0755
    int res = ::mkdir(dirpath, S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
    if (!res || errno == EEXIST)
        return true;
    return false;
#endif
}

static inline void write_parameter_pack(std::ofstream &ofs) {
}

template <typename T, typename... Args>
static inline void write_parameter_pack(std::ofstream& ofs, T value, Args... values) {
    uint32_t cast = uint32_t(value);
    ofs.write((const char *)&cast, sizeof(uint32_t));
    write_parameter_pack(ofs, values...);
}

class DataSerializer {
protected:
    std::mutex _mutex;
    std::string folder;
    std::map<std::string, int> version;
    bool enable;

public:
    DataSerializer(const std::string& build_folder) : enable(true) {
        long unsigned int tstart = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();

        if (build_folder.length() > 0) {
            std::stringstream ss;
            ss << build_folder << "/" << tstart;
            this->folder = ss.str();

            // Try to create directories
            if (!create_directory(build_folder.c_str())) {
                printf("WARNING: Could not create directory '%s' for data instrumentation. Skipping saves.\n",
                    build_folder.c_str());
                this->enable = false;
                return;
            }
            if (!create_directory(this->folder.c_str())) {
                printf("WARNING: Could not create directory '%s' for data instrumentation. Skipping saves.\n",
                    this->folder.c_str());
                this->enable = false;
            }
        }
    }

    ~DataSerializer() {}

    void set_folder(const std::string& folder) {
        this->folder = folder;
    }

    template <typename T>
    void save_symbol(const std::string &symbol_name, const std::string &filename, const T symbol_value) {
        if (!this->enable) return;
        std::lock_guard<std::mutex> guard(this->_mutex);

        // Update version
        int version;
        if (this->version.find(symbol_name) == this->version.end())
            version = 0;
        else
            version = this->version[symbol_name] + 1;
        this->version[symbol_name] = version;

        std::stringstream ss;
        ss << this->folder << "/" << symbol_name;

        // Try to create directory for symbol versions
        if (!create_directory(ss.str().c_str())) {
            if (version == 0)  // Only print the first time
                printf("WARNING: Could not create directory '%s' for data instrumentation.\n", ss.str().c_str());
            return;
        }

        // Write contents to file
        ss << "/" << filename << "_" << version;
        std::ofstream ofs(ss.str(), std::ios::out);
        ofs << symbol_value;
    }

    template <typename T>
    T restore_symbol(const std::string &symbol_name, const std::string &filename) {
        std::lock_guard<std::mutex> guard(this->_mutex);

        // Update version
        int version;
        if (this->version.find(symbol_name) == this->version.end())
            version = 0;
        else
            version = this->version[symbol_name] + 1;
        this->version[symbol_name] = version;

        // Read contents from file
        std::stringstream ss;
        ss << this->folder << "/" << symbol_name << "/" << filename << "_" << version;
        std::ifstream ifs(ss.str(), std::ios::in);

        // Read the symbol back
        T val;
        ifs >> val;
        return val;
    }

    template <typename T, typename... Args>
    void save(const T *buffer, size_t size, const std::string &arrayname, const std::string &filename, Args... shape_stride) {
        // NOTE: The "shape_stride" parameter is two concatenated tuples of shape, strides
        if (!this->enable) return;
        std::lock_guard<std::mutex> guard(this->_mutex);

        // Update version
        int version;
        if (this->version.find(filename) == this->version.end())
            version = 0;
        else
            version = this->version[filename] + 1;
        this->version[filename] = version;

        std::stringstream ss;
        ss << this->folder << "/" << arrayname;

        // Try to create directory for array versions
        if (!create_directory(ss.str().c_str())) {
            if (version == 0)  // Only print the first time
                printf("WARNING: Could not create directory '%s' for data instrumentation.\n", ss.str().c_str());
            return;
        }

        // Write contents to file
        ss << "/" << filename << "_" << version << ".bin";
        std::ofstream ofs(ss.str(), std::ios::binary);
        uint32_t ndims = sizeof...(shape_stride) / 2;
        ofs.write((const char *)&ndims, sizeof(uint32_t));
        write_parameter_pack(ofs, shape_stride...);
        ofs.write((const char *)buffer, sizeof(T) * size);
    }

    template <typename T>
    void restore(T *buffer, size_t size, const std::string &arrayname, const std::string &filename) {
        std::lock_guard<std::mutex> guard(this->_mutex);

        // Update version
        int version;
        if (this->version.find(filename) == this->version.end())
            version = 0;
        else
            version = this->version[filename] + 1;
        this->version[filename] = version;

        // Read contents from file
        std::stringstream ss;
        ss << this->folder << "/" << arrayname << "/" << filename << "_" << version << ".bin";
        std::ifstream ifs(ss.str(), std::ios::binary);
        
        // Ignore header (dimensions, shape, and strides)
        uint32_t ndims;
        ifs.read((char *)&ndims, sizeof(uint32_t));
        ifs.ignore(ndims * 2 * sizeof(uint32_t));

        // Read contents
        ifs.read((char *)buffer, sizeof(T) * size);
    }
};

}  // namespace dace

#endif  // __DACE_SERIALIZATION_H
