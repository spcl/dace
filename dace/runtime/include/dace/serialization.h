// Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_SERIALIZATION_H
#define __DACE_SERIALIZATION_H

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>

namespace dace {

class DataSerializer {
protected:
    std::mutex _mutex;
    std::string folder;
    std::map<std::string, int> _version;

public:
    DataSerializer(const std::string& build_folder) {
        long unsigned int tstart = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        std::stringstream ss;
        ss << build_folder << "/" << tstart;
        this->folder = ss.str();
    }

    ~DataSerializer() {}

    void set_folder(const std::string& folder) {
        this->folder = folder;
    }

    template <typename T>
    void save(const T *buffer, size_t size, const std::string &filename) {
        std::lock_guard<std::mutex> guard(this->_mutex);

        // Update version
        int version;
        if (this->_version.find(filename) == this->_version.end())
            version = 0;
        else
            version = this->_version[filename] + 1;
        this->_version[filename] = version;

        // Write contents to file
        std::stringstream ss;
        ss << this->folder << "/" << filename << "_" << version;
        std::ofstream ofs(ss.str(), std::ios::binary);
        ofs.write((const char *)buffer, sizeof(T) * size);
    }

    template <typename T>
    void restore(T *buffer, size_t size, const std::string &filename) {
        std::lock_guard<std::mutex> guard(this->_mutex);

        // Update version
        int version;
        if (this->_version.find(filename) == this->_version.end())
            version = 0;
        else
            version = this->_version[filename] + 1;
        this->_version[filename] = version;

        // Read contents from file
        std::stringstream ss;
        ss << this->folder << "/" << filename << "_" << version;
        std::ifstream ifs(ss.str(), std::ios::binary);
        ifs.read((char *)buffer, sizeof(T) * size);
    }
};

}  // namespace dace

#endif  // __DACE_SERIALIZATION_H
