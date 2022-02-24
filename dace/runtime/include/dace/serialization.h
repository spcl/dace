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
    if (!res || res == EEXIST)
        return true;
    return false;
#endif
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
    void save(const T *buffer, size_t size, const std::string &arrayname, const std::string &filename) {
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
        ifs.read((char *)buffer, sizeof(T) * size);
    }
};

}  // namespace dace

#endif  // __DACE_SERIALIZATION_H
