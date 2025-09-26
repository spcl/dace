#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cstdint>
#include <stdexcept>
#include <cstring>
#include <random>
#include <cmath>

class ELFGenerator {
public:
    ELFGenerator(size_t alignment = 64, uint32_t base_address = 0xC0000000)
        : alignment_(alignment), base_address_(base_address), current_address_(base_address) {}

    void generate(const std::string& output_file_path,
                  const std::vector<std::vector<uint64_t>>& arrays,
                  const std::vector<std::string>& dtypes,
                  const std::vector<uint32_t>& start_addresses = {}) {
        if (!start_addresses.empty() && start_addresses.size() != arrays.size()) {
            throw std::invalid_argument("start_addresses size must match arrays size or be empty.");
        }

        create_array_file(arrays, dtypes, start_addresses);
        create_linker_script(arrays, start_addresses);
        compile_elf(output_file_path);
        cleanup();
    }

private:
    size_t alignment_;
    uint32_t base_address_;
    uint32_t current_address_;
    std::vector<std::string> section_names_;

    std::string get_ctype(const std::string& dtype) {
        if (dtype == "int8") return "int8_t";
        if (dtype == "uint8") return "uint8_t";
        if (dtype == "int16") return "int16_t";
        if (dtype == "uint16") return "uint16_t";
        if (dtype == "int32") return "int32_t";
        if (dtype == "uint32") return "uint32_t";
        if (dtype == "int64") return "int64_t";
        if (dtype == "uint64") return "uint64_t";
        if (dtype == "float16") return "float16";
        if (dtype == "float32") return "float";
        if (dtype == "float64") return "double";
        throw std::invalid_argument("Unsupported dtype: " + dtype);
    }

    void create_array_file(const std::vector<std::vector<uint64_t>>& arrays,
                           const std::vector<std::string>& dtypes,
                           const std::vector<uint32_t>& start_addresses) {
        std::ofstream array_file("array.cpp");
        if (!array_file.is_open()) {
            throw std::runtime_error("Failed to open array.cpp for writing.");
        }

        array_file << "#include <cstdint>\n";

        for (size_t idx = 0; idx < arrays.size(); ++idx) {
            const auto& array = arrays[idx];
            const std::string& dtype = dtypes[idx];

            std::string c_type = get_ctype(dtype);
            std::string section_name = ".custom_section_" + std::to_string(idx);
            section_names_.push_back(section_name);

            uint32_t start_addr = start_addresses.empty() ? ((current_address_ + alignment_ - 1) & ~(alignment_ - 1)) : start_addresses[idx];

            if (start_addr % alignment_ != 0) {
                throw std::invalid_argument("Provided address is not alignment-byte aligned.");
            }

            array_file << c_type << " array_" << idx << "[] __attribute__((section(\"" << section_name << "\"))) = {";
            for (size_t i = 0; i < array.size(); ++i) {
                if (i > 0) array_file << ", ";
                array_file << array[i];
            }
            array_file << "};\n";

            current_address_ = start_addr + array.size() * sizeof(uint64_t);
        }

        array_file.close();
    }

    void create_linker_script(const std::vector<std::vector<uint64_t>>& arrays,
                               const std::vector<uint32_t>& start_addresses) {
        std::ofstream linker_file("link.ld");
        if (!linker_file.is_open()) {
            throw std::runtime_error("Failed to open link.ld for writing.");
        }

        linker_file << "SECTIONS {\n";
        current_address_ = base_address_;

        for (size_t idx = 0; idx < arrays.size(); ++idx) {
            uint32_t start_addr = start_addresses.empty() ? ((current_address_ + alignment_ - 1) & ~(alignment_ - 1)) : start_addresses[idx];
            linker_file << "    . = 0x" << std::hex << start_addr << ";\n    " << section_names_[idx] << " : { *(" << section_names_[idx] << ") }\n";
            current_address_ = start_addr + arrays[idx].size() * sizeof(uint64_t);
        }

        linker_file << "}\n";
        linker_file.close();
    }

    void compile_elf(const std::string& output_file_path) {
        std::string compile_command = "export PATH=/scratch/dace4softhier/gvsoc/third_party/toolchain/v1.0.16-pulp-riscv-gcc-centos-7/bin:$PATH && riscv32-unknown-elf-g++ -c array.cpp -o array.o";
        system(compile_command.c_str());

        std::string link_command = "export PATH=/scratch/dace4softhier/gvsoc/third_party/toolchain/v1.0.16-pulp-riscv-gcc-centos-7/bin:$PATH && riscv32-unknown-elf-ld -T link.ld array.o -o " + output_file_path;
        system(link_command.c_str());

        std::string strip_command = "export PATH=/scratch/dace4softhier/gvsoc/third_party/toolchain/v1.0.16-pulp-riscv-gcc-centos-7/bin:$PATH && riscv32-unknown-elf-strip --remove-section=.comment --remove-section=.Pulp_Chip.Info " + output_file_path;
        system(strip_command.c_str());
    }


    void cleanup() {
        std::remove("array.cpp");
        std::remove("link.ld");
        std::remove("array.o");
    }
};
