#include <iostream>
#include <iomanip>
#include <cstring>
#include <cmath>

// State structures (empty for now, but needed for function signatures)
struct single_assignment_cond_from_scalar_state_t {};
struct non_fusion_single_assignment_cond_from_scalar_state_t {};

// ============================================================================
// TRANSFORMED VERSION (with branch fusion)
// ============================================================================
inline void single_assignment_cond_from_scalar_1193_4_0_0_2(
    single_assignment_cond_from_scalar_state_t *__state, 
    const double& __tmp_1194_21_r, 
    double& __tmp_1196_12_w) {
    
    double _if_cond_1;
    _if_cond_1 = __tmp_1194_21_r;
    {
        double float__if_cond_1;
        double else_body_tmp_0;
        double tmp_ieassign;
        {
            double _in = __tmp_1196_12_w;
            double _out;
            ///////////////////
            // Tasklet code (identity_assign_0)
            _out = _in;
            ///////////////////
            else_body_tmp_0 = _out;
        }
        {
            double _in___tmp_1194_21_r_0 = __tmp_1194_21_r;
            double _out_float__if_cond_1;
            ///////////////////
            // Tasklet code (ieassign__if_cond_1_to_float__if_cond_1_scalar)
            _out_float__if_cond_1 = (_in___tmp_1194_21_r_0 > 0.0);
            ///////////////////
            tmp_ieassign = _out_float__if_cond_1;
        }
        {
            double _in = tmp_ieassign;
            double _out;
            ///////////////////
            // Tasklet code (assign)
            _out = _in;
            ///////////////////
            float__if_cond_1 = _out;
        }
        {
            double _in_right = else_body_tmp_0;
            double _in_factor = float__if_cond_1;
            double _out;
            ///////////////////
            // Tasklet code (combine_branch_values_for___tmp_1196_12_w_0)
            _out = (_in_right * (1.0 - _in_factor));
            ///////////////////
            __tmp_1196_12_w = _out;
        }
    }
}

void __program_single_assignment_cond_from_scalar_internal(
    single_assignment_cond_from_scalar_state_t*__state, 
    double * __restrict__ a) {
    
    {
        {
            #pragma omp parallel for
            for (auto i = 0; i < 256; i += 1) {
                single_assignment_cond_from_scalar_1193_4_0_0_2(__state, a[(2 * i)], a[i]);
            }
        }
    }
}

// ============================================================================
// NON-FUSED VERSION (original with if statement)
// ============================================================================
inline void fuse_branches_test_single_assignment_cond_from_scalar_1214_4_0_0_2(
    non_fusion_single_assignment_cond_from_scalar_state_t *__state, 
    const double& __tmp_1215_21_r, 
    double& __tmp_1217_12_w) {
    
    double _if_cond_1;
    _if_cond_1 = __tmp_1215_21_r;
    if ((_if_cond_1 > 0.0)) {
        {
            {
                double __out;
                ///////////////////
                // Tasklet code (assign_1217_12)
                __out = 0.0;
                ///////////////////
                __tmp_1217_12_w = __out;
            }
        }
    }
}

void __program_non_fusion_single_assignment_cond_from_scalar_internal(
    non_fusion_single_assignment_cond_from_scalar_state_t*__state, 
    double * __restrict__ a) {
    
    {
        {
            #pragma omp parallel for
            for (auto i = 0; i < 256; i += 1) {
                fuse_branches_test_single_assignment_cond_from_scalar_1214_4_0_0_2(__state, a[(2 * i)], a[i]);
            }
        }
    }
}

// ============================================================================
// DRIVER CODE
// ============================================================================
int main() {
    const int SIZE = 512;  // Need 512 elements since we access a[2*i] where i goes up to 255
    
    // Allocate arrays
    double *a_transformed = new double[SIZE];
    double *a_original = new double[SIZE];
    
    // Initialize with test values
    std::cout << "Initializing arrays with test values...\n\n";
    for (int i = 0; i < SIZE; i++) {
        // Set some pattern: even indices get condition values, odd get data values
        if (i % 2 == 0) {
            // Condition values: mix of positive and negative
            a_transformed[i] = (i / 2) % 3 == 0 ? 1.0 : -1.0;
        } else {
            // Data values
            a_transformed[i] = 100.0 + i;
        }
        a_original[i] = a_transformed[i];
    }
    
    // Print initial state for first few elements
    std::cout << "Initial state (first 20 elements):\n";
    std::cout << std::fixed << std::setprecision(2);
    for (int i = 0; i < 20; i++) {
        std::cout << "a[" << std::setw(2) << i << "] = " << std::setw(8) << a_transformed[i];
        if (i % 2 == 0) {
            std::cout << " (condition for a[" << i/2 << "])";
        } else {
            std::cout << " (data value)";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
    
    // Create state objects
    single_assignment_cond_from_scalar_state_t state_transformed;
    non_fusion_single_assignment_cond_from_scalar_state_t state_original;
    
    // Run transformed version
    std::cout << "Running transformed (fused) version...\n";
    __program_single_assignment_cond_from_scalar_internal(&state_transformed, a_transformed);
    
    // Run original version
    std::cout << "Running original (non-fused) version...\n";
    __program_non_fusion_single_assignment_cond_from_scalar_internal(&state_original, a_original);
    
    // Compare results
    std::cout << "\nComparing results:\n";
    std::cout << std::setw(5) << "i" 
              << std::setw(15) << "Condition" 
              << std::setw(15) << "Original" 
              << std::setw(15) << "Transformed" 
              << std::setw(10) << "Match?\n";
    std::cout << std::string(60, '-') << "\n";
    
    bool all_match = true;
    int mismatches = 0;
    
    for (int i = 0; i < 256; i++) {
        double condition = a_transformed[2 * i];
        double orig_val = a_original[i];
        double trans_val = a_transformed[i];
        
        bool match = std::fabs(orig_val - trans_val) < 1e-10;
        if (!match) {
            all_match = false;
            mismatches++;
        }
        
        // Print first 20 and all mismatches
        if (i < 20 || !match) {
            std::cout << std::setw(5) << i 
                      << std::setw(15) << condition
                      << std::setw(15) << orig_val 
                      << std::setw(15) << trans_val 
                      << std::setw(10) << (match ? "✓" : "✗")
                      << "\n";
        }
    }
    
    // Summary
    std::cout << "\n" << std::string(60, '=') << "\n";
    if (all_match) {
        std::cout << "SUCCESS: All values match!\n";
    } else {
        std::cout << "FAILURE: " << mismatches << " mismatches found out of 256 values\n";
    }
    std::cout << std::string(60, '=') << "\n";
    
    // Cleanup
    delete[] a_transformed;
    delete[] a_original;
    
    return all_match ? 0 : 1;
}