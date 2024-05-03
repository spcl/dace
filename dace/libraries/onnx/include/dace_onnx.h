#pragma once

#include "onnxruntime_c_api.h"
#include "cpu_provider_factory.h"

// helper function to check for status
static void __ort_check_status(const OrtApi* ort_api, OrtStatus* status)
{
    if (status != NULL) {
        const char* msg = ort_api->GetErrorMessage(status);
        fprintf(stderr, "%s\\n", msg);
        ort_api->ReleaseStatus(status);
        exit(1);
    }
}
