#include <unordered_map>
#include <string>
#include <iostream>
#include <sstream>
#include "onnxruntime_c_api.h"
#include "cpu_provider_factory.h"
#include "cuda_provider_factory.h"

#include <dlfcn.h>
#define DACE_EXPORTED extern "C"

// Start global ORT setup
const OrtApi* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);

// Do not free the returned value
DACE_EXPORTED const char* GetErrorMessage(const OrtStatus* status) {
	return ort_api->GetErrorMessage(status);
}

DACE_EXPORTED OrtStatus* CreateEnv(OrtEnv** ort_env) {
    return ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "op_checker_env", ort_env);
}

DACE_EXPORTED OrtStatus* CreateSessionOptions(OrtSessionOptions** options) {
    return ort_api->CreateSessionOptions(options);
}


DACE_EXPORTED OrtStatus* CreateKernelSession(const OrtSessionOptions* options, OrtKernelSession** session) {
	return ort_api->CreateKernelSession(options, session, 12);
}

DACE_EXPORTED OrtStatus* CreateExecutableKernelContext(const char* name, const char* op_type, OrtExecutableKernelContext** kernel) {
	return ort_api->CreateExecutableKernelContext(name, op_type, kernel);
}

DACE_EXPORTED OrtStatus* ExecutableKernelContext_AddInput(OrtExecutableKernelContext* context, ONNXTensorElementDataType type) {
	return ort_api->ExecutableKernelContext_AddInput(context, type);
}

DACE_EXPORTED OrtStatus* ExecutableKernelContext_AddOutput(OrtExecutableKernelContext* context, ONNXTensorElementDataType type) {
	return ort_api->ExecutableKernelContext_AddOutput(context, type);
}

DACE_EXPORTED OrtStatus* ExecutableKernelContext_AddAttributeString(
		OrtExecutableKernelContext* context,
		const char* name,
		const char* value) {
	return ort_api->ExecutableKernelContext_AddAttributeString(context, name, value);

}
DACE_EXPORTED OrtStatus* ExecutableKernelContext_AddAttributeStrings(
		OrtExecutableKernelContext* context,
		const char* name,
		const char** values,
		size_t num_values) {
	return ort_api->ExecutableKernelContext_AddAttributeStrings(context, name, values, num_values);
}

DACE_EXPORTED OrtStatus* ExecutableKernelContext_AddAttributeFloat(
		OrtExecutableKernelContext* context,
		const char* name,
		float value) {
	return ort_api->ExecutableKernelContext_AddAttributeFloat(context, name, value);
}
DACE_EXPORTED OrtStatus* ExecutableKernelContext_AddAttributeFloats(
		OrtExecutableKernelContext* context,
		const char* name,
		float* values,
		size_t num_values) {
	return ort_api->ExecutableKernelContext_AddAttributeFloats(context, name, values, num_values);
}
DACE_EXPORTED OrtStatus* ExecutableKernelContext_AddAttributeInt(
		OrtExecutableKernelContext* context,
		const char* name,
		int64_t value) {
	return ort_api->ExecutableKernelContext_AddAttributeInt(context, name, value);
}
DACE_EXPORTED OrtStatus* ExecutableKernelContext_AddAttributeInts(
		OrtExecutableKernelContext* context,
		const char* name,
		int64_t* values,
		size_t num_values) {
	return ort_api->ExecutableKernelContext_AddAttributeInts(context, name, values, num_values);
}
DACE_EXPORTED OrtStatus* ExecutableKernelContext_AddAttributeTensor(
		OrtExecutableKernelContext* context,
		const char* name,
		void* p_data,
		size_t p_data_len,
		const int64_t* shape,
		size_t shape_len,
		ONNXTensorElementDataType type) {
	return ort_api->ExecutableKernelContext_AddAttributeTensor(context, name, p_data, p_data_len, shape, shape_len, type);
}

DACE_EXPORTED OrtStatus* CreateExecutableKernel(
		OrtKernelSession* session,
		OrtExecutableKernelContext* context,
		size_t provider_id,
		OrtExecutableKernel** kernel) {
	return ort_api->CreateExecutableKernel(session, context, provider_id, kernel);
}

DACE_EXPORTED OrtStatus* ExecutableKernel_IsOutputOnCpu(
		OrtExecutableKernel* kernel, int index, int* is_output_on_cpu) {
	return ort_api->ExecutableKernel_IsOutputOnCpu(kernel, index, is_output_on_cpu);
}

DACE_EXPORTED OrtStatus* ExecutableKernel_IsInputOnCpu(
		OrtExecutableKernel* kernel, int index, int* is_input_on_cpu) {
	return ort_api->ExecutableKernel_IsInputOnCpu(kernel, index, is_input_on_cpu);
}

DACE_EXPORTED void ReleaseExecutableKernel (OrtExecutableKernel* input) {
	ort_api->ReleaseExecutableKernel(input);
}

DACE_EXPORTED void ReleaseExecutableKernelContext (OrtExecutableKernelContext* input) {
	ort_api->ReleaseExecutableKernelContext(input);
}

DACE_EXPORTED void ReleaseKernelSession (OrtKernelSession* input) {
	ort_api->ReleaseKernelSession(input);
}

DACE_EXPORTED void ReleaseSessionOptions (OrtSessionOptions* input) {
	ort_api->ReleaseSessionOptions(input);
}

DACE_EXPORTED void ReleaseStatus (OrtStatus* input) {
	ort_api->ReleaseStatus(input);
}

DACE_EXPORTED void ReleaseEnv (OrtEnv* input) {
	ort_api->ReleaseEnv(input);
}

DACE_EXPORTED ONNXTensorElementDataType GetONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED(){
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
}
DACE_EXPORTED ONNXTensorElementDataType GetONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT() {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}
DACE_EXPORTED ONNXTensorElementDataType GetONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8() {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
}
DACE_EXPORTED ONNXTensorElementDataType GetONNX_TENSOR_ELEMENT_DATA_TYPE_INT8() {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
}
DACE_EXPORTED ONNXTensorElementDataType GetONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16() {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
}
DACE_EXPORTED ONNXTensorElementDataType GetONNX_TENSOR_ELEMENT_DATA_TYPE_INT16() {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
}
DACE_EXPORTED ONNXTensorElementDataType GetONNX_TENSOR_ELEMENT_DATA_TYPE_INT32() {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
}
DACE_EXPORTED ONNXTensorElementDataType GetONNX_TENSOR_ELEMENT_DATA_TYPE_INT64() {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}
DACE_EXPORTED ONNXTensorElementDataType GetONNX_TENSOR_ELEMENT_DATA_TYPE_STRING() {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
}
DACE_EXPORTED ONNXTensorElementDataType GetONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL() {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
}
DACE_EXPORTED ONNXTensorElementDataType GetONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16() {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
}
DACE_EXPORTED ONNXTensorElementDataType GetONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE() {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
}
DACE_EXPORTED ONNXTensorElementDataType GetONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32() {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
}
DACE_EXPORTED ONNXTensorElementDataType GetONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64() {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
}
DACE_EXPORTED ONNXTensorElementDataType GetONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64() {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64;
}
DACE_EXPORTED ONNXTensorElementDataType GetONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128() {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128;
}
DACE_EXPORTED ONNXTensorElementDataType GetONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16() {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
}
