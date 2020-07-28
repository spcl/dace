#include <google/protobuf/text_format.h>
#include <unordered_map>
#include <string>
#include <iostream>
#include <sstream>
#include "onnxruntime_c_api.h"
#include "onnx/onnx_pb.h"
#include "cpu_provider_factory.h"
#include "cuda_provider_factory.h"

// Start global ORT setup
const OrtApi* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);

// helper function to check for status
void __ort_check_status(OrtStatus* status)
{
    if (status != NULL) {
        const char* msg = ort_api->GetErrorMessage(status);
        fprintf(stderr, "%s\n", msg);
        ort_api->ReleaseStatus(status);
        exit(1);
    }
}

struct State {
    int num_params = 0;
    onnx::NodeProto proto;
    std::unordered_map<std::string, onnx::TypeProto> type_map;
};

extern "C" {

    void* init_state(const char* op_type, const char* name) {
        auto *state = new State();
        state->proto.set_op_type(op_type);
        state->proto.set_name(name);

        return state;
    }

    void free_state(void* state_) {
        State *state = static_cast<State *>(state_);
        delete state;
    }

    void add_input(void* state_, int type) {
        State *state = static_cast<State *>(state_);
        std::ostringstream string_stream;
        string_stream << "Arg_In_";
        string_stream << state->num_params++;
        std::string name = string_stream.str();

        onnx::TypeProto type_proto;
        onnx::TypeProto::Tensor *tensor_type = type_proto.mutable_tensor_type();
        tensor_type->set_elem_type(type);

        state->type_map[name] = type_proto;
        state->proto.add_input()->assign(name);
    }

    void add_output(void* state_, int type) {
        State *state = static_cast<State *>(state_);
        std::ostringstream string_stream;
        string_stream << "Arg_Out_";
        string_stream << state->num_params++;
        std::string name = string_stream.str();

        onnx::TypeProto type_proto;
        onnx::TypeProto::Tensor *tensor_type = type_proto.mutable_tensor_type();
        tensor_type->set_elem_type(type);

        state->type_map[name] = type_proto;
        state->proto.add_output()->assign(name);
    }

    int add_attribute(void* state_, const char* serialized_attr_proto) {
        State *state = static_cast<State *>(state_);
        onnx::AttributeProto* attribute = state->proto.add_attribute();
        return (attribute->ParseFromString(serialized_attr_proto));
    }

    char* try_create(void* state_, int provider_index) {
        State *state = static_cast<State *>(state_);

        OrtEnv* ort_env;
        OrtKernelSession* ort_session;
        OrtSessionOptions* session_options;
        OrtMemoryInfo* mem_info;

        __ort_check_status(ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info));
        __ort_check_status(ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "dace_graph", &ort_env));
        __ort_check_status(ort_api->CreateSessionOptions(&session_options));
        __ort_check_status(OrtSessionOptionsAppendExecutionProvider_CPU(session_options, /*use_arena=*/0));

        // INSERT_CUDA

        __ort_check_status(ort_api->CreateKernelSession(session_options, &ort_session));

        OrtExecutableKernelContext *context;
        auto status = ort_api->CreateExecutableKernelContext(ort_session, /*provider_index*/provider_index, &state->proto, &state->type_map, &context);

        ort_api->ReleaseExecutableKernelContext(context);
        ort_api->ReleaseKernelSession(ort_session);
        ort_api->ReleaseMemoryInfo(mem_info);
        ort_api->ReleaseSessionOptions(session_options);
        ort_api->ReleaseEnv(ort_env);

        if (!status)
            return NULL;

        const char* msg = ort_api->GetErrorMessage(status);
        char* ret_message = new char[2000];
        strncpy(ret_message, msg, 1999);
        ret_message[1999] = '\0';
        ort_api->ReleaseStatus(status);


        return ret_message;
    }
}
