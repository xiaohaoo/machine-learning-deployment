//
//  main.c
//  machine-learning-deployment
//
//  Created by xiaohao on 2022/3/1.
//

#include <stdio.h>
#include <onnx/onnxruntime_c_api.h>
#include <time.h>

const OrtApi* ort_api;

#define ORT_ABORT_ON_ERROR(expr)                               \
  do {                                                         \
    OrtStatus* onnx_status = (expr);                           \
    if (onnx_status != NULL) {                                 \
      const char* msg = ort_api->GetErrorMessage(onnx_status); \
      fprintf(stderr, "%s\n", msg);                            \
      ort_api->ReleaseStatus(onnx_status);                     \
      abort();                                                 \
    }                                                          \
  } while (0);


int main(int argc, const char * argv[]) {
    
    
    clock_t start_time = clock();
    
    ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    OrtEnv* env;
    
    ORT_ABORT_ON_ERROR(ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ONNXD", &env));
    
    OrtSessionOptions* session_options;
    
    ORT_ABORT_ON_ERROR(ort_api->CreateSessionOptions(&session_options));

    
    char model_path[] = "onnx.onnx";
    
    OrtSession* session;
    
    ORT_ABORT_ON_ERROR(ort_api->CreateSession(env, model_path, session_options, &session));
    
    OrtMemoryInfo* menory_info;
    
    ORT_ABORT_ON_ERROR(ort_api->CreateCpuMemoryInfo(OrtDeviceAllocator,OrtMemTypeCPU,&menory_info));
    
    float* model_input;
    
    
    const size_t input_shape[] = {1, 3, 299, 299};
    
    OrtValue* input_tensor = NULL;
    
    

    
    
//    ort_api->CreateTensorWithDataAsOrtValue(memory_info, model_input, model_input_len, input_shape,
//                                                               input_shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
//                                            &input_tensor));
    
    
    ort_api->ReleaseSessionOptions(session_options);
    ort_api->ReleaseSession(session);
    ort_api->ReleaseEnv(env);
    
    clock_t end_time = clock();
    
    printf("用时：%.3f秒\n",(double)(end_time - start_time)/CLOCKS_PER_SEC);

    return 0;
}
