#include <iostream>
#include <onnx/onnxruntime_cxx_api.h>

using namespace std;

int main(int argc, const char * argv[]) {
    Ort::Env env(ORT_LOGGING_LEVEL_VERBOSE, "ONNXD");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    
    cout <<session_options <<endl;
    return 0;
}

