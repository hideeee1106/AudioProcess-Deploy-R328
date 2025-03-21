//
// Created by liangtao_deng on 2025/3/17.
//

#ifndef NKF_MNN_DEPLOY_R328_C_API_H
#define NKF_MNN_DEPLOY_R328_C_API_H


#include <stdint.h>

#if defined(_WIN32)
#ifdef SL_BUILD_SHARED_LIB
#define SL_CAPI_EXPORT __declspec(dllexport)
#else
#define SL_CAPI_EXPORT
#endif
#else
#define SL_CAPI_EXPORT __attribute__((visibility("default")))
#endif // _WIN32

#ifdef __cplusplus
extern "C"
{
#endif

    // Snore Audio Predictor - 鼾声音频检测器
    typedef struct  SL_EchoCancelFilter SL_EchoCancelFilter;

//    SL_CAPI_EXPORT extern int
SL_CAPI_EXPORT extern void SL_CreateEchoCancelFilter(SL_EchoCancelFilter *predictor);

SL_CAPI_EXPORT extern void SL_ReleaseEchoCancelFilter(SL_EchoCancelFilter *predictor);

SL_CAPI_EXPORT extern void SL_EchoCancelFilterForWav1C16khz(SL_EchoCancelFilter *predictor);




























#ifdef __cplusplus
}
#endif





#endif //NKF_MNN_DEPLOY_R328_C_API_H
