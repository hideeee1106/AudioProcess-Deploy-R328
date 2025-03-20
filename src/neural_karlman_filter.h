//
// Created by liangtao_deng on 2025/3/17.
//

#ifndef NKF_MNN_DEPLOY_R328_NEURAL_KARLMAN_FILTER_H
#define NKF_MNN_DEPLOY_R328_NEURAL_KARLMAN_FILTER_H


#include "vector"
#include "mnn_adapter.h"
#include <iostream>
#include <fstream>
#include <memory>
#include <complex.h>
#include "../log.h"

using namespace std;

#define SAMEPLERATE  (16000)  // 采样率 16 kHz
#define BLOCK_LEN    (512)    // 处理块长度，每次处理 512 个采样点
#define BLOCK_SHIFT  (256)    // 块移位，每次移动 256 采样点（50% 重叠）
#define FFT_OUT_SIZE (257)    // STFT 变换后单边频谱大小
#define NKF_LEN (4)          // NKF 滤波器的 tap 数
typedef complex<double> cpx_type;  // 复数数据类型

struct nkf_engine {
    float mic_buffer[BLOCK_LEN] = { 0 };  // 近端麦克风信号缓冲区
    float out_buffer[BLOCK_LEN] = { 0 };  // 处理后增强的音频缓冲区
    float lpb_buffer[BLOCK_LEN] = { 0 };  // 低功率带信号（参考信号）缓冲区

    float lpb_real[FFT_OUT_SIZE*NKF_LEN] = { 0 }; // 低功率带信号的实部
    float lpb_imag[FFT_OUT_SIZE*NKF_LEN] = { 0 }; // 低功率带信号的虚部
    double h_prior_real[FFT_OUT_SIZE*NKF_LEN] = { 0 }; // 先验滤波器实部
    double h_prior_imag[FFT_OUT_SIZE*NKF_LEN] = { 0 }; // 先验滤波器虚部
    double h_posterior_real[FFT_OUT_SIZE*NKF_LEN] = { 0 }; // 后验滤波器实部
    double h_posterior_imag[FFT_OUT_SIZE*NKF_LEN] = { 0 }; // 后验滤波器虚部

    std::vector<std::vector<float>> instates; // 存储 ONNX 模型的内部状态
};


class NKFProcessor{
public:
    NKFProcessor(){};
    ~NKFProcessor(){};
    std::vector<float > nearbuffer;
    std::vector<float > farbuffer;
    std::vector<float > outputbuffer;
    int datalens = 512;


    int Aec_Init(std::string &model_data){
        nkf_net = std::make_shared<MNNAudioAdapter>(model_data,1);
        for (int i=0;i<BLOCK_LEN;i++){
            hanning_windows[i]=sinf(PI*i/(BLOCK_LEN-1));
        }
        ResetInout();
    }


    void Aec_prepare(){

        float estimated_block[BLOCK_LEN]={0};
        float mic_real[FFT_OUT_SIZE]={0};
        float mic_imag[FFT_OUT_SIZE]={0};

        double mic_in[BLOCK_LEN]={0};
        std::vector<cpx_type> mic_res(BLOCK_LEN);
        double lpb_in[BLOCK_LEN]={0};
        std::vector<cpx_type> lpb_res(BLOCK_LEN);

        std::vector<size_t> shape;
        shape.push_back(BLOCK_LEN);
        std::vector<size_t> axes;
        axes.push_back(0);
        std::vector<ptrdiff_t> stridel, strideo;
        strideo.push_back(sizeof(cpx_type));
        stridel.push_back(sizeof(double));

        for (int i = 0; i < BLOCK_LEN; i++){
            mic_in[i] = m_pEngine.mic_buffer[i]*hanning_windows[i];
            lpb_in[i]= m_pEngine.lpb_buffer[i]*hanning_windows[i];
        }

        pocketfft::r2c(shape, stridel, strideo, axes, pocketfft::FORWARD, mic_in, mic_res.data(), 1.0);
        pocketfft::r2c(shape, stridel, strideo, axes, pocketfft::FORWARD, lpb_in,lpb_res.data(), 1.0);

        memmove(m_pEngine.lpb_real,m_pEngine.lpb_real+FFT_OUT_SIZE,(NKF_LEN-1)*FFT_OUT_SIZE*sizeof(float));
        memmove(m_pEngine.lpb_imag,m_pEngine.lpb_imag+FFT_OUT_SIZE,(NKF_LEN-1)*FFT_OUT_SIZE*sizeof(float));
        for (int i=0;i<FFT_OUT_SIZE;i++){
            m_pEngine.lpb_real[(NKF_LEN-1)*FFT_OUT_SIZE+i]=static_cast<float>(lpb_res[i].real());
            m_pEngine.lpb_imag[(NKF_LEN-1)*FFT_OUT_SIZE+i]=static_cast<float>(lpb_res[i].imag());
            mic_real[i]=static_cast<float>(mic_res[i].real());
            mic_imag[i]=static_cast<float>(mic_res[i].imag());
        }
        float dh_real[NKF_LEN*FFT_OUT_SIZE]={0};
        float dh_imag[NKF_LEN*FFT_OUT_SIZE]={0};
        for (int i=0;i<NKF_LEN*FFT_OUT_SIZE;i++){
            dh_real[i]=static_cast<float>(m_pEngine.h_posterior_real[i]-m_pEngine.h_prior_real[i]);
            dh_imag[i]=static_cast<float>(m_pEngine.h_posterior_imag[i]-m_pEngine.h_prior_imag[i]);

        }

        memcpy(m_pEngine.h_prior_real,m_pEngine.h_posterior_real,NKF_LEN*FFT_OUT_SIZE*sizeof(double));
        memcpy(m_pEngine.h_prior_imag,m_pEngine.h_posterior_imag,NKF_LEN*FFT_OUT_SIZE*sizeof(double));

        float input_feature_real[(2*NKF_LEN+1)*FFT_OUT_SIZE]={0};
        float input_feature_imag[(2*NKF_LEN+1)*FFT_OUT_SIZE]={0};
        double e_real[FFT_OUT_SIZE]={0};
        double e_imag[FFT_OUT_SIZE]={0};


        int k=2*NKF_LEN+1;
        int is_tensor=1;
        for (int i=0;i<FFT_OUT_SIZE;i++){

            for (int j=0;j<NKF_LEN;j++){
                input_feature_real[k*i+j]=m_pEngine.lpb_real[j*FFT_OUT_SIZE+i];
                input_feature_imag[k*i+j]=m_pEngine.lpb_imag[j*FFT_OUT_SIZE+i];
                input_feature_real[k*i+j+NKF_LEN+1]=dh_real[NKF_LEN*i+j];
                input_feature_imag[k*i+j+NKF_LEN+1]=dh_imag[NKF_LEN*i+j];

                e_real[i] +=m_pEngine.lpb_real[j*FFT_OUT_SIZE+i]*m_pEngine.h_prior_real[NKF_LEN*i+j] -m_pEngine.lpb_imag[j*FFT_OUT_SIZE+i]*m_pEngine.h_prior_imag[NKF_LEN*i+j];
                e_imag[i] +=m_pEngine.lpb_real[j*FFT_OUT_SIZE+i]*m_pEngine.h_prior_imag[NKF_LEN*i+j] +m_pEngine.lpb_imag[j*FFT_OUT_SIZE+i]*m_pEngine.h_prior_real[NKF_LEN*i+j];
            }
            e_real[i]=mic_real[i]-e_real[i];
            e_imag[i]=mic_imag[i]-e_imag[i];
            input_feature_real[k*i+NKF_LEN]=static_cast<float>(e_real[i]);
            input_feature_imag[k*i+NKF_LEN]=static_cast<float>(e_imag[i]);

        }

    }


    void Aec_process(const std::vector<cpx_type>& nearEnd, const std::vector<cpx_type>& farEnd){
        nkf_net->Infer(nearEnd,farEnd);
    };

    std::shared_ptr<MNNAudioAdapter> nkf_net;

private:
    void ResetInout(){

        m_pEngine.instates.clear();
        m_pEngine.instates.resize(4);
        for (int i=0;i<4;i++){
            m_pEngine.instates[i].clear();
            m_pEngine.instates[i].resize(FFT_OUT_SIZE*18);
            std::fill(m_pEngine.instates[i].begin(),m_pEngine.instates[i].end(),0);
        }
        memset(m_pEngine.mic_buffer,0,BLOCK_LEN*sizeof(float));
        memset(m_pEngine.lpb_buffer,0,BLOCK_LEN*sizeof(float));
        memset(m_pEngine.out_buffer,0,BLOCK_LEN*sizeof(float));
        memset(m_pEngine.lpb_real,0,FFT_OUT_SIZE*NKF_LEN*sizeof(float));
        memset(m_pEngine.lpb_imag,0,FFT_OUT_SIZE*NKF_LEN*sizeof(float));
        memset(m_pEngine.h_posterior_real,0,FFT_OUT_SIZE*NKF_LEN*sizeof(double));
        memset(m_pEngine.h_posterior_imag,0,FFT_OUT_SIZE*NKF_LEN*sizeof(double));
        memset(m_pEngine.h_prior_real,0,FFT_OUT_SIZE*NKF_LEN*sizeof(double));
        memset(m_pEngine.h_prior_imag,0,FFT_OUT_SIZE*NKF_LEN*sizeof(double));

    };

private:
    nkf_engine m_pEngine;
    float hanning_windows[BLOCK_LEN]={0};






};
#endif //NKF_MNN_DEPLOY_R328_NEURAL_KARLMAN_FILTER_H
