//
// Created by liangtao_deng on 2025/3/17.
//

#ifndef NKF_MNN_DEPLOY_R328_NEURAL_KARLMAN_FILTER_H
#define NKF_MNN_DEPLOY_R328_NEURAL_KARLMAN_FILTER_H


#include "vector"
#include "mnn_adapter.h"
#include <iostream>
#include <fstream>


class NKFProcessor{
public:
    NKFProcessor(){};
    ~NKFProcessor(){};
    std::vector<short > nearbuffer;
    std::vector<short > farbuffer;

    int Aec_Init(std::string &model_data){
        nkf_net = std::make_shared<MNNAudioAdapter>(model_data,1);;
    }

    void Aec_process(){

    };

    std::shared_ptr<MNNAudioAdapter> nkf_net;


private:






};
#endif //NKF_MNN_DEPLOY_R328_NEURAL_KARLMAN_FILTER_H
