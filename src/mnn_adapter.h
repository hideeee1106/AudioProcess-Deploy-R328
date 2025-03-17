//
// Created by liangtao_deng on 2025/3/17.
//

#ifndef NKF_MNN_DEPLOY_R328_MNN_ADAPTER_H
#define NKF_MNN_DEPLOY_R328_MNN_ADAPTER_H

#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>
#include <MNN/MNNForwardType.h>

#include <iostream>
#include <fstream>
#include <vector>

class MNNAudioAdapter {
public:

    MNNAudioAdapter(const std::string &model, int thread, bool use_model_bin = false) {
        backend_ = MNN_FORWARD_CPU;
        detect_model_ = std::shared_ptr<MNN::Interpreter>(
                MNN::Interpreter::createFromFile(model.c_str()));

        _config.type = backend_;
        _config.numThread = thread;
        MNN::BackendConfig backendConfig;
        backendConfig.precision = MNN::BackendConfig::Precision_High;
        backendConfig.power = MNN::BackendConfig::Power_High;
        _config.backendConfig = &backendConfig;

    }

    MNNAudioAdapter(const solex::Model *model, int thread) {
        backend_ = MNN_FORWARD_CPU;
        detect_model_ =
                std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromBuffer(
                        model->caffemodelBuffer, model->modelsize.caffemodel_size));
        _config.type = backend_;
        _config.numThread = thread;

    }


    ~MNNAudioAdapter() {
        detect_model_->releaseModel();
        detect_model_->releaseSession(sess);
    }

    void Init(const std::string &input, int numcp,int frame_num) {
        sess = detect_model_->createSession(_config);
        input_ = detect_model_->getSessionInput(sess, nullptr);
    }

    void Infer(const std::vector<std::complex<float>> &input) {
        // 预处理：填充输入张量
        auto input_data = input_->host<float>();
        for (size_t i = 0; i < input.size(); ++i) {
            input_data[i * 2] = input[i].real();
            input_data[i * 2 + 1] = input[i].imag();
        }
    }

    std::vector<std::complex<float>> get_output(){
        // 获取输出数据
        auto output_data = output_tensor->host<float>();
        std::vector<std::complex<float>> output(input.size());
        for (size_t i = 0; i < output.size(); ++i) {
            output[i] = std::complex<float>(output_data[i * 2], output_data[i * 2 + 1]);
        }
        return output;
    }


private:
    std::shared_ptr<MNN::Interpreter> detect_model_;
    MNN::Session *sess{};
    MNN::ScheduleConfig _config;
    MNNForwardType backend_;

};


#endif //NKF_MNN_DEPLOY_R328_MNN_ADAPTER_H
