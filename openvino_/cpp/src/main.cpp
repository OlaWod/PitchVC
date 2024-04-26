// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <filesystem>

#include <torch/torch.h>
#include <openvino/openvino.hpp>

#include "json.hpp"
#include "wav.h"
#include "tqdm.hpp"

#include "fft.hpp"
#include "tensor.h"


float tune_f0(float initial_f0, int i) {
    if (i == 0) {
        return initial_f0;
    }
    const static float threshold = 10.0;
    const static float step = (std::log(1100) - std::log(50)) / 256;
    bool voiced = initial_f0 > threshold;
    float initial_lf0 = std::log(initial_f0);
    float lf0 = initial_lf0 + step * i;
    float f0 = std::exp(lf0);
    f0 = voiced ? f0 : initial_f0;
    return f0;
}


void infer(const std::string& title, const std::string& wav_path, const std::string& spk, int f0_shift,
           const std::filesystem::path& assets_dir, const std::filesystem::path& out_dir,
           const nlohmann::json& f0_stats_json, const nlohmann::json& spk2id_json, const nlohmann::json& spk_stats_json,
           const ufft::TorchSTFT& torchstft, const ufft::TorchMel& torchmel,
           ov::InferRequest& infer_request_1, ov::InferRequest& infer_request_2){

    // -------- Step 1. Set up input --------

    // f0_mean_tgt
    float f0_mean_tgt = f0_stats_json[spk]["mean"];
    f0_mean_tgt = tune_f0(f0_mean_tgt, f0_shift);
    std::vector<float> f0_mean_tgt_vec = {f0_mean_tgt};

    // spk_id
    int64_t spk_id = spk2id_json[spk];
    std::vector<int64_t> spk_id_vec = {spk_id};

    // spk_emb
    std::string best_spk_emb = spk_stats_json[spk]["best_spk_emb"];
    best_spk_emb = best_spk_emb + ".npy";
    std::filesystem::path spk_emb_path = assets_dir / "dataset" / "spk" / spk / best_spk_emb;
    std::shared_ptr<unsigned char> spk_emb_data;
    std::vector<size_t> spk_emb_shape;
    std::tie(spk_emb_data, spk_emb_shape) = ut::load_numpy_data(spk_emb_path);

    // wav
    wenet::WavReader wav_reader(wav_path);
    float* wav_data = const_cast<float*>(wav_reader.data());
    torch::Tensor wav_tensor_torch = torch::from_blob(wav_data, {1, wav_reader.num_samples()});
    wav_tensor_torch = wav_tensor_torch / wenet::MAX_WAV_VALUE;

    // mel
    torch::Tensor mel_tensor_torch = torchmel.get_mel(wav_tensor_torch);

    // ov tensors
    ov::Tensor wav_tensor_ov = ut::wrap_torch_tensor_as_ov(wav_tensor_torch);
    ov::Tensor mel_tensor_ov = ut::wrap_torch_tensor_as_ov(mel_tensor_torch);
    ov::Tensor spk_emb_tensor_ov = ov::Tensor(ov::element::f32, ov::Shape({1, 256}), spk_emb_data.get());
    ov::Tensor spk_id_tensor_ov = ov::Tensor(ov::element::i64, ov::Shape({1, 1}), spk_id_vec.data());
    ov::Tensor f0_mean_tgt_tensor_ov = ov::Tensor(ov::element::f32, ov::Shape({1, 1}), f0_mean_tgt_vec.data());

    // -------- Step 2. g1 --------
    infer_request_1.set_input_tensor(0, wav_tensor_ov);
    infer_request_1.set_input_tensor(1, mel_tensor_ov);
    infer_request_1.set_input_tensor(2, spk_emb_tensor_ov);
    infer_request_1.set_input_tensor(3, spk_id_tensor_ov);
    infer_request_1.set_input_tensor(4, f0_mean_tgt_tensor_ov);

    infer_request_1.infer();

    const ov::Tensor& x_tensor_ov = infer_request_1.get_output_tensor(0);
    const ov::Tensor& har_source_tensor_ov = infer_request_1.get_output_tensor(1);

    // -------- Step 3. stft --------
    torch::Tensor har_source_tensor_torch = ut::wrap_ov_tensor_as_torch(har_source_tensor_ov);
    torch::Tensor har_spec_tensor_torch, har_phase_tensor_torch;
    std::tie(har_spec_tensor_torch, har_phase_tensor_torch) = torchstft.transform(har_source_tensor_torch);

    ov::Tensor har_spec_tensor_ov = ut::wrap_torch_tensor_as_ov(har_spec_tensor_torch);
    ov::Tensor har_phase_tensor_ov = ut::wrap_torch_tensor_as_ov(har_phase_tensor_torch);

    // -------- Step 4. g2 --------
    infer_request_2.set_input_tensor(0, x_tensor_ov);
    infer_request_2.set_input_tensor(1, har_spec_tensor_ov);
    infer_request_2.set_input_tensor(2, har_phase_tensor_ov);

    infer_request_2.infer();

    const ov::Tensor& spec_tensor_ov = infer_request_2.get_output_tensor(0);
    const ov::Tensor& phase_tensor_ov = infer_request_2.get_output_tensor(1);

    // -------- Step 5. istft --------
    torch::Tensor spec_tensor_torch = ut::wrap_ov_tensor_as_torch(spec_tensor_ov);
    torch::Tensor phase_tensor_torch = ut::wrap_ov_tensor_as_torch(phase_tensor_ov);
    torch::Tensor wav_out_tensor_torch = torchstft.inverse(spec_tensor_torch, phase_tensor_torch);
    wav_out_tensor_torch = wav_out_tensor_torch / torch::max(torch::abs(wav_out_tensor_torch)) * 0.95;
    wav_out_tensor_torch = wav_out_tensor_torch * wenet::MAX_WAV_VALUE;

    // -------- Step 6. Write out wav --------
    std::string full_title = title + ".wav";
    std::filesystem::path out_path = out_dir / full_title;
    const float* wav_out = wav_out_tensor_torch.data_ptr<float>();
    int num_samples_out = wav_out_tensor_torch.size(-1);
    wenet::WavWriter wav_writer(wav_out, num_samples_out, 1, 16000, 16);
    wav_writer.Write(out_path);
}


// Main function
int main(int argc, char* argv[]) {
    try {
        // -------- Get OpenVINO runtime version --------
        std::cout << ov::get_openvino_version() << std::endl;

        // -------- Parsing and validation of input arguments --------
        if (argc != 5) {
            std::cout << "Usage : " << argv[0] << " <path_to_assets_dir> <path_to_infer_file> <path_to_output_dir> <device_name>" << std::endl;
            return EXIT_FAILURE;
        }

        const std::string args = argv[0];
        const std::string assets_dir_ = argv[1];
        const std::string infer_file = argv[2];
        const std::string out_dir_ = argv[3];
        const std::string device_name = argv[4];

        if (!std::filesystem::exists(out_dir_)) {
            if (!std::filesystem::create_directory(out_dir_)) {
                std::cerr << "Failed to create directory: " << out_dir_ << std::endl;
                return EXIT_FAILURE;
            }
        }
        std::filesystem::path assets_dir = assets_dir_;
        std::filesystem::path out_dir = out_dir_;

        // -------- Read infer file --------
        std::ifstream file(infer_file);
        if (!file.is_open()) {
            std::cerr << "Failed to open file." << std::endl;
            return EXIT_FAILURE;
        }

        std::vector<std::string> lines;
        std::string line;
        while (std::getline(file, line)) {
            lines.push_back(line);
        }
        file.close();

        std::vector<std::tuple<std::string, std::string, std::string, std::int64_t>> candidates;
        for (const auto& line : lines) {
            std::istringstream iss(line);
            std::string title, src_wav, tgt_spk, f0_shift;

            if (std::getline(iss, title, '|') &&
                std::getline(iss, src_wav, '|') &&
                std::getline(iss, tgt_spk, '|') &&
                std::getline(iss, f0_shift, '|')) {
                candidates.push_back(std::make_tuple(title, src_wav, tgt_spk, std::stoi(f0_shift)));
            } else {
                std::cerr << "Invalid line: " << line << std::endl;
            }
        }

        // -------- Read json files --------
        std::filesystem::path jsons_dir = assets_dir / "jsons";

        std::filesystem::path spk2id_path = jsons_dir / "spk2id.json";
        std::ifstream spk2id_file(spk2id_path);
        nlohmann::json spk2id_json = nlohmann::json::parse(spk2id_file);
        spk2id_file.close();

        std::filesystem::path f0_stats_path = jsons_dir / "f0_stats.json";
        std::ifstream f0_stats_file(f0_stats_path);
        nlohmann::json f0_stats_json = nlohmann::json::parse(f0_stats_file);
        f0_stats_file.close();
        
        std::filesystem::path spk_stats_path = jsons_dir / "spk_stats.json";
        std::ifstream spk_stats_file(spk_stats_path);
        nlohmann::json spk_stats_json = nlohmann::json::parse(spk_stats_file);
        spk_stats_file.close();

        // -------- Initialize stft & mel functions --------
        ufft::TorchSTFT torchstft(16, 4, 16);
        ufft::TorchMel torchmel(assets_dir);

        // -------- Step 1. Initialize OpenVINO Runtime Core --------
        ov::Core core;

        // -------- Step 2. Read a model --------
        std::filesystem::path model_dir = assets_dir / "models";
        std::filesystem::path g1_path = model_dir / "g1.xml";
        std::filesystem::path g2_path = model_dir / "g2.xml";

        std::cout << "Loading models..." << std::endl;
        
        std::shared_ptr<ov::Model> g1 = core.read_model(g1_path);
        std::shared_ptr<ov::Model> g2 = core.read_model(g2_path);

        // -------- Step 3. Loading a model to the device --------
        ov::CompiledModel compiled_g1 = core.compile_model(g1, device_name);
        ov::CompiledModel compiled_g2 = core.compile_model(g2, device_name);

        // -------- Step 4. Create an infer request --------
        ov::InferRequest infer_request_1 = compiled_g1.create_infer_request();
        ov::InferRequest infer_request_2 = compiled_g2.create_infer_request();

        // -------- Step 5. Infer --------
        std::cout << "Inferencing..." << std::endl;
        for(const auto& [title, src_wav, tgt_spk, f0_shift] : tq::tqdm(candidates)) {
            infer(title, src_wav, tgt_spk, f0_shift, 
                  assets_dir, out_dir, 
                  f0_stats_json, spk2id_json, spk_stats_json, 
                  torchstft, torchmel, 
                  infer_request_1, infer_request_2);
        }
        std::cout << std::endl << "Done." << std::endl;
        
        // -----------------------------------------------------------------------------------------------------
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
