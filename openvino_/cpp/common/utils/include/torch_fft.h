#include <torch/torch.h>

#include "format_reader_ptr.h"

namespace F = torch::nn::functional;


namespace tfft{

class TorchSTFT : public torch::nn::Module {
public:
    TorchSTFT(int n_fft=1024, int hop_length=320, int win_length=1024)
        : n_fft(n_fft), hop_length(hop_length), win_length(win_length) {
        lpad = std::floor((n_fft - hop_length) / 2);
        rpad = lpad + hop_length;

        window = torch::hann_window(win_length);
    }

    std::tuple<torch::Tensor, torch::Tensor> transform(torch::Tensor y) {
        y = F::pad(y.unsqueeze(1), F::PadFuncOptions({lpad, rpad}).mode(torch::kReflect));
        y = y.squeeze(1);

        torch::Tensor forward_transform = torch::stft(
            y, n_fft, hop_length, win_length, window,
            false, true, true
        );
        forward_transform = forward_transform.contiguous();

        torch::Tensor magnitude = torch::abs(forward_transform); 
        torch::Tensor phase = torch::angle(forward_transform);

        return std::make_tuple(magnitude, phase);
    }

    torch::Tensor inverse(torch::Tensor magnitude, torch::Tensor phase) {
        torch::Tensor zero = torch::zeros_like(phase);
        phase = torch::complex(zero, phase);
        torch::Tensor x = magnitude * torch::exp(phase);

        torch::Tensor inverse_transform = torch::istft(
            x, n_fft, hop_length, win_length, window
        );

        return inverse_transform.unsqueeze(-2);
    }

private:
    int n_fft;
    int hop_length;
    int win_length;
    int lpad;
    int rpad;
    torch::Tensor window;
};


class TorchMel : public torch::nn::Module {
public:
    TorchMel(std::filesystem::path assets_dir) {
            n_fft = 1024;
            hop_length = 320;
            win_length = 1024;
            pad = std::floor((n_fft - hop_length) / 2);

            window = torch::hann_window(win_length);

            std::filesystem::path mel_basis_path = assets_dir / "mel_basis.npy";
            FormatReader::ReaderPtr mel_basis_reader(mel_basis_path.c_str());
            if (mel_basis_reader.get() == nullptr) {
                throw std::logic_error("mel_basis.npy cannot be read!");
            }
            std::shared_ptr<unsigned char> mel_basis_data = mel_basis_reader->getData();
            mel_basis = torch::from_blob(mel_basis_data.get(), {80, 513});
    }

    torch::Tensor get_mel(torch::Tensor y) {
        y = F::pad(y.unsqueeze(1), F::PadFuncOptions({pad, pad}).mode(torch::kReflect));
        y = y.squeeze(1);

        torch::Tensor spec = torch::stft(
            y, n_fft, hop_length, win_length, window, 
            false, true, true
        );
        spec = torch::view_as_real(spec);
        spec = torch::sqrt(spec.pow(2).sum(-1) + 1e-9);

        spec = torch::matmul(mel_basis, spec);
        spec = dynamic_range_compression_torch(spec);

        return spec;
    }

private:
    int n_fft;
    int hop_length;
    int win_length;
    int pad;
    torch::Tensor window;
    torch::Tensor mel_basis;

    torch::Tensor dynamic_range_compression_torch(torch::Tensor x, float C=1, float clip_val=1e-5) {
        return torch::log(torch::clamp(x, clip_val) * C);
    }
};

} // namespace fft
