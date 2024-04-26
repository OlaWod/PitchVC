/* 
   https://github.com/intel/openvino-plugins-ai-audacity/blob/main/mod-openvino/musicgen/musicgen_utils.h
   https://github.com/openvinotoolkit/openvino/blob/master/samples/cpp/common/format_reader/src/npy.cpp
*/

#pragma once

#include <iostream>
#include <fstream>

#include <openvino/openvino.hpp>
#include <torch/torch.h>


namespace ut {

torch::Tensor load_torch_tensor(const std::string& path) 
{
   std::ifstream rfile(path, std::ios::binary);
   std::vector<char> bytes(
      (std::istreambuf_iterator<char>(rfile)),
      (std::istreambuf_iterator<char>()));
   rfile.close();

   torch::IValue x = torch::pickle_load(bytes);
   torch::Tensor tensor = x.toTensor();

   return tensor;
}


void save_torch_tensor(const torch::Tensor& tensor, const std::string& path) 
{
   std::vector<char> bytes = torch::pickle_save(tensor);

   std::ofstream wfile(path, std::ios::binary);
   wfile.write(bytes.data(), bytes.size());
   wfile.close();
}


ov::Tensor wrap_torch_tensor_as_ov(torch::Tensor tensor_torch)
{

   if (!tensor_torch.defined())
   {
      throw std::runtime_error("wrap_torch_tensor_as_ov: received undefined tensor");
   }
   size_t element_byte_size;
   void* pData = tensor_torch.data_ptr();
   ov::element::Type ov_element_type;
   switch (tensor_torch.dtype().toScalarType())
   {
   case torch::kFloat32:
      ov_element_type = ov::element::f32;
      element_byte_size = sizeof(float);
      break;

   case torch::kFloat16:
      ov_element_type = ov::element::f16;
      element_byte_size = sizeof(short);
      break;

   case torch::kInt64:
      ov_element_type = ov::element::i64;
      element_byte_size = sizeof(int64_t);
      break;
   default:
      std::cout << "type = " << tensor_torch.dtype() << std::endl;
      throw std::invalid_argument("wrap_torch_tensor_as_ov: unsupported type");
      break;
   }

   std::vector<size_t> ov_shape;
   for (auto s : tensor_torch.sizes())
      ov_shape.push_back(s);

   //OV strides are in bytes, whereas torch strides are in # of elements.
   std::vector<size_t> ov_strides;
   for (auto s : tensor_torch.strides())
      ov_strides.push_back(s * element_byte_size);

   return ov::Tensor(ov_element_type, ov_shape, pData, ov_strides);
}


torch::Tensor wrap_ov_tensor_as_torch(ov::Tensor ov_tensor)
{
   if (!ov_tensor)
   {
      throw std::invalid_argument("wrap_ov_tensor_as_torch: invalid ov_tensor");
   }

   //first, determine torch dtype from ov type
   at::ScalarType torch_dtype;
   size_t element_byte_size;
   void* pOV_Tensor;
   switch (ov_tensor.get_element_type())
   {
   case ov::element::i8:
      torch_dtype = torch::kI8;
      element_byte_size = sizeof(unsigned char);
      pOV_Tensor = ov_tensor.data();
      break;

   case ov::element::f32:
      torch_dtype = torch::kFloat32;
      element_byte_size = sizeof(float);
      pOV_Tensor = ov_tensor.data<float>();
      break;

   case ov::element::f16:
      torch_dtype = torch::kFloat16;
      element_byte_size = sizeof(short);
      pOV_Tensor = ov_tensor.data<ov::float16>();
      break;

   case ov::element::i64:
      torch_dtype = torch::kInt64;
      element_byte_size = sizeof(int64_t);
      pOV_Tensor = ov_tensor.data<int64_t>();
      break;

   default:
      std::cout << "type = " << ov_tensor.get_element_type() << std::endl;
      throw std::invalid_argument("wrap_ov_tensor_as_torch: unsupported type");
      break;
   }

   //fill torch shape
   std::vector<int64_t> torch_shape;
   for (auto s : ov_tensor.get_shape())
      torch_shape.push_back(s);

   std::vector<int64_t> torch_strides;
   for (auto s : ov_tensor.get_strides())
      torch_strides.push_back(s / element_byte_size); //<- torch stride is in term of # of elements, whereas openvino stride is in terms of bytes

   auto options =
      torch::TensorOptions()
      .dtype(torch_dtype);

   return torch::from_blob(pOV_Tensor, torch_shape, torch_strides, options);
}


auto load_numpy_data(const std::string& path) 
{
   // open binary file
   std::ifstream file(path, std::ios::binary);

   // full file size
   file.seekg(0, std::ios_base::end);
   size_t full_file_size = static_cast<std::size_t>(file.tellg());
   file.seekg(0, std::ios_base::beg);

   // move to the formal beginning of the file
   std::string magic_string(6, ' ');
   file.read(&magic_string[0], magic_string.size());

   file.ignore(2);
   unsigned short header_size;
   file.read((char*)&header_size, sizeof(header_size));

   std::string header(header_size, ' ');
   file.read(&header[0], header.size());

   // get data shape
   const std::string shape_key = "'shape':";

   int idx = header.find(shape_key);
   int from = header.find('(', idx + shape_key.size()) + 1;
   int to = header.find(')', from);

   std::string shape_data = header.substr(from, to - from);

   std::vector<size_t> shape;

   if (!shape_data.empty()) {
      shape_data.erase(std::remove(shape_data.begin(), shape_data.end(), ','), shape_data.end());

      std::istringstream shape_data_stream(shape_data);
      size_t value;
      while (shape_data_stream >> value) {
         shape.push_back(value);
      }
   }

   // formal file size
   size_t size = full_file_size - static_cast<std::size_t>(file.tellg());

   // read data
   std::shared_ptr<unsigned char> data;

   data.reset(new unsigned char[size], std::default_delete<unsigned char[]>());
   for (size_t i = 0; i < size; i++) {
      unsigned char buffer = 0;
      file.read(reinterpret_cast<char*>(&buffer), sizeof(buffer));
      data.get()[i] = buffer;
   }

   return std::make_tuple(data, shape);
}

} // namespace ut
