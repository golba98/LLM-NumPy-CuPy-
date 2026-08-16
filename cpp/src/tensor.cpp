#include "codexa/tensor.hpp"

#include <numeric>
#include <stdexcept>

namespace codexa {

std::size_t Tensor::element_count(const std::vector<std::size_t>& shape) {
  return std::accumulate(shape.begin(), shape.end(), std::size_t{1},
                         [](std::size_t total, std::size_t extent) {
                           if (extent == 0) {
                             throw std::invalid_argument("tensor dimensions must be positive");
                           }
                           return total * extent;
                         });
}

Tensor::Tensor(std::vector<std::size_t> shape, double fill)
    : shape_(std::move(shape)), data_(element_count(shape_), fill) {
  strides_.resize(shape_.size());
  std::size_t stride = 1;
  for (std::size_t axis = shape_.size(); axis-- > 0;) {
    strides_[axis] = stride;
    stride *= shape_[axis];
  }
}

Tensor::Tensor(std::vector<std::size_t> shape, std::vector<double> data)
    : Tensor(std::move(shape), 0.0) {
  if (data.size() != data_.size()) {
    throw std::invalid_argument("tensor data size does not match shape");
  }
  data_ = std::move(data);
}

std::size_t Tensor::flat_index(const std::vector<std::size_t>& index) const {
  if (index.size() != shape_.size()) {
    throw std::out_of_range("tensor index rank does not match shape");
  }
  std::size_t offset = 0;
  for (std::size_t axis = 0; axis < index.size(); ++axis) {
    if (index[axis] >= shape_[axis]) {
      throw std::out_of_range("tensor index is outside shape");
    }
    offset += index[axis] * strides_[axis];
  }
  return offset;
}

double& Tensor::at(const std::vector<std::size_t>& index) {
  return data_[flat_index(index)];
}

const double& Tensor::at(const std::vector<std::size_t>& index) const {
  return data_[flat_index(index)];
}

Tensor Tensor::reshape(std::vector<std::size_t> new_shape) const {
  if (element_count(new_shape) != size()) {
    throw std::invalid_argument("reshape must preserve element count");
  }
  return Tensor(std::move(new_shape), data_);
}

double Tensor::sum() const noexcept {
  return std::accumulate(data_.begin(), data_.end(), 0.0);
}

template <typename Operation>
Tensor elementwise(const Tensor& left, const Tensor& right, Operation operation) {
  if (left.shape() != right.shape()) {
    throw std::invalid_argument("elementwise tensors must have identical shapes");
  }
  std::vector<double> result(left.size());
  for (std::size_t i = 0; i < result.size(); ++i) {
    result[i] = operation(left.data()[i], right.data()[i]);
  }
  return Tensor(left.shape(), std::move(result));
}

Tensor add(const Tensor& left, const Tensor& right) {
  return elementwise(left, right, [](double a, double b) { return a + b; });
}

Tensor subtract(const Tensor& left, const Tensor& right) {
  return elementwise(left, right, [](double a, double b) { return a - b; });
}

Tensor multiply(const Tensor& left, const Tensor& right) {
  return elementwise(left, right, [](double a, double b) { return a * b; });
}

}  // namespace codexa
