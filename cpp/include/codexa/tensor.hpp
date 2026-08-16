#pragma once

#include <cstddef>
#include <initializer_list>
#include <vector>

namespace codexa {

// The first CPU port deliberately keeps the storage model small and explicit:
// row-major contiguous float64 data with a derived stride table.
class Tensor {
 public:
  explicit Tensor(std::vector<std::size_t> shape, double fill = 0.0);
  Tensor(std::vector<std::size_t> shape, std::vector<double> data);

  const std::vector<std::size_t>& shape() const noexcept { return shape_; }
  const std::vector<std::size_t>& strides() const noexcept { return strides_; }
  const std::vector<double>& data() const noexcept { return data_; }
  std::vector<double>& data() noexcept { return data_; }
  std::size_t size() const noexcept { return data_.size(); }

  double& at(const std::vector<std::size_t>& index);
  const double& at(const std::vector<std::size_t>& index) const;
  Tensor reshape(std::vector<std::size_t> new_shape) const;
  double sum() const noexcept;

 private:
  std::size_t flat_index(const std::vector<std::size_t>& index) const;
  static std::size_t element_count(const std::vector<std::size_t>& shape);

  std::vector<std::size_t> shape_;
  std::vector<std::size_t> strides_;
  std::vector<double> data_;
};

Tensor add(const Tensor& left, const Tensor& right);
Tensor subtract(const Tensor& left, const Tensor& right);
Tensor multiply(const Tensor& left, const Tensor& right);

}  // namespace codexa
