#include "codexa/tensor.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>

int main() {
  codexa::Tensor tensor({2, 3}, {1, 2, 3, 4, 5, 6});
  assert((tensor.strides() == std::vector<std::size_t>{3, 1}));
  assert(tensor.at({1, 2}) == 6.0);
  tensor.at({0, 1}) = 20.0;
  assert(tensor.sum() == 39.0);

  const auto reshaped = tensor.reshape({3, 2});
  assert(reshaped.at({1, 0}) == 3.0);
  const auto combined = codexa::add(tensor, tensor);
  assert(combined.at({0, 1}) == 40.0);
  const auto product = codexa::multiply(tensor, tensor);
  assert(product.sum() == 487.0);

  bool rejected = false;
  try {
    (void)codexa::add(tensor, codexa::Tensor({3, 2}, 1.0));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);

  rejected = false;
  try {
    (void)tensor.at({2, 0});
  } catch (const std::out_of_range&) {
    rejected = true;
  }
  assert(rejected);
  return 0;
}
