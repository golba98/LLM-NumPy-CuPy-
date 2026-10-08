from llm_numpy.tensor import Tensor
from llm_numpy.ops.basic import to_tensor

def mse_loss(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """
    Computes Mean Squared Error (MSE) loss: mean((y_pred - y_true)^2)
    """
    y_pred_t = to_tensor(y_pred)
    y_true_t = to_tensor(y_true)
    diff = y_pred_t - y_true_t
    return (diff ** 2).mean()
