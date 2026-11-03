import torch
from torch import Tensor
from typing import Optional, Tuple


class STEQuantize(torch.autograd.Function):
    """
    A custom autograd function for quantizing tensors using a Straight-Through Estimator (STE).
    The forward pass performs quantization, and the backward pass approximates the gradient
    by allowing gradient flow only for input values within a specified range.
    """

    @staticmethod
    def forward(ctx, inputs: Tensor, quant_limit: float, quant_level: int) -> Tensor:
        """
        Forward pass: clamp the input and quantize it.

        Args:
            ctx: Context object for saving information for backward.
            inputs: Input tensor to quantize.
            quant_limit: Absolute value limit; values will be clamped to [-quant_limit, +quant_limit].
            quant_level: Number of quantization levels.

        Returns:
            Quantized tensor.
        """
        # Save input for use in the backward pass
        ctx.save_for_backward(inputs)

        # Define limits
        abs_limit = quant_limit
        range_limit = 2.0 * abs_limit

        # Clamp inputs to the range [-quant_limit, quant_limit]
        clamped_inputs = torch.clamp(inputs, -abs_limit, abs_limit)

        # Quantize based on the number of levels
        if quant_level == 2:
            # Binary quantization: map to {-1, 0, +1}
            quantized = torch.sign(clamped_inputs)
        else:
            # For multi-bit quantization, first shift the inputs to the [0, range_limit] range.
            shifted_inputs = clamped_inputs + abs_limit
            # Scale the inputs to the range [0, quant_level-1]
            scale_factor = (quant_level - 1) / range_limit
            scaled_inputs = shifted_inputs * scale_factor
            # Round to the nearest quantization level
            quantized_levels = torch.round(scaled_inputs)
            # Scale back to the original range and shift back to [-quant_limit, quant_limit]
            quantized = quantized_levels * (range_limit / (quant_level - 1)) - abs_limit

        return quantized

    @staticmethod
    def backward(
        ctx, grad_output: Tensor
    ) -> Tuple[Tensor, Optional[None], Optional[None]]:
        """
        Backward pass: approximate the gradient using a custom STE.

        Args:
            ctx: Context object with saved tensors.
            grad_output: Gradient of the loss with respect to the output.

        Returns:
            A tuple containing:
              - Gradients with respect to the input.
              - None for quant_limit since it is a hyperparameter.
              - None for quant_level since it is also a hyperparameter.
        """
        # Retrieve the saved input from the forward pass
        (inputs,) = ctx.saved_tensors

        # Create a copy of grad_output to modify
        grad_inputs = grad_output.clone()

        # Zero out gradients for inputs outside the desired region [-1, 1]
        grad_inputs[inputs > 1.0] = 0
        grad_inputs[inputs < -1.0] = 0

        # Optionally, further clamp the gradient magnitudes to stabilize training
        grad_inputs = torch.clamp(grad_inputs, -0.25, 0.25)

        # Return gradients for each input to forward:
        # grad_inputs for the input tensor and None for quant_limit and quant_level.
        return grad_inputs, None, None


class MySign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        # or simply
        # print("in backward, grad_out =", grad_output)

        # print(
        #     "in backward, max grad_out =",
        #     grad_output.max(),
        #     file=open("grad_out.txt", "a"),
        # )
        (input,) = ctx.saved_tensors
        mask = (input.abs() <= 1).float()
        grad_input = grad_output * mask
        # ! optinal: test if the gradient is the key to better performance
        # grad_input = torch.clamp(grad_input, -0.25, 0.25)
        return grad_input


# Usage example:
if __name__ == "__main__":
    # Create a sample tensor of shape (2, 3, 4)
    x = torch.randn(2, 3, 4)
    # x = torch.linspace(-2, 2, steps=10, requires_grad=True)

    # Define quantization parameters
    quant_limit = 1.0
    quant_level = 2

    # Apply our custom function
    quantized_x = STEQuantize.apply(x, quant_limit, quant_level)
    print("Quantized Output:", quantized_x)

    # Perform a dummy backward pass
    quantized_x.sum().backward()
    print("Gradient:", x.grad)
