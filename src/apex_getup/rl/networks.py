"""Small NumPy MLP and Adam implementation used by PPO."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class MLP:
    def __init__(
        self,
        input_size: int,
        hidden_sizes: tuple[int, ...],
        output_size: int,
        rng: np.random.Generator,
        *,
        output_scale: float = 1.0,
    ) -> None:
        sizes = (input_size, *hidden_sizes, output_size)
        self.weights: list[NDArray[np.float64]] = []
        self.biases: list[NDArray[np.float64]] = []
        for index, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            if index == len(sizes) - 2:
                scale *= output_scale
            self.weights.append(rng.normal(0.0, scale, (fan_in, fan_out)))
            self.biases.append(np.zeros(fan_out, dtype=np.float64))

    @property
    def parameters(self) -> list[NDArray[np.float64]]:
        result: list[NDArray[np.float64]] = []
        for weight, bias in zip(self.weights, self.biases):
            result.extend((weight, bias))
        return result

    def forward(
        self, inputs: NDArray[np.float64], *, cache: bool = False
    ) -> NDArray[np.float64] | tuple[NDArray[np.float64], list[NDArray[np.float64]]]:
        activation = np.atleast_2d(np.asarray(inputs, dtype=np.float64))
        activations = [activation]
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            activation = activation @ weight + bias
            if index != len(self.weights) - 1:
                activation = np.tanh(activation)
            activations.append(activation)
        return (activation, activations) if cache else activation

    def backward(
        self,
        output_gradient: NDArray[np.float64],
        activations: list[NDArray[np.float64]],
    ) -> list[NDArray[np.float64]]:
        gradient = output_gradient
        parameter_gradients: list[NDArray[np.float64]] = []
        layer_gradients: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
        for index in range(len(self.weights) - 1, -1, -1):
            input_activation = activations[index]
            weight_gradient = input_activation.T @ gradient
            bias_gradient = np.sum(gradient, axis=0)
            layer_gradients.append((weight_gradient, bias_gradient))
            gradient = gradient @ self.weights[index].T
            if index > 0:
                gradient *= 1.0 - np.square(activations[index])
        for weight_gradient, bias_gradient in reversed(layer_gradients):
            parameter_gradients.extend((weight_gradient, bias_gradient))
        return parameter_gradients


class Adam:
    def __init__(self, parameters: list[NDArray[np.float64]], learning_rate: float):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.first = [np.zeros_like(parameter) for parameter in parameters]
        self.second = [np.zeros_like(parameter) for parameter in parameters]
        self.step_count = 0

    def step(self, gradients: list[NDArray[np.float64]], maximum_norm: float) -> float:
        if len(gradients) != len(self.parameters):
            raise ValueError("gradient count does not match parameter count")
        norm = float(np.sqrt(sum(np.sum(np.square(gradient)) for gradient in gradients)))
        multiplier = min(1.0, maximum_norm / (norm + 1e-8))
        self.step_count += 1
        beta1, beta2 = 0.9, 0.999
        for parameter, gradient, first, second in zip(
            self.parameters, gradients, self.first, self.second
        ):
            gradient = gradient * multiplier
            first *= beta1
            first += (1.0 - beta1) * gradient
            second *= beta2
            second += (1.0 - beta2) * np.square(gradient)
            corrected_first = first / (1.0 - beta1**self.step_count)
            corrected_second = second / (1.0 - beta2**self.step_count)
            parameter -= self.learning_rate * corrected_first / (
                np.sqrt(corrected_second) + 1e-8
            )
        return norm
