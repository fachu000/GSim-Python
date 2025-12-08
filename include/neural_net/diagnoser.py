"""
This module provides tools for diagnosing issues in neural network training,
such as NaN or Inf values in gradients and intermediate outputs.
"""

import logging
import os
import tempfile
import numpy as np
import torch
from torch import nn
from pyparsing import Callable, Any
from torch import Tensor
from gsim.include.neural_net.defs import InputType, LossFunType, TargetType
from gsim.include.neural_net.neural_net import Diagnoser, NeuralNet
from gsim.gfigure import GFigure

logger = logging.getLogger("gsim")

# Forward passes ###########################################################


def do_log_forward(model: nn.Module, loss: torch.Tensor):
    loss_this_batch = torch.mean(loss).item()

    all_params = []
    for _, param in model.named_parameters():
        all_params.append(param.data.detach().cpu().flatten().numpy())

    all_params_concat = np.concatenate(all_params)
    param_mean = np.mean(all_params_concat)
    param_std = np.std(all_params_concat)
    param_max_abs = np.max(np.abs(all_params_concat))

    logger.info(f"Forward pass: loss={loss_this_batch:.6f}, "
                f"param_mean={param_mean:.6e}, param_std={param_std:.6e}, "
                f"param_max_abs={param_max_abs:.6e}")


def check_parameters(model: nn.Module,
                     verbose: bool = True,
                     large_param_threshold: float = 1e6) -> dict[str, Any]:
    """
    Check model parameters for NaN, Inf, and extremely large values.
    """
    results = {
        'has_nan': False,
        'has_inf': False,
        'has_large': False,
        'nan_params': [],
        'inf_params': [],
        'large_params': [],
        'params_ok': False,
    }

    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            results['has_nan'] = True
            results['nan_params'].append(name)
            if verbose:
                logger.warning(f"NaN parameter in {name}")

        if torch.isinf(param).any():
            results['has_inf'] = True
            results['inf_params'].append(name)
            if verbose:
                logger.warning(f"Inf parameter in {name}")

        if param.abs().max() > large_param_threshold:
            results['has_large'] = True
            results['large_params'].append((name, param.abs().max().item()))
            if verbose:
                logger.warning(
                    f"Large parameter in {name}: {param.abs().max().item():.2e}"
                )

    results['params_ok'] = not (results['has_nan'] or results['has_inf']
                                or results['has_large'])
    return results


# Backward passes ###########################################################


def do_log_backward(model: nn.Module):
    all_grads = []
    for _, param in model.named_parameters():
        if param.grad is not None:
            all_grads.append(param.grad.detach().cpu().flatten().numpy())

    if not all_grads:
        logger.info("No gradients to log.")
        return

    all_grads_concat = np.concatenate(all_grads)
    grad_mean = np.mean(all_grads_concat)
    grad_2_norm = np.linalg.norm(all_grads_concat)
    grad_inf_norm = np.max(np.abs(all_grads_concat))

    logger.info(
        f"Backward pass: grad_mean={grad_mean:.2e}, "
        f"grad_2_norm={grad_2_norm:.2e}, grad_inf_norm={grad_inf_norm:.2e}")


def gradient_norm(model: nn.Module) -> float:
    """Returns the Frobenius norm of the gradient."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item()**2
    return total_norm**0.5


def check_gradients(model: nn.Module,
                    verbose: bool = True,
                    large_gradient_threshold: float = 1e6) -> dict[str, Any]:
    """
    Check gradients for NaN, Inf, and extremely large values.
    
    Returns dict with diagnostic information.
    """
    results = {
        'has_nan': False,
        'has_inf': False,
        'has_large': False,
        'nan_params': [],
        'inf_params': [],
        'large_params': [],
        'max_grad': 0.0,
        'min_grad': 0.0,
        'gradient_norm': gradient_norm(model),
        'gradient_ok': True,
    }

    max_grad = float('-inf')
    min_grad = float('inf')

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad

            grad_max = grad.max().item()
            grad_min = grad.min().item()
            max_grad = max(max_grad, grad_max)
            min_grad = min(min_grad, grad_min)

            if torch.isnan(grad).any():
                results['has_nan'] = True
                results['nan_params'].append(name)
                if verbose:
                    logger.warning(f"NaN gradient in {name}")

            if torch.isinf(grad).any():
                results['has_inf'] = True
                results['inf_params'].append(name)
                if verbose:
                    logger.warning(f"Inf gradient in {name}")

            if grad.abs().max() > large_gradient_threshold:
                results['has_large'] = True
                results['large_params'].append((name, grad.abs().max().item()))
                if verbose:
                    logger.warning(
                        f"Large gradient in {name}: {grad.abs().max().item():.2e}"
                    )

    results['max_grad'] = max_grad if max_grad != float('-inf') else 0.0
    results['min_grad'] = min_grad if min_grad != float('inf') else 0.0

    results['gradient_ok'] = not (results['has_nan'] or results['has_inf']
                                  or results['has_large'])

    return results


def do_check_repeated_gradient_computation(
        model: 'NeuralNet', data: tuple[InputType, TargetType],
        f_loss: LossFunType) -> dict[str, Any]:
    """
    Computes gradients twice and compares them to detect non-deterministic behavior.
    
    Returns dict with comparison results including whether gradients differ
    and details about which parameters changed.
    """

    # First gradient computation
    #
    # Due to a bug in the MPS PyTorch backend, two forward passes can result in
    # a different gradient.
    loss = model._get_loss(data, f_loss)
    loss = model._get_loss(data, f_loss)
    model.zero_grad()
    logger.info(
        f"First loss: {torch.mean(loss).item():.6f}, has_nan={torch.isnan(loss).any()}, has_inf={torch.isinf(loss).any()}"
    )
    torch.mean(loss).backward()

    # Save gradients for comparison
    gradients_1 = {
        name: param.grad.clone() if param.grad is not None else None
        for name, param in model.named_parameters()
    }

    # Second gradient computation
    model.zero_grad()
    loss = model._get_loss(data, f_loss)
    logger.info(
        f"Second loss: {torch.mean(loss).item():.6f}, has_nan={torch.isnan(loss).any()}, has_inf={torch.isinf(loss).any()}"
    )
    torch.mean(loss).backward()

    # Check if recomputed gradients differ
    logger.info("Comparing original vs recomputed gradients:")
    differs = []
    for name, param in model.named_parameters():
        if param.grad is not None and gradients_1[name] is not None:
            grad_1 = gradients_1[name]
            assert grad_1 is not None  # Type checker hint
            diff = (param.grad - grad_1).abs().max().item()
            if diff > 1e-6:
                logger.warning(f"{name}: gradients differ by {diff:.2e}")
                differs.append((name, diff))

    has_nan_first = any(
        torch.isnan(grad).any() for grad in gradients_1.values()
        if grad is not None)
    has_nan_second = any(
        torch.isnan(param.grad).any() for _, param in model.named_parameters()
        if param.grad is not None)
    logger.info(f"Gradients differ = {len(differs) > 0}")

    return {
        'gradients_differ': len(differs) > 0,
        'differing_params': differs,
        'has_nan_first': has_nan_first,
        'has_nan_second': has_nan_second,
    }


def do_plot_parameter_histograms(
        model: 'NeuralNet',
        bins: int = 50,
        max_curves_per_subplot: int = 8) -> list[GFigure]:
    """
    Constructs a GFigure object where each curve is the histogram of a parameter
    tensor.
    
    Args:
        model: NeuralNet model whose parameters will be plotted.
        bins: Number of bins for the histograms.
        
    Returns:
        GFigure object with histograms for each parameter tensor.
    """
    l_G: list[GFigure] = []

    for ind, (name, param) in enumerate(model.named_parameters()):
        param_data = param.detach().cpu().flatten().numpy()

        if ind % max_curves_per_subplot == 0:
            G = GFigure(xlabel='Parameter Value',
                        ylabel='Frequency',
                        title='Parameter Histograms')
            l_G.append(G)

        l_G[ind // max_curves_per_subplot].add_histogram_curve(
            data=param_data,
            hist_args={
                'bins': bins,
                'density': True
            },
            styles=f'#{ind}',
            legend=name)

    return l_G


def do_plot_gradient_histograms(
        model: 'NeuralNet',
        data: tuple[InputType, TargetType],
        f_loss: LossFunType,
        bins: int = 50,
        max_curves_per_subplot: int = 8) -> list['GFigure']:
    """
    Constructs GFigure objects where each curve is the histogram of a gradient
    tensor.
    
    Args:
        model: NeuralNet model.
        data: Tuple of (input, target) data.
        f_loss: Loss function.
        bins: Number of bins for the histograms.
        max_curves_per_subplot: Maximum number of curves per subplot.
        
    Returns:
        List of GFigure objects with histograms for each gradient tensor.
    """
    # Obtain the gradients
    model.zero_grad()
    loss = model._get_loss(data, f_loss)
    torch.mean(loss).backward()

    l_G: list[GFigure] = []

    for ind, (name, param) in enumerate(model.named_parameters()):
        if ind % max_curves_per_subplot == 0:
            G = GFigure(xlabel='Gradient Value',
                        ylabel='Frequency',
                        title='Gradient Histograms')
            l_G.append(G)

        if param.grad is not None:
            grad_data = param.grad.detach().cpu().flatten().numpy()

            l_G[ind // max_curves_per_subplot].add_histogram_curve(
                data=grad_data,
                hist_args={
                    'bins': bins,
                    'density': True
                },
                styles=f'#{ind}',
                legend=name)

    return l_G


def do_plot_gradient_histogram(model: 'NeuralNet',
                               data: tuple[InputType, TargetType],
                               f_loss: LossFunType,
                               bins: int = 50) -> GFigure:
    """
    Constructs a GFigure object where the curve is the histogram of all gradients
    combined.
    
    Args:
        model: NeuralNet model.
        data: Tuple of (input, target) data.
        f_loss: Loss function.
        bins: Number of bins for the histogram.
        
    Returns:
        GFigure object with histogram for all gradients combined.
    """
    # Obtain the gradients
    model.zero_grad()
    loss = model._get_loss(data, f_loss)
    torch.mean(loss).backward()

    all_grads = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_data = param.grad.detach().cpu().flatten().numpy()
            all_grads.append(grad_data)

    all_grads_concat = np.concatenate(all_grads)

    G = GFigure(xlabel='Gradient Value',
                ylabel='Frequency',
                title='Gradient Histogram')

    G.add_histogram_curve(data=all_grads_concat,
                          hist_args={
                              'bins': bins,
                              'density': True
                          },
                          styles='#0',
                          legend='All Gradients Combined')

    return G


def do_plot_loss_vs_parameter_with_greatest_gradient_abs(
        model: 'NeuralNet', data: tuple[InputType,
                                        TargetType], f_loss: LossFunType,
        num_points: int, delta_range: tuple[float, float]) -> GFigure:
    """
    Plots loss vs perturbation along the parameter with greatest absolute gradient.

    This helps determining if the gradient is correctly computed.
    
    This function plots:
    1. f(delta) = loss(parameters + delta * indicator), where indicator is a 
       one-hot vector with 1 at the position of the parameter with greatest 
       absolute gradient.
    2. g(delta) = f(0) + delta * grad, the linear approximation using the gradient.
    
    Args:
        model: NeuralNet model.
        data: Tuple of (input, target) data.
        f_loss: Loss function.
        num_points: Number of points to sample along delta axis.
        delta_range: Tuple (min_delta, max_delta) for perturbation range.
        
    Returns:
        GFigure object with the loss curves.
    """

    # Store original training mode and set to eval
    was_training = model.training
    model.eval()

    # Compute gradients
    model.zero_grad()
    loss_0 = model._get_loss(data, f_loss)
    loss_0_mean = torch.mean(loss_0)
    loss_0_mean.backward()

    # Find parameter with greatest absolute gradient
    max_grad_abs = 0.0
    max_grad_param = None
    max_grad_name = None
    max_grad_value = 0.0
    max_grad_idx = None

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_abs_max = param.grad.abs().max().item()
            if grad_abs_max > max_grad_abs:
                max_grad_abs = grad_abs_max
                max_grad_param = param
                max_grad_name = name
                # Find the index of the maximum gradient
                flat_grad = param.grad.flatten()
                max_grad_idx = flat_grad.abs().argmax()
                max_grad_value = flat_grad[max_grad_idx].item()

    if max_grad_param is None:
        logger.warning("No gradients found")
        if was_training:
            model.train()
        return GFigure(title="No gradients found")

    logger.info(f"Parameter with greatest gradient: {max_grad_name}")
    logger.info(f"Gradient value: {max_grad_value:.6e}")
    logger.info(
        "Plotting loss vs parameter with greatest absolute gradient...")

    # Sample deltas
    v_delta = torch.linspace(delta_range[0],
                             delta_range[1],
                             num_points,
                             device=max_grad_param.device)
    v_loss_actual = []
    v_loss_linear = []

    # Save original parameter value
    original_param = max_grad_param.data.clone()

    for delta in v_delta:
        # Create a perturbed copy of the parameter
        perturbed_param = original_param.clone()
        flat_perturbed = perturbed_param.flatten()
        flat_perturbed[max_grad_idx] = flat_perturbed[max_grad_idx] + delta

        # Reshape back to original shape
        perturbed_param = flat_perturbed.reshape(original_param.shape)

        # Set the perturbed parameter
        max_grad_param.data = perturbed_param

        # Compute loss
        with torch.no_grad():
            loss = model._get_loss(data, f_loss)
            loss_mean = torch.mean(loss).item()
            v_loss_actual.append(loss_mean)

        # Linear approximation: f(0) + delta * grad
        loss_linear = loss_0_mean.item() + delta.item() * max_grad_value
        v_loss_linear.append(loss_linear)

    # Restore original parameter
    max_grad_param.data = original_param

    # Restore original training mode
    if was_training:
        model.train()

    # Create plot
    v_delta_np = v_delta.cpu().numpy()

    G = GFigure(
        xlabel='Delta (perturbation)',
        ylabel='Loss',
        title=
        f'Loss vs Perturbation\n(param: {max_grad_name}, grad: {max_grad_value:.3e})',
        ylim=(min(v_loss_actual), max(v_loss_actual)))

    G.add_curve(v_delta_np,
                v_loss_actual,
                styles='o-#0',
                legend='Loss in this batch f(delta)')
    G.add_curve(v_delta_np,
                v_loss_linear,
                styles='--#1',
                legend='Linear approx g(delta)')

    return G


# Saving logic ###########################################################


def save_snapshot(model: 'NeuralNet', loss: torch.Tensor,
                  data: tuple[InputType, TargetType], results: dict[str, Any]):
    if model.nn_folder is not None:
        folder = os.path.join(model.nn_folder, 'diagnosing')
        os.makedirs(folder, exist_ok=True)
    else:
        folder = tempfile.mkdtemp(prefix='diagnosing_')

    existing_snapshots = [
        f for f in os.listdir(folder)
        if f.startswith('snapshot_') and f.endswith('.pth')
    ]

    if existing_snapshots:
        indices = []
        for f in existing_snapshots:
            try:
                idx = int(f.replace('snapshot_', '').replace('.pth', ''))
                indices.append(idx)
            except ValueError:
                continue
        n = max(indices) + 1 if indices else 1
    else:
        n = 1

    snapshot_path = os.path.join(folder, f'snapshot_{n}.pth')

    snapshot = {
        'loss': loss,
        'data': data,
        'results': results,
        'weights': model.state_dict()
    }

    torch.save(snapshot, snapshot_path)
    logger.info(f"Snapshot saved to: {snapshot_path}")


def load_snapshot(snapshot_spec: int | str,
                  model: 'NeuralNet') -> dict[str, Any]:
    """
    Args: 
        snapshot_spec: Either an integer index of the snapshot to load from
        the model's diagnosing folder, or a full path to a snapshot file.
        
        model: The neural network model instance to load weights into.
    
    """

    if isinstance(snapshot_spec, int):
        assert model.nn_folder is not None, "model.nn_folder must be set when using integer snapshot_spec"
        snapshot_path = os.path.join(model.nn_folder, 'diagnosing',
                                     f'snapshot_{snapshot_spec}.pth')
    else:
        snapshot_path = snapshot_spec

    snapshot = torch.load(snapshot_path, map_location=model.device_type)

    model.load_state_dict(snapshot['weights'])
    model.to(device=model.device_type)

    return {
        'loss': snapshot['loss'],
        'data': snapshot['data'],
        'results': snapshot['results'],
        'model': model
    }


def reproduce_from_snapshot(snapshot_spec: int | str, model: 'NeuralNet',
                            f_loss: LossFunType) -> torch.Tensor:
    snapshot = load_snapshot(snapshot_spec, model)
    data = snapshot['data']

    # Perform a forward and a backward pass to reproduce the issue
    model.zero_grad()
    loss = model._get_loss(data, f_loss)
    torch.mean(loss).backward()

    return loss


class StandardDiagnoser(Diagnoser):

    def __init__(
        self,
        log_forward: bool = False,
        check_parameters: bool = False,
        log_backward: bool = False,
        check_gradients: bool = False,
        check_repeated_gradient_computation: bool = False,
        plot_parameter_histograms: bool = False,
        plot_gradient_histograms: bool = False,
        plot_gradient_histogram: bool = False,
        plot_loss_vs_parameter_with_greatest_gradient_abs: bool = False,
        num_points_loss_vs_parameter_with_greatest_gradient_abs: int = 21,
        delta_range_loss_vs_parameter_with_greatest_gradient_abs: tuple[
            float, float] = (-0.1, 0.1),
    ):
        """
        Args:

            log_forward: If True, logs the loss value and other statistics after
            every training forward pass (batch).

            check_parameters: If True, checks for NaN/Inf/large values in model
            parameters after every training forward pass (batch).     

            log_backward: If True, logs gradient statistics after every training
            backward pass.

            check_gradients: If True, checks for NaN/Inf/large values in model
            gradients after every training backward pass. If issues are found, a
            snapshot before the optimizer update is saved for later analysis.

            check_repeated_gradient_computation: If True, computes gradients
            twice after every training backward pass and compares them to detect
            non-deterministic behavior.

            plot_parameter_histograms: If True, it plots histograms of model
            parameters after every training backward pass.

            plot_gradient_histograms: If True, it plots histograms of model
            gradients after every training backward pass.

            plot_gradient_histogram: If True, it plots a single histogram of all
            gradients after every training backward pass.

            plot_loss_vs_parameter_with_greatest_gradient_abs: If True, it plots
            the loss versus the parameter with the greatest absolute gradient
            after every training backward pass. It is useful to see if the
            gradient is correctly computed. 

            num_points_loss_vs_parameter_with_greatest_gradient_abs: Number of
            points in the x-axis for the loss vs parameter plot.

            delta_range_loss_vs_parameter_with_greatest_gradient_abs: Tuple
            (min_delta, max_delta) defining the range of perturbations for the
            loss vs parameter plot.
            
            """
        super().__init__()
        self.log_forward = log_forward
        self.check_parameters = check_parameters
        self.log_backward = log_backward
        self.check_gradients = check_gradients
        self.check_repeated_gradient_computation = check_repeated_gradient_computation
        self.plot_parameter_histograms = plot_parameter_histograms
        self.plot_gradient_histograms = plot_gradient_histograms
        self.plot_gradient_histogram = plot_gradient_histogram
        self.plot_loss_vs_parameter_with_greatest_gradient_abs = plot_loss_vs_parameter_with_greatest_gradient_abs
        self.num_points__loss_vs_parameter_with_greatest_gradient_abs = num_points_loss_vs_parameter_with_greatest_gradient_abs
        self.delta_range__loss_vs_parameter_with_greatest_gradient_abs = delta_range_loss_vs_parameter_with_greatest_gradient_abs

    def check_forward(self, model: 'NeuralNet', loss: torch.Tensor,
                      data: tuple[InputType, TargetType], f_loss: LossFunType):
        """
        This method is run after every forward pass (batch).
        """

        if self.check_parameters:
            results = check_parameters(model, verbose=True)

            if not results['params_ok']:
                logger.error("Parameter check failed during forward pass.")
                save_snapshot(model, loss, data, results)
                # We repeat the forward pass to help with debugging.
                # This is a good place to set a breakpoint.
                _ = model._get_loss(data, f_loss)

        if self.log_forward:
            do_log_forward(model, loss)

        return

    def check_backward(self, model: 'NeuralNet', loss: torch.Tensor,
                       data: tuple[InputType,
                                   TargetType], f_loss: LossFunType):
        """
        This method is run after every training backward pass.
        """

        if self.log_backward:
            do_log_backward(model)

        if self.check_repeated_gradient_computation:
            comparison = do_check_repeated_gradient_computation(
                model, data, f_loss)
            if comparison['gradients_differ']:
                logger.warning(
                    "Non-deterministic gradient computation detected! "
                    f"{len(comparison['differing_params'])} parameters differ."
                )
            else:
                logger.info(
                    "Gradients are deterministic (identical on recomputation)."
                )

        if self.check_gradients:
            results = check_gradients(model, verbose=True)
            if not results['gradient_ok']:
                logger.error("Gradient check failed during backward pass.")
                save_snapshot(model, loss, data, results)

        # Figures
        l_G: list[GFigure] = []
        if self.plot_parameter_histograms:
            l_G += do_plot_parameter_histograms(model)
        if self.plot_gradient_histograms:
            l_G += do_plot_gradient_histograms(model, data, f_loss)
        if self.plot_gradient_histogram:
            l_G.append(do_plot_gradient_histogram(model, data, f_loss))
        if self.plot_loss_vs_parameter_with_greatest_gradient_abs:
            l_G.append(
                do_plot_loss_vs_parameter_with_greatest_gradient_abs(
                    model, data, f_loss, self.
                    num_points__loss_vs_parameter_with_greatest_gradient_abs,
                    self.
                    delta_range__loss_vs_parameter_with_greatest_gradient_abs))
        if l_G:
            [g.plot() for g in l_G]
            l_G[0].show()

        return
