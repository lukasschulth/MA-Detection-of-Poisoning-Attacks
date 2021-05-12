import torch
import numpy as np
import torch.nn.functional as F
from .utils import pprint, Flatten
from.dummpy_model import New_parallel_chain_dummy
#import TrafficSignAI

def module_tracker_parallel(fwd_hook_func):
    """
    Wrapper for tracking the layers throughout the forward pass.

    Args:
        fwd_hook_func: Forward hook function to be wrapped.

    Returns:
        Wrapped method.

    """
    def hook_wrapper(relevance_propagator_instance, layer, *args):
        if relevance_propagator_instance.module_list is not None and len(relevance_propagator_instance.module_list) > 0 and isinstance(relevance_propagator_instance.module_list[-1], list):
            relevance_propagator_instance.module_list[-1][-1].append(layer)
        else:
            # relevance_propagator_instance.module_list.append([layer])
            raise NotImplementedError("This should never happen")
        return fwd_hook_func(relevance_propagator_instance, layer, *args)

    return hook_wrapper

def module_tracker_parallel_new_path(fwd_hook_func):
    """
    Wrapper for tracking the layers throughout the forward pass.

    Args:
        fwd_hook_func: Forward hook function to be wrapped.

    Returns:
        Wrapped method.

    """
    def hook_wrapper(relevance_propagator_instance, layer, *args):
        if relevance_propagator_instance.module_list is not None and len(
                relevance_propagator_instance.module_list) > 0 and isinstance(relevance_propagator_instance.module_list[-1], list):
            relevance_propagator_instance.module_list[-1].append([])
        else:
            relevance_propagator_instance.module_list.append([[]])
        return fwd_hook_func(relevance_propagator_instance, layer, *args)

    return hook_wrapper

def module_tracker(fwd_hook_func):
    """
    Wrapper for tracking the layers throughout the forward pass.

    Args:
        fwd_hook_func: Forward hook function to be wrapped.

    Returns:
        Wrapped method.

    """
    def hook_wrapper(relevance_propagator_instance, layer, *args):
        relevance_propagator_instance.module_list.append(layer)

        return fwd_hook_func(relevance_propagator_instance, layer, *args)

    return hook_wrapper


class RelevancePropagator:
    """
    Class for computing the relevance propagation and supplying
    the necessary forward hooks for all layers.
    """

    # All layers that do not require any specific forward hooks.
    # This is due to the fact that they are all one-to-one
    # mappings and hence no normalization is needed (each node only
    # influences exactly one other node -> relevance conservation
    # ensures that relevance is just inherited in a one-to-one manner, too).
    #import TrafficSignAI
    from coding.Aenderungen_LRP.TrafficSignAI.LRP.dummpy_model import New_parallel_chain_dummy
    from coding.Aenderungen_LRP.TrafficSignAI.Models.Net import Cat
    allowed_pass_layers = (torch.nn.ReLU, torch.nn.ELU, Flatten,
                           torch.nn.Dropout, torch.nn.Dropout2d,
                           torch.nn.Dropout3d,
                           torch.nn.Softmax,
                           torch.nn.LogSoftmax,
                           torch.nn.Sigmoid,
                           torch.nn.InstanceNorm2d,
                           Cat)
    batchNorm_layers = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                        torch.nn.BatchNorm3d)

    # Implemented rules for relevance propagation.
    available_methods = ["e-rule", "b-rule"]

    def __init__(self, lrp_exponent, beta, method, epsilon, device):

        self.device = device
        self.layer = None
        self.p = lrp_exponent
        self.beta = beta
        self.eps = epsilon
        self.warned_log_softmax = False
        self.module_list = []
        if method not in self.available_methods:
            raise NotImplementedError("Only methods available are: " +
                                      str(self.available_methods))
        self.method = method
        self.batchNorm_id = 0
        self.batchNorm_dict = {}

    def reset_module_list(self):
        """
        The module list is reset for every evaluation, in change the order or number
        of layers changes dynamically.

        Returns:
            None

        """
        self.module_list = []
        # Try to free memory
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def compute_propagated_relevance(self, layer, relevance):

        """
        This method computes the backward pass for the incoming relevance
        for the specified layer.

        Args:
            layer: Layer to be reverted.
            relevance: Incoming relevance from higher up in the network.

        Returns:
            The relevance of the current layer

        """
        if isinstance(layer,
                      (torch.nn.MaxPool1d, torch.nn.MaxPool2d, torch.nn.MaxPool3d)):
            return self.max_pool_nd_inverse(layer, relevance).detach()

        elif isinstance(layer, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
            return self.conv_nd_inverse(layer, relevance).detach()

        elif isinstance(layer, torch.nn.LogSoftmax):
            # Only layer that does not conserve relevance. Mainly used
            # to make probability out of the log values. Should probably
            # be changed to pure passing and the user should make sure
            # the layer outputs are sensible (0 would be 100% class probability,
            # but no relevance could be passed on).
            if relevance.sum() < 0:
                relevance[relevance == 0] = -1e6
                relevance = relevance.exp()
                if not self.warned_log_softmax:
                    pprint("WARNING: LogSoftmax layer was "
                           "turned into probabilities.")
                    self.warned_log_softmax = True
            return relevance

        elif isinstance(layer, self.batchNorm_layers):

            print('Batch Normalization detected')
            # Increase batch_normID
            self.batchNorm_id += 1
            #print(layer.weight.shape)
            #print(layer.bias.shape)
            return self.batch_nd_inverse(layer, relevance).detach()
            #return relevance

        elif isinstance(layer, self.allowed_pass_layers):
            # The above layers are one-to-one mappings of input to
            # output nodes. All the relevance in the output will come
            # entirely from the input node. Given the conservation
            # of relevance, the input is as relevant as the output.
            return relevance

        elif isinstance(layer, torch.nn.Linear):
            return self.linear_inverse(layer, relevance).detach()
        else:
            raise NotImplementedError("The network contains layers that"
                                      " are currently not supported {0:s}".format(str(layer)))

    def get_layer_fwd_hook(self, layer, parallel=False):
        """
        Each layer might need to save very specific data during the forward
        pass in order to allow for relevance propagation in the backward
        pass. For example, for max_pooling, we need to store the
        indices of the max values. In convolutional layers, we need to calculate
        the normalizations, to ensure the overall amount of relevance is conserved.

        Args:
            layer: Layer instance for which forward hook is needed.

        Returns:
            Layer-specific forward hook.

        """
        # print("LAYER:", layer, isinstance(layer, TrafficSignAI.LRP.dummpy_model.New_parallel_chain_dummy))
        #if isinstance(layer, (TrafficSignAI.LRP.dummpy_model.New_parallel_chain_dummy)):
        if isinstance(layer, New_parallel_chain_dummy):
            return self.new_parall_path

        if parallel is False:
            if isinstance(layer,
                          (torch.nn.MaxPool1d, torch.nn.MaxPool2d, torch.nn.MaxPool3d)):
                return self.max_pool_nd_fwd_hook

            if isinstance(layer,
                          (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
                return self.conv_nd_fwd_hook

            if isinstance(layer, self.allowed_pass_layers):
                return self.silent_pass  # No hook needed.

            if isinstance(layer, torch.nn.Linear):
                return self.linear_fwd_hook

            if isinstance(layer, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                return self.batchNorm_fwd_hook

        else:
            if isinstance(layer,
                          (torch.nn.MaxPool1d, torch.nn.MaxPool2d, torch.nn.MaxPool3d)):
                return self.max_pool_nd_fwd_hook_parallel

            if isinstance(layer,
                          (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
                return self.linear_fwd_hook_parallel

            if isinstance(layer, self.allowed_pass_layers):
                return self.silent_pass_parallel  # No hook needed.

            if isinstance(layer, torch.nn.Linear):
                return self.linear_fwd_hook_parallel

            if isinstance(layer, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                return self.batchNorm_fwd_hook_parallel

        raise NotImplementedError("The network contains layers that"
                                      " are currently not supported {0:s}".format(str(layer)))

    @staticmethod
    def get_conv_method(conv_module):
        """
        Get dimension-specific convolution.
        The forward pass and inversion are made in a
        'dimensionality-agnostic' manner and are the same for
        all nd instances of the layer, except for the functional
        that needs to be used.

        Args:
            conv_module: instance of convolutional layer.

        Returns:
            The correct functional used in the convolutional layer.

        """

        conv_func_mapper = {
            torch.nn.Conv1d: F.conv1d,
            torch.nn.Conv2d: F.conv2d,
            torch.nn.Conv3d: F.conv3d
        }
        return conv_func_mapper[type(conv_module)]

    @staticmethod
    def get_inv_conv_method(conv_module):
        """
        Get dimension-specific convolution inversion layer.
        The forward pass and inversion are made in a
        'dimensionality-agnostic' manner and are the same for
        all nd instances of the layer, except for the functional
        that needs to be used.

        Args:
            conv_module: instance of convolutional layer.

        Returns:
            The correct functional used for inverting the convolutional layer.

        """

        conv_func_mapper = {
            torch.nn.Conv1d: F.conv_transpose1d,
            torch.nn.Conv2d: F.conv_transpose2d,
            torch.nn.Conv3d: F.conv_transpose3d
        }
        return conv_func_mapper[type(conv_module)]

    @module_tracker_parallel_new_path
    def new_parall_path(self, m, in_tensor: torch.Tensor, out_tensor: torch.Tensor):
        # Placeholder forward hook for layers that do not need
        # to store any specific data. Still useful for module tracking.
        pass

    @module_tracker_parallel
    def silent_pass_parallel(self, m, in_tensor: torch.Tensor,
                    out_tensor: torch.Tensor):
        # Placeholder forward hook for layers that do not need
        # to store any specific data. Still useful for module tracking.
        pass


    @module_tracker
    def silent_pass(self, m, in_tensor: torch.Tensor,
                    out_tensor: torch.Tensor):
        # Placeholder forward hook for layers that do not need
        # to store any specific data. Still useful for module tracking.
        pass

    @staticmethod
    def get_inv_max_pool_method(max_pool_instance):
        """
        Get dimension-specific max_pooling layer.
        The forward pass and inversion are made in a
        'dimensionality-agnostic' manner and are the same for
        all nd instances of the layer, except for the functional
        that needs to be used.

        Args:
            max_pool_instance: instance of max_pool layer.

        Returns:
            The correct functional used in the max_pooling layer.

        """

        conv_func_mapper = {
            torch.nn.MaxPool1d: F.max_unpool1d,
            torch.nn.MaxPool2d: F.max_unpool2d,
            torch.nn.MaxPool3d: F.max_unpool3d
        }
        return conv_func_mapper[type(max_pool_instance)]

    def linear_inverse(self, m, relevance_in):

        if self.method == "e-rule":
            m.in_tensor = m.in_tensor.pow(self.p)
            w = m.weight.pow(self.p)
            norm = F.linear(m.in_tensor, w, bias=None)

            norm = norm + torch.sign(norm) * self.eps
            relevance_in[norm == 0] = 0
            norm[norm == 0] = 1
            relevance_out = F.linear(relevance_in / norm,
                                     w.t(), bias=None)
            relevance_out *= m.in_tensor
            del m.in_tensor, norm, w, relevance_in
            return relevance_out

        if self.method == "b-rule":
            out_c, in_c = m.weight.size()
            w = m.weight.repeat((4, 1))
            # First and third channel repetition only contain the positive weights
            w[:out_c][w[:out_c] < 0] = 0
            w[2 * out_c:3 * out_c][w[2 * out_c:3 * out_c] < 0] = 0
            # Second and fourth channel repetition with only the negative weights
            w[1 * out_c:2 * out_c][w[1 * out_c:2 * out_c] > 0] = 0
            w[-out_c:][w[-out_c:] > 0] = 0

            # Repeat across channel dimension (pytorch always has channels first)
            m.in_tensor = m.in_tensor.repeat((1, 4))
            m.in_tensor[:, :in_c][m.in_tensor[:, :in_c] < 0] = 0
            m.in_tensor[:, -in_c:][m.in_tensor[:, -in_c:] < 0] = 0
            m.in_tensor[:, 1 * in_c:3 * in_c][m.in_tensor[:, 1 * in_c:3 * in_c] > 0] = 0

            # Normalize such that the sum of the individual importance values
            # of the input neurons divided by the norm
            # yields 1 for an output neuron j if divided by norm (v_ij in paper).
            # Norm layer just sums the importance values of the inputs
            # contributing to output j for each j. This will then serve as the normalization
            # such that the contributions of the neurons sum to 1 in order to
            # properly split up the relevance of j amongst its roots.

            norm_shape = m.out_shape
            norm_shape[1] *= 4
            norm = torch.zeros(norm_shape).to(self.device)

            for i in range(4):
                norm[:, out_c * i:(i + 1) * out_c] = F.linear(
                    m.in_tensor[:, in_c * i:(i + 1) * in_c], w[out_c * i:(i + 1) * out_c], bias=None)

            # Double number of output channels for positive and negative norm per
            # channel.
            norm_shape[1] = norm_shape[1] // 2
            new_norm = torch.zeros(norm_shape).to(self.device)
            new_norm[:, :out_c] = norm[:, :out_c] + norm[:, out_c:2 * out_c]
            new_norm[:, out_c:] = norm[:, 2 * out_c:3 * out_c] + norm[:, 3 * out_c:]
            norm = new_norm

            # Some 'rare' neurons only receive either
            # only positive or only negative inputs.
            # Conservation of relevance does not hold, if we also
            # rescale those neurons by (1+beta) or -beta.
            # Therefore, catch those first and scale norm by
            # the according value, such that it cancels in the fraction.

            # First, however, avoid NaNs.
            mask = norm == 0
            # Set the norm to anything non-zero, e.g. 1.
            # The actual inputs are zero at this point anyways, that
            # is why norm is zero in the first place.
            norm[mask] = 1
            # The norm in the b-rule has shape (N, 2*out_c, *spatial_dims).
            # The first out_c block corresponds to the positive norms,
            # the second out_c block corresponds to the negative norms.
            # We find the rare neurons by choosing those nodes per channel
            # in which either the positive norm ([:, :out_c]) is zero, or
            # the negative norm ([:, :out_c]) is zero.
            rare_neurons = (mask[:, :out_c] + mask[:, out_c:])

            # Also, catch new possibilities for norm == zero to avoid NaN..
            # The actual value of norm again does not really matter, since
            # the pre-factor will be zero in this case.

            norm[:, :out_c][rare_neurons] *= 1 if self.beta == -1 else 1 + self.beta
            norm[:, out_c:][rare_neurons] *= 1 if self.beta == 0 else -self.beta
            # Add stabilizer term to norm to avoid numerical instabilities.
            norm += self.eps * torch.sign(norm)
            input_relevance = relevance_in.squeeze(dim=-1).repeat(1, 4)
            input_relevance[:, :2*out_c] *= (1+self.beta)/norm[:, :out_c].repeat(1, 2)
            input_relevance[:, 2*out_c:] *= -self.beta/norm[:, out_c:].repeat(1, 2)
            inv_w = w.t()
            relevance_out = torch.zeros_like(m.in_tensor)
            for i in range(4):
                relevance_out[:, i*in_c:(i+1)*in_c] = F.linear(
                    input_relevance[:, i*out_c:(i+1)*out_c],
                    weight=inv_w[:, i*out_c:(i+1)*out_c], bias=None)

            relevance_out *= m.in_tensor

            relevance_out = sum([relevance_out[:, i*in_c:(i+1)*in_c] for i in range(4)])

            del input_relevance, norm, rare_neurons, \
                mask, new_norm, m.in_tensor, w, inv_w

            return relevance_out

    @module_tracker_parallel
    def linear_fwd_hook_parallel(self, m, in_tensor: torch.Tensor,
                        out_tensor: torch.Tensor):

        setattr(m, "in_tensor", in_tensor[0])
        setattr(m, "out_shape", list(out_tensor.size()))
        return

    @module_tracker_parallel
    def batchNorm_fwd_hook_parallel(self, m, in_tensor: torch.Tensor, out_tensor: torch.Tensor):

        setattr(m, "in_tensor", in_tensor[0])
        setattr(m, "out_shape", list(out_tensor.size()))
        return

    @module_tracker
    def linear_fwd_hook(self, m, in_tensor: torch.Tensor,
                        out_tensor: torch.Tensor):

        setattr(m, "in_tensor", in_tensor[0])
        setattr(m, "out_shape", list(out_tensor.size()))
        return

    @module_tracker
    def batchNorm_fwd_hook(self, m, in_tensor: torch.Tensor, out_tensor: torch.Tensor):

        setattr(m, "in_tensor", in_tensor[0])
        setattr(m, "out_shape", list(out_tensor.size()))
        return

    def max_pool_nd_inverse(self, layer_instance, relevance_in):

        # In case the output had been reshaped for a linear layer,
        # make sure the relevance is put into the same shape as before.
        relevance_in = relevance_in.view(layer_instance.out_shape)

        invert_pool = self.get_inv_max_pool_method(layer_instance)
        inverted = invert_pool(relevance_in, layer_instance.indices,
                               layer_instance.kernel_size, layer_instance.stride,
                               layer_instance.padding, output_size=layer_instance.in_shape)
        del layer_instance.indices

        return inverted

    @module_tracker_parallel
    def max_pool_nd_fwd_hook_parallel(self, m, in_tensor: torch.Tensor,
                             out_tensor: torch.Tensor):
        # Ignore unused for pylint
        _ = self

        # Save the return indices value to make sure
        tmp_return_indices = bool(m.return_indices)
        m.return_indices = True
        _, indices = m.forward(in_tensor[0])
        m.return_indices = tmp_return_indices
        setattr(m, "indices", indices)
        setattr(m, 'out_shape', out_tensor.size())
        setattr(m, 'in_shape', in_tensor[0].size())

    @module_tracker
    def max_pool_nd_fwd_hook(self, m, in_tensor: torch.Tensor,
                             out_tensor: torch.Tensor):
        # Ignore unused for pylint
        _ = self

        # Save the return indices value to make sure
        tmp_return_indices = bool(m.return_indices)
        m.return_indices = True
        _, indices = m.forward(in_tensor[0])
        m.return_indices = tmp_return_indices
        setattr(m, "indices", indices)
        setattr(m, 'out_shape', out_tensor.size())
        setattr(m, 'in_shape', in_tensor[0].size())

    def conv_nd_inverse(self, m, relevance_in):

        # In case the output had been reshaped for a linear layer,
        # make sure the relevance is put into the same shape as before.
        relevance_in = relevance_in.view(m.out_shape)

        # Get required values from layer
        inv_conv_nd = self.get_inv_conv_method(m)
        conv_nd = self.get_conv_method(m)

        if self.method == "e-rule":
            with torch.no_grad():
                m.in_tensor = m.in_tensor.pow(self.p).detach()
                w = m.weight.pow(self.p).detach()
                norm = conv_nd(m.in_tensor, weight=w, bias=None,
                               stride=m.stride, padding=m.padding,
                               groups=m.groups)

                norm = norm + torch.sign(norm) * self.eps
                relevance_in[norm == 0] = 0
                norm[norm == 0] = 1
                relevance_out = inv_conv_nd(relevance_in/norm,
                                            weight=w, bias=None,
                                            padding=m.padding, stride=m.stride,
                                            groups=m.groups)
                relevance_out *= m.in_tensor
                del m.in_tensor, norm, w
                return relevance_out

        if self.method == "b-rule":
            with torch.no_grad():
                w = m.weight

                out_c, in_c = m.out_channels, m.in_channels
                repeats = np.array(np.ones_like(w.size()).flatten(), dtype=int)
                repeats[0] *= 4
                w = w.repeat(tuple(repeats))
                # First and third channel repetition only contain the positive weights
                w[:out_c][w[:out_c] < 0] = 0
                w[2 * out_c:3 * out_c][w[2 * out_c:3 * out_c] < 0] = 0
                # Second and fourth channel repetition with only the negative weights
                w[1 * out_c:2 * out_c][w[1 * out_c:2 * out_c] > 0] = 0
                w[-out_c:][w[-out_c:] > 0] = 0
                repeats = np.array(np.ones_like(m.in_tensor.size()).flatten(), dtype=int)
                repeats[1] *= 4
                # Repeat across channel dimension (pytorch always has channels first)
                m.in_tensor = m.in_tensor.repeat(tuple(repeats))
                m.in_tensor[:, :in_c][m.in_tensor[:, :in_c] < 0] = 0
                m.in_tensor[:, -in_c:][m.in_tensor[:, -in_c:] < 0] = 0
                m.in_tensor[:, 1 * in_c:3 * in_c][m.in_tensor[:, 1 * in_c:3 * in_c] > 0] = 0
                groups = 4

                # Normalize such that the sum of the individual importance values
                # of the input neurons divided by the norm
                # yields 1 for an output neuron j if divided by norm (v_ij in paper).
                # Norm layer just sums the importance values of the inputs
                # contributing to output j for each j. This will then serve as the normalization
                # such that the contributions of the neurons sum to 1 in order to
                # properly split up the relevance of j amongst its roots.
                norm = conv_nd(m.in_tensor, weight=w, bias=None, stride=m.stride,
                               padding=m.padding, dilation=m.dilation, groups=groups * m.groups)
                # Double number of output channels for positive and negative norm per
                # channel. Using list with out_tensor.size() allows for ND generalization
                new_shape = m.out_shape
                new_shape[1] *= 2
                new_norm = torch.zeros(new_shape).to(self.device)
                new_norm[:, :out_c] = norm[:, :out_c] + norm[:, out_c:2 * out_c]
                new_norm[:, out_c:] = norm[:, 2 * out_c:3 * out_c] + norm[:, 3 * out_c:]
                norm = new_norm
                # Some 'rare' neurons only receive either
                # only positive or only negative inputs.
                # Conservation of relevance does not hold, if we also
                # rescale those neurons by (1+beta) or -beta.
                # Therefore, catch those first and scale norm by
                # the according value, such that it cancels in the fraction.

                # First, however, avoid NaNs.
                mask = norm == 0
                # Set the norm to anything non-zero, e.g. 1.
                # The actual inputs are zero at this point anyways, that
                # is why norm is zero in the first place.
                norm[mask] = 1
                # The norm in the b-rule has shape (N, 2*out_c, *spatial_dims).
                # The first out_c block corresponds to the positive norms,
                # the second out_c block corresponds to the negative norms.
                # We find the rare neurons by choosing those nodes per channel
                # in which either the positive norm ([:, :out_c]) is zero, or
                # the negative norm ([:, :out_c]) is zero.
                rare_neurons = (mask[:, :out_c] + mask[:, out_c:])

                # Also, catch new possibilities for norm == zero to avoid NaN..
                # The actual value of norm again does not really matter, since
                # the pre-factor will be zero in this case.

                norm[:, :out_c][rare_neurons] *= 1 if self.beta == -1 else 1 + self.beta
                norm[:, out_c:][rare_neurons] *= 1 if self.beta == 0 else -self.beta
                # Add stabilizer term to norm to avoid numerical instabilities.
                norm += self.eps * torch.sign(norm)
                spatial_dims = [1] * len(relevance_in.size()[2:])

                input_relevance = relevance_in.repeat(1, 4, *spatial_dims)
                input_relevance[:, :2*out_c] *= (1+self.beta)/norm[:, :out_c].repeat(1, 2, *spatial_dims)
                input_relevance[:, 2*out_c:] *= -self.beta/norm[:, out_c:].repeat(1, 2, *spatial_dims)
                # Each of the positive / negative entries needs its own
                # convolution. TODO: Can this be done in groups, too?

                relevance_out = torch.zeros_like(m.in_tensor)
                # Weird code to make up for loss of size due to stride
                tmp_result = result = None
                for i in range(4):
                    tmp_result = inv_conv_nd(
                        input_relevance[:, i*out_c:(i+1)*out_c],
                        weight=w[i*out_c:(i+1)*out_c],
                        bias=None, padding=m.padding, stride=m.stride,
                        groups=m.groups)
                    result = torch.zeros_like(relevance_out[:, i*in_c:(i+1)*in_c])
                    tmp_size = tmp_result.size()
                    slice_list = [slice(0, l) for l in tmp_size]
                    result[slice_list] += tmp_result
                    relevance_out[:, i*in_c:(i+1)*in_c] = result
                relevance_out *= m.in_tensor

                sum_weights = torch.zeros([in_c, in_c * 4, *spatial_dims]).to(self.device)
                for i in range(m.in_channels):
                    sum_weights[i, i::in_c] = 1
                relevance_out = conv_nd(relevance_out, weight=sum_weights, bias=None)

                del sum_weights, m.in_tensor, result, mask, rare_neurons, norm, \
                    new_norm, input_relevance, tmp_result, w

                return relevance_out

    @module_tracker_parallel
    def conv_nd_fwd_hook_parallel(self, m, in_tensor: torch.Tensor,
                         out_tensor: torch.Tensor):

        setattr(m, "in_tensor", in_tensor[0])
        setattr(m, 'out_shape', list(out_tensor.size()))
        return


    @module_tracker
    def conv_nd_fwd_hook(self, m, in_tensor: torch.Tensor,
                         out_tensor: torch.Tensor):

        setattr(m, "in_tensor", in_tensor[0])
        setattr(m, 'out_shape', list(out_tensor.size()))
        return

    @module_tracker
    def batch_nd_inverse(self, m, relevance):
        batchNorm_easy_pass = False
        print(m.in_tensor.shape)
        if batchNorm_easy_pass:

            relevance_out = relevance
        else:
        # For the implementation we follow "Opening the Machine Learning Black Box
        # with Layer-wise Relevance Propagation", Chapter 2.3.1:
        # https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjDh-Lr17fwAhUCNuwKHSZzCgcQFjAAegQIBhAD&url=https%3A%2F%2Fdepositonce.tu-berlin.de%2Fbitstream%2F11303%2F8813%2F4%2Flapuschkin_sebastian.pdf&usg=AOvVaw2lteeO-jhXhjUQLA25O6aZ

            gamma = m.weight
            beta = m.bias
            mean = self.batchNorm_dict[str(self.batchNorm_id-1)]['mean']
            std = self.batchNorm_dict[str(self.batchNorm_id-1)]['var']
            layer_in = m.in_tensor

            mean = mean/(layer_in.shape[-1]*layer_in.shape[-2])

            print('batchID: ', self.batchNorm_id -1)
            print('layer_in:', layer_in.shape)
            print('mean_shape:', mean.shape)
            a = torch.squeeze(layer_in) - mean

            std = std.sum(1).sum(1)
            std = std/(layer_in.shape[-1]*layer_in.shape[-2])

            print('std: ', std.shape)
            b = torch.Tensor(std + self.eps)
            b = torch.sqrt(b)

            gamma = torch.unsqueeze(torch.tensor(gamma), 1)
            gamma = torch.unsqueeze(gamma, 1)

            c = gamma/b

            beta = torch.unsqueeze(beta, 1)
            beta = torch.unsqueeze(beta, 1)
            rel_out = a * gamma

            relevance_out = rel_out + beta
            print('Relevance calculated')
        return relevance_out
