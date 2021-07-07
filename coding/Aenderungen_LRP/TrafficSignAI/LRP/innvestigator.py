import torch
import numpy as np

#import TrafficSignAI
from .inverter_util import RelevancePropagator
#from TrafficSignAI.LRP.utils import pprint, Flatten
from coding.Aenderungen_LRP.TrafficSignAI.LRP.utils import pprint, Flatten
from .dummpy_model import New_parallel_chain_dummy
from coding.Aenderungen_LRP.TrafficSignAI.Models.Net import Cat

class InnvestigateModel(torch.nn.Module):
    """
    ATTENTION:
        Currently, innvestigating a network only works if all
        layers that have to be inverted are specified explicitly
        and registered as a module. If., for example,
        only the functional max_poolnd is used, the inversion will not work.
    """

    def __init__(self, the_model, lrp_exponent=1, beta=.5, epsilon=1e-6,
                 method="e-rule"):
        """
        Model wrapper for pytorch models to 'innvestigate' them
        with layer-wise relevance propagation (LRP) as introduced by Bach et. al
        (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140).
        Given a class level probability produced by the model under consideration,
        the LRP algorithm attributes this probability to the nodes in each layer.
        This allows for visualizing the relevance of input pixels on the resulting
        class probability.

        Args:
            the_model: Pytorch model, e.g. a pytorch.nn.Sequential consisting of
                        different layers. Not all layers are supported yet.
            lrp_exponent: Exponent for rescaling the importance values per node
                            in a layer when using the e-rule method.
            beta: Beta value allows for placing more (large beta) emphasis on
                    nodes that positively contribute to the activation of a given node
                    in the subsequent layer. Low beta value allows for placing more emphasis
                    on inhibitory neurons in a layer. Only relevant for method 'b-rule'.
            epsilon: Stabilizing term to avoid numerical instabilities if the norm (denominator
                    for distributing the relevance) is close to zero.
            method: Different rules for the LRP algorithm, b-rule allows for placing
                    more or less focus on positive / negative contributions, whereas
                    the e-rule treats them equally. For more information,
                    see the paper linked above.
        """
        super(InnvestigateModel, self).__init__()
        self.model = the_model
        #self.device = torch.device("cpu", 0)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.method = str(method)
        self.prediction = None
        self.r_values_per_layer = None
        self.only_max_score = None
        # Initialize the 'Relevance Propagator' with the chosen rule.
        # This will be used to back-propagate the relevance values
        # through the layers in the innvestigate method.
        self.inverter = RelevancePropagator(lrp_exponent=lrp_exponent,
                                            beta=beta, method=method, epsilon=epsilon,
                                            device=self.device)
        # Dictionary for saving mean and std per batch norm layer
        self.batch_norm_dict = {}
        # Parsing the individual model layers
        self.register_hooks(self.model)
        if method == "b-rule" and float(beta) in (-1., 0):
            which = "positive" if beta == -1 else "negative"
            which_opp = "negative" if beta == -1 else "positive"
            print("WARNING: With the chosen beta value, "
                  "only " + which + " contributions "
                  "will be taken into account.\nHence, "
                  "if in any layer only " + which_opp +
                  " contributions exist, the "
                  "overall relevance will not be conserved.\n")

    def cuda(self, device=None):
        self.device = torch.device("cuda", device)
        self.inverter.device = self.device
        return super(InnvestigateModel, self).cuda(device)

    def cpu(self):
        self.device = torch.device("cpu", 0)
        self.inverter.device = self.device
        return super(InnvestigateModel, self).cpu()

    def register_hooks(self, parent_module, parallel=False):
        """
        Recursively unrolls a model and registers the required
        hooks to save all the necessary values for LRP in the forward pass.

        Args:
            parent_module: Model to unroll and register hooks for.

        Returns:
            None

        """
        def handle_inceptionA(mod):
            # New_parallel_chain_dummy
            children = list(mod.children())
            for child in children[:]:
                if list(child.children()):
                    self.register_hooks(child, parallel=True)
                    continue
                child.register_forward_hook(
                    self.inverter.get_layer_fwd_hook(child, parallel=True))




        #import TrafficSignAI
        #special_layers = (TrafficSignAI.Models.Net.InceptionA)

        from coding.Aenderungen_LRP.TrafficSignAI.Models.Net import InceptionA
        special_layers = (InceptionA)
        special_layer_handler_dict = {
            "InceptionA" : handle_inceptionA
        }

        for mod in parent_module.children():
            # print("Mod Name:", mod, "\n")
            if isinstance(mod, special_layers):
                handler = special_layer_handler_dict[mod.__class__.__name__]
                handler(mod)
            else:
                if list(mod.children()):
                    self.register_hooks(mod, parallel=parallel)
                    continue
                mod.register_forward_hook(
                    self.inverter.get_layer_fwd_hook(mod, parallel=parallel))
            if isinstance(mod, torch.nn.ReLU):
                mod.register_backward_hook(
                    self.relu_hook_function
                )

    # def register_inception_hook(self, module, grad_in, grad_out):
    #     pass

    @staticmethod
    def relu_hook_function(module, grad_in, grad_out):
        """
        If there is a negative gradient, change it to zero.
        """
        return (torch.clamp(grad_in[0], min=0.0),)

    def __call__(self, in_tensor):
        """
        The innvestigate wrapper returns the same prediction as the
        original model, but wraps the model call method in the evaluate
        method to save the last prediction.

        Args:
            in_tensor: Model input to pass through the pytorch model.

        Returns:
            Model output.
        """
        return self.evaluate(in_tensor)

    def evaluate(self, in_tensor):
        """
        Evaluates the model on a new input. The registered forward hooks will
        save all the data that is necessary to compute the relevance per neuron per layer.

        Args:
            in_tensor: New input for which to predict an output.

        Returns:
            Model prediction
        """
        # Reset module list. In case the structure changes dynamically,
        # the module list  is tracked for every forward pass.
        self.inverter.reset_module_list()
        self.prediction = self.model(in_tensor)
        return self.prediction

    def get_batch_norm_inputs(self, in_tensor):

        if in_tensor is not None:
            self.evaluate(in_tensor)

        # Get model list
        rev_model = self.inverter.module_list[::-1]
        # Iterate through model list like in innvestigate()
        # wenn eine layer asl bacthnorm erkannt wird Speichere den Input für die layer oder gebe ihn zurück
        #print(model_list)
        batch_norm_inputs = []
        print('Getting batchNorm inputs:')
        for layer in rev_model:
            print(layer)
            if isinstance(layer, list) and isinstance(layer[-1][0], Cat):
                for i, parallel_path in enumerate(layer[:-1]):
                    for layer_p in parallel_path[::-1]:
                        if isinstance(layer_p, torch.nn.BatchNorm2d):
                            print('a: ', layer_p.in_tensor.shape)
                            batch_norm_inputs.append(layer_p.in_tensor)
            else:
                if isinstance(layer, torch.nn.BatchNorm2d):
                    print('b: ', layer.in_tensor.shape)
                    batch_norm_inputs.append(layer.in_tensor)

        return batch_norm_inputs

    def get_r_values_per_layer(self):
        if self.r_values_per_layer is None:
            pprint("No relevances have been calculated yet, returning None in"
                   " get_r_values_per_layer.")
        return self.r_values_per_layer

    def innvestigate(self, in_tensor=None, rel_for_class=None):
        """
        Method for 'innvestigating' the model with the LRP rule chosen at
        the initialization of the InnvestigateModel.
        Args:
            in_tensor: Input for which to evaluate the LRP algorithm.
                        If input is None, the last evaluation is used.
                        If no evaluation has been performed since initialization,
                        an error is raised.
            rel_for_class (int): Index of the class for which the relevance
                        distribution is to be analyzed. If None, the 'winning' class
                        is used for indexing.

        Returns:
            Model output and relevances of nodes in the input layer.
            In order to get relevance distributions in other layers, use
            the get_r_values_per_layer method.
        """
        if self.r_values_per_layer is not None:
            for elt in self.r_values_per_layer:
                del elt
            self.r_values_per_layer = None

        with torch.no_grad():
            # Check if innvestigation can be performed.
            if in_tensor is None and self.prediction is None:
                raise RuntimeError("Model needs to be evaluated at least "
                                   "once before an innvestigation can be "
                                   "performed. Please evaluate model first "
                                   "or call innvestigate with a new input to "
                                   "evaluate.")

            # Evaluate the model anew if a new input is supplied.
            if in_tensor is not None:
                self.evaluate(in_tensor)

            # If no class index is specified, analyze for class
            # with highest prediction.
            if rel_for_class is None:
                # Default behaviour is innvestigating the output
                # on an arg-max-basis, if no class is specified.
                org_shape = self.prediction[0].size()
                # Make sure shape is just a 1D vector per batch example.
                self.prediction = self.prediction[0].view(org_shape[0], -1)
                max_v, _ = torch.max(self.prediction, dim=1, keepdim=True)
                only_max_score = torch.zeros_like(self.prediction).to(self.device)
                only_max_score[max_v == self.prediction] = self.prediction[max_v == self.prediction]
                relevance_tensor = only_max_score.view(org_shape)
                self.prediction.view(org_shape)
            else:
                org_shape = self.prediction[0].size()
                self.prediction = self.prediction[0].view(org_shape[0], -1)
                only_max_score = torch.zeros_like(self.prediction).to(self.device)
                only_max_score[:, rel_for_class] += self.prediction[:, rel_for_class]
                relevance_tensor = only_max_score.view(org_shape)
                self.prediction.view(org_shape)

            # We have to iterate through the model backwards.
            # The module list is computed for every forward pass
            # by the model inverter.
            rev_model = self.inverter.module_list[::-1]
            relevance = relevance_tensor.detach()
            del relevance_tensor
            # List to save relevance distributions per layer
            r_values_per_layer = [relevance]

            # Reset batchNorm_iD
            # We have to reset the id before iterating through the model again
            self.inverter.batchNorm_id = 0

            for layer in rev_model:
                #print(layer)

                #if isinstance(layer, list) and isinstance(layer[-1][0], TrafficSignAI.Models.Net.Cat): #Handle parallel flow with concat at the end
                if isinstance(layer, list) and isinstance(layer[-1][0], Cat):
                    # In dieser Schleife ist die layer aus rev_model selbst wieder eine Liste
                    # UND
                    # Der erste Eintrag der letzten Liste ist vom Typ Cat,
                    # also nur eine Konkatenierung. Wird hier ein Inception-Modul gefunden?
                    # dAS InceptionA Modul des verinfachten Netzwerkes wird erkannt
                    cat_layer = layer[-1][0]
                    dims = cat_layer.dims
                    concat_dim = cat_layer.concat_dim
                    relevance_list = []
                    for i, parallel_path in enumerate(layer[:-1]):
                        from_element = 0 if i == 0 else dims[i-1]
                        to_element = sum(dims) if i == len(dims) else dims[i]
                        to_element += from_element
                        # TODO: Make this less ugly
                        if concat_dim == 0:
                            sub_relevance = relevance[from_element:to_element, :, :, :]
                        elif concat_dim == 1:
                            sub_relevance = relevance[:, from_element:to_element, :, :]
                        elif concat_dim == 2:
                            sub_relevance = relevance[:, :, from_element:to_element, :]
                        else:
                            sub_relevance = relevance[:, :, :, from_element:to_element]

                        for layer_p in parallel_path[::-1]:
                            # Compute layer specific backwards-propagation of relevance values
                            sub_relevance = self.inverter.compute_propagated_relevance(layer_p, sub_relevance)
                        relevance_list.append(sub_relevance)
                    relevance = relevance_list
                    continue

                if not isinstance(relevance, list):
                    relevance = [relevance] #This is a list now. If the list has more than two elements, this is the node where multiple paths came frome

                new_relevance = None
                # Compute layer specific backwards-propagation of relevance values
                for rel in relevance:
                    new_rel = self.inverter.compute_propagated_relevance(layer, rel)
                    if new_relevance is None:
                        new_relevance = new_rel
                    else:
                        new_relevance = new_relevance + new_rel
                relevance = new_relevance
                r_values_per_layer.append(relevance.cpu())

            self.r_values_per_layer = r_values_per_layer

            del relevance
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            return self.prediction, r_values_per_layer[-1]

    def forward(self, in_tensor):
        return self.model.forward(in_tensor)

    def extra_repr(self):
        r"""Set the extra representation of the module

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        return self.model.extra_repr()
