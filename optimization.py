
import torch
from functools import reduce
from operator import mul
import copy

# ZO, only pack up the common parameter. 
class ZO_output_optim():
    def __init__(self, mu=0.001, u_type="Uniform", d=10):
        self.mu=mu
        self.u_type = u_type
        self.d = d
        # This implementation only works for the clients model has same output dimension.
        # otherwise the Phi should implement in backward. 
        if self.u_type == "Uniform":
            self.phi = d
        elif self.u_type == "Normal":
            self.phi = 1
        elif self.u_type == "Coordinate":
            self.phi = 0.5 # block 
        else:
            raise Exception("no selected u_type.")

    # ZOFO
    def forward(self, output, l=0):
        # Normal or Uniform. 
        if self.u_type == "Normal":
            u = torch.randn_like(output)
        elif self.u_type == "Uniform":
            u = torch.randn_like(output)
            u = torch.nn.functional.normalize(u, dim=-1)
        elif self.u_type == "Coordinate":
            u = torch.zeros_like(output)
            u[:, l] = torch.ones_like(u[:, l]) # change the l th to 1. (unit vector. )
            perturbed_output_plus = output + self.mu * u
            perturbed_output_minus = output - self.mu * u
            return u, perturbed_output_plus, perturbed_output_minus, output
        else:
            raise Exception("no selected u_type.")
        
        # perturb the gradient. 
        perturbed_output = output + self.mu * u
        return u, perturbed_output, output

    def backward(self, u, delta):
        partial = self.phi/self.mu * delta.view(-1, 1) * u
        return partial

    # ZO
    def ZO_forward(self, data, model, memory_save=False):
        # not using params for now, save the interface for future. 
        flat_params = self.get_flat_params(model)

        # random sample a u
        if self.u_type == "Normal":
            u = torch.randn_like(flat_params)
        elif self.u_type == "Uniform":
            u = torch.randn_like(flat_params)
            u = torch.nn.functional.normalize(u, dim=-1)

        # ZO: calculate c and c hat. 
        output = model(data)
        if memory_save ==False:
            # get perturb output (c hat), with a perturbed copyed model. 
            perturbed_model = copy.deepcopy(model)
            perturbed_model_flat_params = flat_params + self.mu * u
            self.set_params(perturbed_model, perturbed_model_flat_params)
            perturbed_output = perturbed_model(data)
            del perturbed_model

        elif memory_save == True:
            perturbed_model_flat_params = flat_params + self.mu * u
            # modify model and output. 
            self.set_params(model, perturbed_model_flat_params)
            perturbed_output = model(data)
            # restore model
            self.set_params(model, flat_params)

        return u, output, perturbed_output

    # may need to changed if not using linear layer...
    def ZO_backward_step(self, perturbed_loss, loss, ZO_part_model, u, lr):
        flat_parmas = self.get_flat_params(ZO_part_model)
        # model update with the gradient estimator.
        if self.u_type == "Uniform":
            self.phi = flat_parmas.size(0)
        elif self.u_type == "Normal":
            self.phi = 1
        grads =  self.phi/ self.mu * (perturbed_loss - loss) * u
        
        flat_parmas = flat_parmas - lr * grads
        #print("test*** ", lr* grads)
        self.set_params(ZO_part_model, flat_parmas)
        return grads

    def get_flat_params(self, model):
        params = []
        for name, module in model.named_modules():
            if len(module._parameters) != 0:
                params.append(module._parameters['weight'].data.view(-1))
                try:
                    params.append(module._parameters['bias'].data.view(-1))
                except:
                    pass
        return torch.cat(params)

    def set_params(self, model, flat_params):
        # Restore original shapes
        offset = 0
        for module in model.modules():
            if len(module._parameters) != 0:
                weight_shape = module._parameters['weight'].size()
                weight_flat_size = reduce(mul, weight_shape, 1)
                module._parameters['weight'].data = flat_params[offset:offset + weight_flat_size].view(*weight_shape)
                try:
                    bias_shape = module._parameters['bias'].size()
                    bias_flat_size = reduce(mul, bias_shape, 1)
                    module._parameters['bias'].data = flat_params[
                                                        offset + weight_flat_size:offset + weight_flat_size + bias_flat_size].view(
                        *bias_shape)
                except:
                    bias_flat_size = 0
                offset += weight_flat_size + bias_flat_size
				
				
				
# FO
def get_optimizer(FO_part_model, lr):
    optimizer = torch.optim.SGD(FO_part_model.parameters(), lr=lr )
    return optimizer

def FO_backward_step(output, target, loss_fn, optimizer):
    optimizer.zero_grad()
    loss_fn(output, target).backward()
    optimizer.step()