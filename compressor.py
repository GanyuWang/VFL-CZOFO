import torch
import numpy as np

class Scale_Compressor():
    def __init__(self, bit):
        self.compression_type = "Scale"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.bit = bit
        if bit == 8:
            self.dtype = torch.uint8
        else:
            self.dtype = torch.uint8

    def compress(self, tensor):
        # input a tensor. 
        # high, low, bit. 
        low = tensor.min()
        high = tensor.max()
        boundaries = torch.tensor(np.linspace(low.cpu(), high.cpu(), 2**self.bit)).to(self.device)
        compressed_tensor = torch.bucketize(input = tensor, boundaries=boundaries)
        compressed = compressed_tensor.clone().detach().to(self.dtype)
        return compressed, low, high
    
    def decompress(self, compressed, low, high):
        assert self.bit <= 8
        decomp = low + compressed*(high - low)/(2**self.bit)
        return decomp

    def comm_cost_in_bit(self, compressed):
        n_element = torch.numel(compressed)
        base = n_element * self.bit
        extra = 32 * 2 # float. low and high. 
        return base + extra

class Top_K_Compressor():
    pass

def tensor_size_in_bit(tensor):
    n_element = torch.numel(tensor)
    dtype = tensor.dtype
    if dtype == torch.uint8:
        b = 8
    elif dtype == torch.float32:
        b = 32
    elif dtype == torch.float64:
        b = 64
    else:
        raise Exception(f"unsupported dtype {dtype}")
    return n_element * b
	
def compress_decompress(tensor, compression_type, compressor):
    if compression_type == "Scale":
        compressed, low, high  = compressor.compress( tensor )
        output = compressor.decompress(compressed, low, high)
    elif compression_type == "None":
        output = tensor
    else:
        raise Exception("No selected compression type.")
    return output
	
	
def compression_cost_in_bit(output, compression_type, compressor):
    if compression_type == "Scale":
        embedding_comm_cost_in_bit = compressor.comm_cost_in_bit(output)
    elif compression_type == "None":
        embedding_comm_cost_in_bit = tensor_size_in_bit(output)
    return embedding_comm_cost_in_bit