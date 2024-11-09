import torch
import torch.distributed as dist
class column_parallel_linear_forward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, tp_group):
        ctx.save_for_backward(x, weight, bias)
        ctx.tp_group = tp_group
        x = torch.matmul(x, weight.T) + bias 
        return x
    
    def backward(ctx, grad_output):
        
        x, weight, bias = ctx.saved_tensors
        grad_input = torch.matmul(grad_output, weight) # [B, S, 2D] @ [2D, D] = [B, S, D]
        dist.all_reduce(grad_input, group=ctx.tp_group, op=dist.ReduceOp.SUM)
        grad_weight = torch.matmul(grad_output.transpose(-2, -1), x).sum(dim=0) # [B, 2D, S] @ [B, D, S] = [B, 2D, D]
        grad_bias = grad_output.sum(dim=(0,1)) # [B, S, D] -> [D]
        return grad_input, grad_weight, grad_bias , None
    

class row_parallel_linear_forward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, tp_group):
        ctx.save_for_backward(x, weight, bias)
        ctx.tp_group = tp_group
        x = torch.matmul(x, weight.T) + bias
        return x
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        grad_input = torch.matmul(grad_output, weight) # [B, S, D] @ [D, 2D] = [B, S, 2D]
        grad_weight = torch.matmul(grad_output.transpose(-2, -1), x).sum(dim=0) # [B, D, S] @ [B, S, D] = [B, D, D]
        grad_bias = grad_output.sum(dim=(0,1)) # [B, S, D] -> [D]
        return grad_input, grad_weight, grad_bias, None
    
class row_parallel_embedding_forward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tp_group):
        ctx.save_for_backward(x)
        ctx.tp_group = tp_group
        dist.all_reduce(x, group=ctx.tp_group, op=dist.ReduceOp.SUM)
        return x
    
    def backward(ctx, grad_output):
        x = ctx.saved_tensors
        return grad_output, None, None
    

class parallel_attention_forward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tp_group):
        ctx.save_for_backward(x)
        ctx.tp_group = tp_group
        dist.all_reduce(x, group=ctx.tp_group, op=dist.ReduceOp.SUM)
        return x
    
    def backward(ctx, grad_output):
        x = ctx.saved_tensors
        return grad_output, None