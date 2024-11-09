import torch
import torch.distributed as dist

def align_size(data, bucket_size):
    shape = data.shape
    data = data.view(-1)
    if data.numel() % bucket_size != 0:
        data = torch.cat([data, torch.zeros(bucket_size - data.numel() % bucket_size, dtype=data.dtype, device=data.device)])
    return data

class SimpleOptimizer:
    def __init__(self, model, 
                 lr=0.001, 
                 weight_decay=0.0,
                 gradient_accumulation_steps=1):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.total_steps = 0

    def step(self):

        self.total_steps += 1
        if self.total_steps % self.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

class ZeroOptimizer:
    def __init__(self, model, 
                 dp_group=None, 
                 lr=0.001, 
                 weight_decay=0.0,
                 gradient_accumulation_steps=1,
                 stage=0):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.stage = stage
        self.gradient_accumulation_steps = gradient_accumulation_steps
        if dp_group.size() == 1:
            self.dp_group = dist.group.WORLD
        else:
            self.dp_group = dp_group

        self.dp_size = self.dp_group.size()
        self.dp_rank = self.dp_group.rank()
        
        if self.stage == 0:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.stage == 1:
            self.pbuckets = []
            self.partition_parameters()
            self.optimizer = torch.optim.AdamW(self.pbuckets, lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"We only support stage 0 and 1, but got {self.stage}") 
        
        self.total_steps = 0

    def partition_parameters(self):
        
        ### TODO: Partition parameters equally among processes
        ### Hint: You should forloop through all parameters, and maintain a param shard of each parameter, 
        ### and parameterize the param shard so you can send them into optimizer
        for param in self.model.parameters():
            data = param.data.view(-1)
    
            data = align_size(data, self.dp_size)
            # only select the data of this rank, and copy to the param_shard
            data = data.view(self.dp_size, -1)
            param_shard = data[self.dp_rank].clone()

            self.pbuckets.append(torch.nn.Parameter(param_shard, requires_grad=True))
        ### TODO END

    def step(self):

        self.total_steps += 1

        if self.total_steps % self.gradient_accumulation_steps == 0:
            if self.stage == 0:
                
                ### TODO: all reduce the grad, and then step
                for param in self.model.parameters():
                    dist.all_reduce(param.grad, group=self.dp_group, op=dist.ReduceOp.SUM)
                    param.grad = param.grad / self.dp_group.size() / self.gradient_accumulation_steps
                self.optimizer.step()
                self.optimizer.zero_grad()
                ### TODO END
            elif self.stage == 1:
                ### TODO: Forloop through gradients of all parameters, and do reduce scatter, 
                ### and send the grad into self.pbuckets
                for (pbucket, param) in zip(self.pbuckets, self.model.parameters()):
                    
                    grad = torch.zeros_like(pbucket,requires_grad=False)

                    data = param.grad.view(-1).clone()
                    data = align_size(data, self.dp_size)

                    data = data / self.dp_size / self.gradient_accumulation_steps
                    dist.reduce_scatter_tensor(grad, data, group=self.dp_group)
                    
                    pbucket.grad = grad

                ### TODO END

                self.optimizer.step()
                
                ### TODO: Forloop through all param shards, and do all gather to get a global model param
                ### and update the param of the model

                for (pbucket, param) in zip(self.pbuckets, self.model.parameters()):
                    shape = param.shape
                    model_param = torch.zeros_like(param, dtype=param.dtype, device=param.device)
                    model_param = align_size(model_param, self.dp_size).view(-1)

                    dist.all_gather_into_tensor(model_param, pbucket.data.view(-1), group=self.dp_group)
                    param.data = model_param[:param.numel()].clone().view(shape)
                    
                ### TODO END

                self.optimizer.zero_grad()
                for param in self.model.parameters():
                    param.grad = None