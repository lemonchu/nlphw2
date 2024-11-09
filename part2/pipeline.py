import torch.nn as nn
import torch
import torch.distributed as dist

class NoPipe:
    def __init__(self, model, criterion):
        self.model = model
        self.criterion = criterion
    
    def forward_backward(self, input_ids, labels):
        outputs = self.model(input_ids)
        loss = self.criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        loss.backward()
        return loss

class SimplePipe:
    def __init__(self, model, criterion , rank, pp_group, tp_size):
        super(SimplePipe, self).__init__()
        self.model = model
        self.criterion = criterion
        self.rank = rank
        self.pp_group = pp_group
        self.tp_size = tp_size
        self.next_rank = (self.rank + tp_size)
        self.prev_rank = (self.rank - tp_size)
        print(f'[rank {self.rank}] PP GROUP {self.pp_group.rank()}')
        self.is_first_pp = self.pp_group.rank() == 0
        self.is_last_pp = self.pp_group.rank() == self.pp_group.size() - 1

    # def forward(self, x):
    #     return self.model(x)
    
    def forward_imp(self, input_ids, labels):   
        if self.is_first_pp:
            inputs = input_ids
        else:
            inputs = torch.zeros(
                (input_ids.shape[0], input_ids.shape[1], 256), 
                device=input_ids.device, 
                dtype=torch.float32,
                requires_grad=True)
            dist.recv(inputs, src=self.prev_rank, group=self.pp_group)
            # print(f"[rank {self.rank}] received inputs from rank {self.prev_rank}")
        outputs = self.model(inputs)
        if self.is_last_pp:
            loss = self.criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        else:
            loss = None
            dist.send(outputs, dst=self.next_rank, group=self.pp_group)
            # print(f"[rank {self.rank}] sent outputs with shape {outputs.shape} to rank {self.next_rank}")
        return inputs, outputs , loss

    def backward_imp(self, inputs, outputs, loss):
        if self.is_last_pp:
            loss.backward()
        else:
            grads = torch.zeros((inputs.shape[0], inputs.shape[1], 256), device=inputs.device, dtype=torch.float32)
            dist.recv(grads, src=self.next_rank, group=self.pp_group)
            torch.autograd.backward(outputs, grads)

        if self.is_first_pp:
            return loss
        else:
            dist.send(inputs.grad, dst=self.prev_rank, group=self.pp_group)
            # print(f"[rank {self.rank}] sent grads to rank {self.prev_rank}")
            return loss


    def forward_backward(self, input_ids, labels):
        inputs, outputs, loss = self.forward_imp(input_ids, labels)
        loss = self.backward_imp(inputs, outputs, loss)
        return loss       

class GPipe:
    def __init__(self, model, criterion, rank, pp_group, tp_size):
        super(GPipe, self).__init__()
        self.model = model
        self.criterion = criterion
        self.rank = rank
        self.pp_group = pp_group
        self.tp_size = tp_size
        self.next_rank = (self.rank + tp_size)
        self.prev_rank = (self.rank - tp_size)
        self.is_first_pp = self.pp_group.rank() == 0
        self.is_last_pp = self.pp_group.rank() == self.pp_group.size() - 1

    def forward_imp(self, input_ids, labels):   
        if self.is_first_pp:
            inputs = input_ids
        else:
            inputs = torch.zeros(
                (input_ids.shape[0], input_ids.shape[1], 256), 
                device=input_ids.device, 
                dtype=torch.float32,
                requires_grad=True)
            dist.recv(inputs, src=self.prev_rank, group=self.pp_group)
            # print(f"[rank {self.rank}] received inputs from rank {self.prev_rank}")
        outputs = self.model(inputs)
        if self.is_last_pp:
            loss = self.criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        else:
            loss = None
            dist.send(outputs, dst=self.next_rank, group=self.pp_group)
            # print(f"[rank {self.rank}] sent outputs with shape {outputs.shape} to rank {self.next_rank}")
        return inputs, outputs , loss
    
    def backward_imp(self, inputs, outputs, loss):
        if self.is_last_pp:
            loss.backward()
        else:
            grads = torch.zeros((inputs.shape[0], inputs.shape[1], 256), device=inputs.device, dtype=torch.float32)
            dist.recv(grads, src=self.next_rank, group=self.pp_group)
            torch.autograd.backward(outputs, grads)

        if self.is_first_pp:
            return loss
        else:
            dist.send(inputs.grad, dst=self.prev_rank, group=self.pp_group)
            # print(f"[rank {self.rank}] sent grads to rank {self.prev_rank}")
            return loss

    def forward_backward(self, samples):
        
        buffers = []
        for sample in samples:
            inputs, outputs, loss = self.forward_imp(**sample)
            buffers.append((inputs,outputs,loss))
        
        losses = []
        for i in range(len(buffers)-1,-1,-1):
            inputs, outputs, loss = buffers[i]
            losses.append(self.backward_imp(inputs, outputs, loss))
        return losses
