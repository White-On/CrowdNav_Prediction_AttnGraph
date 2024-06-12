import torch
import torch.nn as nn
from torch.nn import functional as F
import logging


from rl.networks.GAT import GAT

# hyperparameters
# batch_size = 64 # how many independent sequences will we process in parallel?
# vehicle_node_size = 8
# max_iters = 5000
# eval_interval = 500
# learning_rate = 3e-4
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# eval_iters = 200
# n_embd = 64
# n_feature_graph = 64
# n_head = 4
# n_layer = 2
# n_action = 2
# dropout = 0.2



class CryoShell(nn.Module):
    def __init__(self, vehicle_node_size, n_embd, n_feature_graph, n_head, n_layer, n_action):
        super().__init__()
        logging.debug(f'CryoShell init with vehicle_node_size: {vehicle_node_size}, n_embd: {n_embd}, n_feature_graph: {n_feature_graph}, n_head: {n_head}, n_layer: {n_layer}, n_action: {n_action}')
        self.vehicle_embedding = nn.Linear(vehicle_node_size, n_embd)
        # self.AttentionGraphLayer = nn.Sequential(*[GraphAttentionLayer(n_embd, n_embd, n_head, concat=True, dropout=dropout) for _ in range(n_layer)])
        self.AttentionGraphLayer = GAT(n_layer, [n_head]*n_layer, (torch.zeros(n_layer+1)+n_feature_graph).int().tolist())
        self.GatedRecurrentUnit = nn.GRU(n_embd, n_embd, n_layer)
        self.action_head = nn.Linear(n_embd, n_action)

        self.apply(self._init_weights)

    def forward(self, graph, adj_mat, vehicle_node):
        # vehicle_node: [batch_size, vehicle_node_size]
        # graph: [batch_size, n_embd, n_embd]

        # vehicle_node: [batch_size, n_embd]
        # logging.debug(f'vehicle_node: {vehicle_node.shape}')
        # logging.debug(f'graph: {graph.shape}')
        # logging.debug(f'adj_mat: {adj_mat.shape}')

        vehicle_node = self.vehicle_embedding(vehicle_node)

        # graph: [batch_size, n_embd, n_embd]
        output, attn_mask, attn = self.AttentionGraphLayer(graph, adj_mat)

        # sum the vehicle node and the graph
        # [batch_size, n_embd]
        # logging.info(f'{vehicle_node.shape = }')
        # logging.info(f'{output.shape = }')
        # logging.info(f'{attn_mask.shape = }')
        # logging.info(f'{attn.shape = }')

        output_reduced = output.mean(dim=-1)
        x = vehicle_node + output_reduced
        x, hn = self.GatedRecurrentUnit(x)
        x = self.action_head(x)

        return x
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    
# model = CryoShell()
# m = model.to(device)
# # print the number of parameters in the model
# print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# # create a PyTorch optimizer
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# # ici
# adj_mat = torch.randint(0, 2, (batch_size, n_embd)).to(device)
# graph = torch.randn(batch_size, n_feature_graph).to(device)
# vehicle_node = torch.randint(0, vehicle_node_size, (batch_size,)).to(device)



# for iter in range(max_iters):

#     # every once in a while evaluate the loss on train and val sets
#     if iter % eval_interval == 0 or iter == max_iters - 1:
#         losses = estimate_loss()
#         print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

#     # sample a batch of data
#     xb, yb = get_batch('train')

#     # evaluate the loss
#     logits, loss = model(xb, yb)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()