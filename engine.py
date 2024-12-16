import numpy as np
import pandas as pd
import time
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from simulator.network_generator import InterfGraphDataset
from model.actor_critic import Actor, Critic
from utility.buffer import Buffer, get_buffer_dataloader
import wandb
import networkx as nx
import gc

import os
class Engine:
    def __init__(self, params_file, device):
        self._device = device
        # load configuration file
        self._config = {}
        conf_dir = Path(__file__).parents[0]
        with open(conf_dir / params_file, 'r') as f:
            self._config = yaml.safe_load(f)
        self._num_freq_ch = self._config['num_freq_ch']
        self._eval_iter = self._config['eval.iter']
        # calculate the boundaries for quantizing power attenuation
        start = self._config['power_attn.min']
        end = self._config['power_attn.max']
        step = self._power_attn_n_cls = self._config['power_attn.num_level']
        self._power_attn_boundaries = torch.linspace(start, end, step).to(self._device)
        # set up models
        model_params = {k[6:]: self._config[k] for k in self._config.keys() if k.startswith('model.')}
        self.neuron_layer = model_params["d_model"]
        self._actor = Actor(num_freq_ch=self._num_freq_ch, power_attn_num_level=self._power_attn_n_cls,
                            model_params=model_params, device=device)
        self._critic = Critic(num_freq_ch=self._num_freq_ch, power_attn_num_level=self._power_attn_n_cls,
                              model_params=model_params, device=device)
        # dataset
        train_dataset_name = self._config['train.train_dataset']
        self._train_graph_dataset = InterfGraphDataset(train_dataset_name)
        self._train_graph_batch_size = self._config['train.graph_batch_size']
        self._train_graph_dataloader = DataLoader(self._train_graph_dataset, batch_size=self._train_graph_batch_size,
                                                  shuffle=True)
        eval_dataset_name = self._config['train.eval_dataset']        
        self._eval_graph_dataset = InterfGraphDataset(eval_dataset_name)
        self._eval_batch_size = self._config['train.eval_batch_size']
        self._eval_graph_dataloader = DataLoader(self._eval_graph_dataset, batch_size=self._eval_batch_size,
                                                 shuffle=True)
        
        # train parameters
        self._num_graph_repeat = self._config['train.num_graph_repeat']
        self._cir_thresh = self._config['train.cir_thresh']
        self._gamma = self._config['train.gamma']
        self._lambda = self._config['train.lambda']
        self._buffer_batch_size = self._config['train.buffer_batch_size']
        self._actor_lr = self._config['train.actor_lr']
        self._critic_lr = self._config['train.critic_lr']
        self._clip_max_norm = self._config['train.clip_max_norm']
        self._entropy_loss_weight = self._config['train.entropy_loss_weight']
        self._num_train_iter = self._config['train.num_train_iter']
        self._PPO_clip = torch.Tensor([self._config['train.PPO_clip']]).to(torch.float).to(self._device)
        self._act_prob_ratio_exponent_clip = self._config['train.act_prob_ratio_exponent_clip']
        self._eval_period = self._config['train.eval_period']  
        
        self._num_graph_sample = self._config['train.graph_sample']      
        self._graph_color_map = self._config['eval.graph_color_map']
        self._use_graph = self._config['eval.use_graph']


    def quantize_power_attn(self, g_batch):
        g_list = g_batch.to_data_list()
        g2_list = []
        for g in g_list:
            # convert node power attenuation to one hot form
            node_power_attn = g.get_tensor('x').to(self._device)
            node_power_attn = torch.bucketize(node_power_attn, self._power_attn_boundaries, right=True) - 1
            node_power_attn[node_power_attn == -1] = 0
            node_power_attn = F.one_hot(node_power_attn, num_classes=self._power_attn_n_cls).to(torch.float32)
            # convert edge power attenuation to one hot form
            edge_power_attn = g.get_tensor('edge_attr').to(self._device)
            edge_power_attn = torch.bucketize(edge_power_attn, self._power_attn_boundaries, right=True) - 1
            valid_edge_idx = edge_power_attn >= 0
            edge_power_attn = edge_power_attn[valid_edge_idx]
            edge_power_attn = F.one_hot(edge_power_attn, num_classes=self._power_attn_n_cls).to(torch.float32)
            edge_index = g.edge_index.to(self._device)
            edge_index = edge_index[:, valid_edge_idx]
            
            # Add IAB properties
            net_map = g.net_map.to(self._device)
            node_channel_req = g.node_channel_req.to(self._device)
            # make a new graph
            g2 = Data(x=node_power_attn, edge_index=edge_index, edge_attr=edge_power_attn, net_map=net_map,
                      node_channel_req = node_channel_req)
            g2_list.append(g2)
        g2_batch = Batch.from_data_list(g2_list)
        return g2_batch

    def handle_iab_data(self, g_batch):
        g_list = g_batch.to_data_list()
        iab_batch_list = []
        r_iab_batch_list = []
        for g in g_list:
            # Take the network mapping information from the graph
            net_map = g.net_map.to(self._device)
            # Extract indices and new values from the mapping tensor
            indices = net_map[:, 0]
            new_values = net_map[:, 1]
            AP_size = torch.unique(new_values).size(0) 
            # Get all available variables
            node_attn = torch.ones(AP_size).unsqueeze(1).to(self._device) 
            # new_node_attn = torch.zeros(g.x.size(0)).to(self._device)
            # node_attn = torch.cat((new_node_attn,node_attn), dim = 0)
            new_values += torch.max(indices) + 1
            
            r_edge_index = torch.stack([indices, new_values], dim=0)
            edge_index = torch.cat((r_edge_index, torch.flip(r_edge_index,dims=[0])),dim = 1)
            
            # make a new graph
            iab = Data(x=node_attn, edge_index=edge_index)
            r_iab = Data(x=None, edge_index=r_edge_index)
            iab_batch_list.append(iab)
            r_iab_batch_list.append(r_iab)
        iab_batch = Batch.from_data_list(iab_batch_list)
        r_iab_batch = Batch.from_data_list(r_iab_batch_list)

        return iab_batch,r_iab_batch
    
    def random_freq_alloc(self, g, alloc_ratio=1.0):
        num_node = g.num_nodes
        freq_alloc = torch.randint(low=0, high=self._num_freq_ch, size=(num_node,), device=self._device)
        freq_alloc_one_hot = F.one_hot(freq_alloc, num_classes=self._num_freq_ch).to(torch.float32)
        mask = torch.rand((num_node,), device=self._device) <= alloc_ratio
        freq_alloc[~mask] = -1
        freq_alloc_one_hot[~mask, :] = 0
        return freq_alloc, freq_alloc_one_hot

    def cal_cir(self, g, freq_alloc):
        # get tx and rx power
        num_batch_node = freq_alloc.shape[0]
        freq_alloc.to(self._device)
        node_power_attn = g.x[:, None].to(self._device)
        
        node_channel_req = g.node_channel_req.to(self._device)

        tx_power = g.node_tx_power[:, None].to(self._device)
        tx_power = tx_power + 10 * torch.log10(freq_alloc)
        rx_power = tx_power + node_power_attn
        # get interference
        edge_power_attn = g.edge_attr[:, None].to(self._device)
        num_edge = edge_power_attn.shape[0]
        edge_index = g.edge_index.to(self._device)
        index_j, index_i = edge_index[0, :], edge_index[1, :]
        index_j = torch.broadcast_to(index_j[:, None], size=(num_edge, self._num_freq_ch))
        index_i = torch.broadcast_to(index_i[:, None], size=(num_edge, self._num_freq_ch))
        tx_power_j = torch.gather(input=tx_power, dim=0, index=index_j)
        interf_db = tx_power_j + edge_power_attn
        interf = torch.pow(10, interf_db * 0.1)
        sum_interf = torch.zeros(size=(num_batch_node, self._num_freq_ch), device=self._device)
        sum_interf = torch.scatter_add(input=sum_interf, dim=0, index=index_i, src=interf)
        sum_interf_db = 10 * torch.log10(sum_interf)
        # get cir
        node_freq_unalloc = (freq_alloc < 1)
        rx_power[node_freq_unalloc] = 0.0
        rx_power = torch.sum(rx_power, dim=1)
        sum_interf_db[node_freq_unalloc] = 0.0
        sum_interf_db = torch.sum(sum_interf_db, dim=1)
        cir = rx_power - sum_interf_db
        node_unalloc = torch.all(node_freq_unalloc, dim=1)
        cir[node_unalloc] = -torch.inf
        
        return cir

    def roll_out(self, g):
        g.to(self._device)
        batch, ptr = g.batch, g.ptr
        batch_size = int(ptr.shape[0]) - 1
        num_batch_node = batch.shape[0]
        g2 = self.quantize_power_attn(g)
        freq_alloc = torch.zeros(size=(num_batch_node, self._num_freq_ch)).to(torch.float).to(self._device)
        unallocated_node = torch.full(size=(num_batch_node,), fill_value=True).to(self._device)
        ongoing = torch.full(size=(batch_size,), fill_value=True).to(self._device)
        
        # Handle IAB data, change all the edge_power_attn and edge_index into Access Points
        r_freq_alloc = None
        rG = {}
        rG["x"] = None
        rG["edge_index"] = None
        r_iab_G = {}
        r_iab_G["edge_index"] = None
        r_ptr = None
        AP_ptr = None
        r_batch = None
        # Relation Graph
        if g2.net_map.numel() != 0:
            # Should be changed later
            rG, r_iab_G = self.handle_iab_data(g2)
            
            r_batch, r_ptr = rG.batch, rG.ptr
            r_freq_alloc = torch.zeros(size = (r_batch.shape[0], self._num_freq_ch)).to(torch.float).to(self._device)
            AP_ptr = rG.ptr + ptr
        self._actor.eval()
        self._critic.eval()
        freq_alloc_buf = []  # seq, (batch, node), freq
        r_freq_alloc_buf = []
        action_buf = []  # seq, batch, 2(node, freq)
        act_log_prob_buf = []  # seq, batch
        value_buf = []  # seq, batch
        ongoing_buf = []  # seq, batch
        cir_buf = []  # seq, (batch, node)

        with torch.no_grad():
            while torch.any(ongoing):
                # Actor Critic network
                act_dist = self._actor(freq_alloc=freq_alloc, node_power_attn=g2['x'],
                                    edge_power_attn=g2['edge_attr'], edge_index=g2['edge_index'], ptr=ptr, 
                                    r_freq_alloc = r_freq_alloc, r_embedding = rG['x'], r_edge_index = rG['edge_index'],
                                    r_ptr = r_ptr, AP_ptr = AP_ptr, net_map = r_iab_G['edge_index'], 
                                    channel_req = g2['node_channel_req'])
                value = self._critic(freq_alloc=freq_alloc, node_power_attn=g2['x'],
                                    edge_power_attn=g2['edge_attr'], edge_index=g2['edge_index'], batch=batch,
                                    r_freq_alloc = r_freq_alloc, r_embedding = rG['x'], r_edge_index = rG['edge_index'],
                                    r_batch = r_batch, ptr = ptr, r_ptr = r_ptr, AP_ptr = AP_ptr,net_map = r_iab_G['edge_index'])
                
                action = act_dist.sample()    
                act_log_prob = act_dist.log_prob(action)          
                      
                # record data
                freq_alloc_buf.append(freq_alloc.detach().clone().cpu())
                
                if r_freq_alloc is not None: r_freq_alloc_buf.append(r_freq_alloc.detach().clone().cpu()) 
                action_buf.append(action.detach().clone().cpu())
                act_log_prob_buf.append(act_log_prob.detach().clone().cpu())
                value_buf.append(value.detach().clone().cpu())
                ongoing_buf.append(ongoing.detach().clone().cpu())
                # update frequency allocation
                for idx, act in enumerate(action):
                    if ongoing[idx]:
                        node, freq = act[0], act[1]
                        # Check if the data is an IAB network or a Trunk network (True = Trunk, False = IAB)
                        if r_freq_alloc is not None:
                            net_map_batch = g.net_map[ptr[idx]:ptr[idx+1]].cpu()
                            
                            ptr = ptr.cpu()
                            
                            # Gather every occurence of the Access point in the node
                            r_ptr = r_ptr.cpu()
                            r_freq_alloc[r_ptr[idx] + node, freq] = 1

                            AP_index = torch.where(net_map_batch[:,1] == node.cpu())[0]
                            AP_index = net_map_batch[:,0][AP_index]
                            freq_alloc[AP_index+ ptr[idx],freq] = 1 
                            unallocated_node[ptr[idx] + AP_index] = False

                        else:
                            freq_alloc[ptr[idx] + node, freq] = 1.0
                            unallocated_node[ptr[idx] + node] = False
                            unallocated_node = (torch.sum(freq_alloc, dim=1, keepdim=True).squeeze(1)
                                                < g['node_channel_req'])   
                        ongoing[idx] = torch.any(unallocated_node[ptr[idx]: ptr[idx+1]])

                # compute cir and record
                cir = self.cal_cir(g, freq_alloc)
                cir_buf.append(cir.detach().clone().cpu())
        # Update the last Frequency Allocation
        freq_alloc_buf.append(freq_alloc.detach().clone().cpu())
        
        freq_alloc_buf = torch.stack(freq_alloc_buf, dim=0)
        if r_freq_alloc is not None: r_freq_alloc_buf = torch.stack(r_freq_alloc_buf, dim=0)
        action_buf = torch.stack(action_buf, dim=0)
        act_log_prob_buf = torch.stack(act_log_prob_buf, dim=0)
        value_buf = torch.stack(value_buf, dim=0)
        ongoing_buf = torch.stack(ongoing_buf, dim=0)
        cir_buf = torch.stack(cir_buf, dim=0)
        buf = Buffer(graph = g2, freq_alloc=freq_alloc_buf, 
                     action=action_buf, act_log_prob=act_log_prob_buf, value=value_buf,
                     ongoing=ongoing_buf, cir=cir_buf, 
                     r_graph = rG, r_iab_graph = r_iab_G, r_freq_alloc = r_freq_alloc_buf, device=self._device)
        return buf

    def train(self, use_wandb=False, save_model=True):
        if use_wandb:
            wandb.init(project='spectrum_allocation_trunk', config=self._config)
            wandb.define_metric("train/step")
            wandb.define_metric("train/*", step_metric="train/step")
            wandb.define_metric("evaluate/*")
            if not save_model:
                wandb.watch((self._actor, self._critic), log="all")
        train_step = 0
        actor_param_dicts = [{"params": [p for n, p in self._actor.named_parameters() if p.requires_grad]}]
        actor_optimizer = torch.optim.Adam(actor_param_dicts, lr=self._actor_lr)
        critic_param_dicts = [{"params": [p for n, p in self._critic.named_parameters() if p.requires_grad]}]
        critic_optimizer = torch.optim.Adam(critic_param_dicts, lr=self._critic_lr)
        for repeat_idx in range(self._num_graph_repeat):
            for graph_idx, g in enumerate(self._train_graph_dataloader):
                buf = self.roll_out(g)

                buf.cal_reward(cir_thresh=self._cir_thresh)
                buf.cal_lambda_return(gamma=self._gamma, lamb=self._lambda)
                buffer_dataloader = get_buffer_dataloader(buf, batch_size=self._buffer_batch_size, shuffle=True)  
                self._actor.train()
                self._critic.train()
                for minibatch_idx, d in enumerate(buffer_dataloader):
                    g = d['graph']
                    freq_alloc = d['freq_alloc']
                    action = d['action']
                    init_act_log_prob = d['act_log_prob']
                    lambda_return = d['return']                    
                    advantage = lambda_return - d['value']
                    
                    # IAB parameters
                    r_freq_alloc = None
                    rG = {}
                    rG["x"] = None
                    rG["edge_index"] = None
                    rG["ptr"] = None
                    rG["batch"] = None
                    r_iab_G = {}
                    r_iab_G["edge_index"] = None
                    AP_ptr = None
                    if g.net_map.numel() != 0:
                        r_freq_alloc = d["r_freq_alloc"]
                        rG = d["r_graph"]
                        r_iab_G = d["r_iab_graph"]
                        AP_ptr = rG.ptr + g.ptr

                    # Train actor
                    for it in range(self._num_train_iter):
                        torch.cuda.empty_cache()

                        act_dist = self._actor(freq_alloc=freq_alloc, node_power_attn=g['x'],
                                    edge_power_attn=g['edge_attr'], edge_index=g['edge_index'], ptr=g["ptr"], 
                                    r_freq_alloc = r_freq_alloc, r_embedding = rG['x'], r_edge_index = rG['edge_index'],
                                    r_ptr = rG["ptr"], AP_ptr = AP_ptr, net_map = r_iab_G['edge_index'],
                                    channel_req = g['node_channel_req'])

                        # Calculate PPO actor loss
                        act_log_prob = act_dist.log_prob(action)
                        act_prob_ratio = torch.exp(torch.clamp(act_log_prob - init_act_log_prob,
                                                               max=self._act_prob_ratio_exponent_clip))
                        
                        actor_loss = torch.where(advantage >= 0,
                                                 torch.minimum(act_prob_ratio, 1 + self._PPO_clip),
                                                 torch.maximum(act_prob_ratio, 1 - self._PPO_clip))
                        actor_loss = -torch.mean(actor_loss * advantage)
                        entropy_loss = -torch.mean(act_dist.entropy())
                        actor_entropy_loss = actor_loss + self._entropy_loss_weight * entropy_loss
                        actor_optimizer.zero_grad()
                        actor_entropy_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self._actor.parameters(), self._clip_max_norm)
                        actor_optimizer.step()
                     # Train critic
                    value = self._critic(freq_alloc=freq_alloc, node_power_attn=g['x'],
                                    edge_power_attn=g['edge_attr'], edge_index=g['edge_index'], batch=g["batch"],
                                    r_freq_alloc = r_freq_alloc, r_embedding = rG['x'], r_edge_index = rG['edge_index'],
                                    r_batch = rG["batch"], ptr = g["ptr"], r_ptr = rG["ptr"], AP_ptr = AP_ptr,net_map = r_iab_G['edge_index'])
                    value_loss = nn.MSELoss()(value, lambda_return)
                    critic_optimizer.zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self._critic.parameters(), self._clip_max_norm)
                    critic_optimizer.step()
                    train_step += 1
                    # Logging
                    log = (f"repeat: {repeat_idx+1}/{self._num_graph_repeat}, "
                           f"graph: {graph_idx+1}/{len(self._train_graph_dataloader)}, "
                           f"minibatch: {minibatch_idx+1}/{len(buffer_dataloader)}, "
                           f"actor loss: {actor_loss}, value loss: {value_loss}, "
                           f"entropy loss: {entropy_loss}")
                    print(log)
                    if use_wandb:
                        wandb_log = {"train/step": train_step, "train/actor_loss": actor_loss,
                                     "train/value_loss": value_loss, "train/entropy_loss": entropy_loss}
                        wandb.log(wandb_log)
                # Save memory after each iteration
                torch.cuda.empty_cache()
                if graph_idx % self._eval_period == 0:
                    self.evaluate(use_wandb)
                    if save_model:
                        self.save_model(use_wandb)

    def evaluate(self, use_wandb=False):
        success = 0
        total = 0
        cir = []
        for g in self._eval_graph_dataloader:
            buf = self.roll_out(g)
            buf.cal_reward(cir_thresh=self._cir_thresh)
            c, succ, tot = buf.get_performance(cir_thresh=self._cir_thresh)
            success += succ
            total += tot
            cir.append(c)
            if self._use_graph:
                networkx_plot = self.plot_freq_alloc(g, buf, show_graph = self._use_graph)

        success_ratio = success / total
        print(f"success ratio: {success_ratio}")
        cir = torch.concatenate(cir, dim=0).numpy()
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set(title='ecdf', xlabel='cir', ylabel='prob')
        ax.ecdf(cir)
        plt.close()

        # plt.close()
        # plt.show()
        if use_wandb:
            log = {"evaluate/success_ratio": success_ratio}
            wandb.log(log)
            wandb.log({'cir': wandb.Image(fig)})
            plt.close()
            #plt.close("all") # If this doesn't work, use this one
            if self._use_graph:
                # Add NetworkX Plot into the Wandb
                for i in range(len(networkx_plot)):
                    wandb.log({f'Network{i}': wandb.Image(networkx_plot[i])})

    def plot_freq_alloc(self, g, buf, show_graph = False):
        graph_iter = self._num_graph_sample if self._num_graph_sample < len(g) else len(g)
        # Take the fully allocated frequency array
        freq_alloc = buf._final_freq_alloc

        # Generate n random non-repeating numbers between eval_batch_size using torch.randperm
        random_numbers = torch.randperm(len(g))[:graph_iter]
        graph_eval = g.to_data_list()
          
                
        idx = torch.tensor(buf._idx)
        
        networkx_plot = []
        # Loop over the chosen graph sample        
        for i in random_numbers:
            # Do some additional indexing for IAB data
            if g.net_map.numel() != 0:
                tmp_net_map = g.net_map.clone().cpu()
                ptr = g.ptr.clone().cpu()
                old_idx = 0
                for j in range(len(ptr)-1):
                    tmp_net_map[ptr[j]:ptr[j+1],0] += old_idx 
                    tmp_net_map[ptr[j]:ptr[j+1],1] = j
                    old_idx =  tmp_net_map[ptr[j+1]-1,0] + 1
                tmp_net_map = torch.flip(tmp_net_map, [1]) 
                idx = tmp_net_map
                
            batch_indices_for_batch = (idx[:, 0] == i).nonzero(as_tuple=True)[0]
            new_freq_alloc = freq_alloc[batch_indices_for_batch]
            tx = graph_eval[i].node_tx
            rx = graph_eval[i].node_rx
            
            # Generates all nodes available on Trunk / IAB
            nodes = torch.cat((tx,rx))
            nodes = torch.unique(nodes)
            nodes = torch.sort(nodes).values
            pos = graph_eval[i].node_pos[nodes].to('cpu')
            pos = {idx: pos for idx, pos in zip(nodes.tolist(), pos.tolist())}

            # Create NetworkX
            G = nx.DiGraph()
            G.add_nodes_from(nodes.tolist())    # Add the Nodes
            node_labels = {node: str(node) for node in G.nodes()}  # Create labels with node identifiers
            edges = []
            
            # Add the edges
            edges = [(x, y) for x, y in zip(tx.tolist(), rx.tolist())]
            G.add_edges_from(edges)    

            freq_alloc_index = torch.nonzero(new_freq_alloc)[:,1]
            edge_colors = [self._graph_color_map[x.item()] for x in freq_alloc_index]

            color_map = {edge: color for edge, color in zip(edges, edge_colors)}

            fig = plt.figure()
            
            # Draw the NetworkX on the fig
            nx.draw_networkx_nodes(G, pos = pos, node_color="lightblue", node_size=100)
            nx.draw_networkx_edges(G, pos = pos, connectionstyle="arc3,rad=0.1", arrows=True, edge_color=[color_map[edge] for edge in G.edges()])
            nx.draw_networkx_labels(G, pos = pos, labels=node_labels, font_size=6)  # Draw node labels

            # Append into the list
            networkx_plot.append(fig) 
        
        if show_graph:
            plt.show()
        else:
            plt.close()
        return networkx_plot
         
    def save_model(self, use_wandb=False):
        if use_wandb:
            path = Path(wandb.run.dir)
        else:
            path = Path(__file__).parents[0].resolve() / 'saved_model'
        torch.save(self._actor, path / 'actor.pt')
        torch.save(self._critic, path / 'critic.pt')

    def load_model(self):
        path = Path(__file__).parents[0].resolve() / 'saved_model'
        self._actor = torch.load(path / 'actor.pt')
        self._critic = torch.load(path / 'critic.pt')

    def run_hedge_IAB(self):
        #for g in self._train_graph_dataloader:
        
        result = []
        threshold_list = [-170+5*n for n in range(6)]
        for threshold in threshold_list:
            cir_result = []
            time_record = []
            for g in self._eval_graph_dataset:
                start_time = time.time()
                g.to(self._device)

                # pruning edge
                saved_edge_index = g.edge_index

                mask = g.edge_attr >= threshold
                edge_index = g.edge_index[:, mask]

                # make undirected graph
                reversed_edge_index = edge_index[[1,0], :]
                undirected_edge_index = torch.unique(torch.cat([edge_index, reversed_edge_index], dim=1), dim=1)
                
                net_map = g.net_map
                num_nodes = g.num_nodes

                edge_index = undirected_edge_index
                degree_edge_index = edge_index

                node_index = torch.arange(num_nodes).to(self._device)
                node_degree = degree(edge_index[0], num_nodes, dtype=torch.long)

                sorted_index = torch.argsort(node_degree, descending=True)
            
                freq_alloc = torch.zeros(num_nodes, dtype=torch.long).to(self._device)
                freq_count = torch.zeros(self._num_freq_ch+2, dtype=torch.long).to(self._device)

                HEDGE = torch.stack([node_index, node_degree, freq_alloc], dim=1).T.to(self._device) 
                # 0 node_index_sort, 1 node_degree_sort, 2 is_empty, 3 freq_alloc
                
                for _ in range(torch.unique(net_map[:,1]).size(0)):
                    node_degree = degree(degree_edge_index[0], num_nodes, dtype=torch.long)
                    sorted_index = torch.argsort(node_degree, descending=True)

                    target_node_index = sorted_index[HEDGE[2][sorted_index]==0][0]

                    target_map_index = net_map[net_map[:,0] == target_node_index][0][1]
                    target_map = net_map[net_map[:,1] == target_map_index][:,0]

                    neighbor_nodes = torch.unique(edge_index[0][torch.isin(edge_index[1], target_node_index)])
                    neighbor_freq = torch.unique(HEDGE[2][torch.isin(node_index, neighbor_nodes)])

                    avaiable_freq = ~torch.isin(torch.arange(self._num_freq_ch+2).to(self._device), neighbor_freq)
                    avaiable_freq[0] = False
                    avaiable_freq[-1] = False
                    avaiable_freq_count = freq_count * avaiable_freq
                    avaiable_freq_count[~avaiable_freq] = -1e6
                    freq = torch.argmax(avaiable_freq_count)
                    if freq == 0:
                        # if can't assign freq(neighbor node use all freq), just disconnect edge
                        
                        tmp_result = []
                        for f in range(self._num_freq_ch):
                            tmp_freq_alloc = HEDGE[2]
                            f += 1
                            tmp_freq_alloc[target_map] = f
                            tmp_freq_alloc = torch.nn.functional.one_hot(tmp_freq_alloc, num_classes=self._num_freq_ch+1)[:,1:]
                            tmp_cir = self.cal_cir(g, freq_alloc=tmp_freq_alloc)
                            tmp_num_success = sum(tmp_cir>=8).item()
                            tmp_result.append((tmp_num_success, f))
                        tmp_result.sort(reverse=True)
                        freq = tmp_result[0][1]


                    freq_count[freq] += 6
                    HEDGE[2][target_map] = freq

                    # remove node (disconnect edge)
                    mask = (~torch.isin(degree_edge_index[0], target_map)) & (~torch.isin(degree_edge_index[1], target_map))
                    degree_edge_index = degree_edge_index[:,mask]
                    
                HEDGE[2][HEDGE[2]==-1] = 0
                freq_alloc = HEDGE[2]
                freq_alloc = torch.nn.functional.one_hot(freq_alloc, num_classes=self._num_freq_ch+1)[:,1:]
                g.edge_index = saved_edge_index

                cir = self.cal_cir(g, freq_alloc=freq_alloc)
                cir_result.append(cir)
                end_time = time.time()
                time_record.append(end_time-start_time)
            # result save
            output = []
            for i in range(len(cir_result)):
                num_success = sum(cir_result[i]>=8).item()
                total_link = cir_result[i].size(0)
                output.append([num_success, total_link, num_success/total_link, time_record[i]])
            output = pd.DataFrame(output)
            result.append([output.mean(axis=0)[2], threshold, output.mean(axis=0)[3]]+freq_count.tolist())
        pd.DataFrame(result).to_csv('result_IAB.csv', index=False)

    def run_hedge_trunk(self):

        result = []
        threshold_list =  [-180+10*n for n in range(12)]
        start_time = time.time()
        for threshold in threshold_list:
            cir_result = []

            for g in self._train_graph_dataset:
                g.to(self._device)
                
                # pruning edge
                saved_edge_index = g.edge_index

                mask = g.edge_attr >= threshold # threshold
                edge_index = g.edge_index[:, mask]

                # make undirected graph
                reversed_edge_index = edge_index[[1,0], :]
                undirected_edge_index = torch.unique(torch.cat([edge_index, reversed_edge_index], dim=1), dim=1)
                
                num_nodes = g.num_nodes

                edge_index = undirected_edge_index
                degree_edge_index = edge_index

                node_index = torch.arange(num_nodes).to(self._device)
                node_degree = degree(degree_edge_index[0], num_nodes, dtype=torch.long)

                sorted_index = torch.argsort(node_degree, descending=True)
                
                freq_alloc = torch.zeros(num_nodes, dtype=torch.long).to(self._device)
                freq_count = torch.zeros(self._num_freq_ch+2, dtype=torch.long).to(self._device)

                HEDGE = torch.stack([node_index, node_degree, freq_alloc], dim=1).T.to(self._device)
                # 0 node_index, 1 node_degree, 2 freq_alloc

                for _ in range(num_nodes):
                    node_degree = degree(degree_edge_index[0], num_nodes, dtype=torch.long)
                    sorted_index = torch.argsort(node_degree, descending=True)
                    
                    target_node_index = sorted_index[HEDGE[2][sorted_index]==0][0]

                    neighbor_nodes = torch.unique(edge_index[0][torch.isin(edge_index[1], target_node_index)])
                    neighbor_freq = torch.unique(HEDGE[2][torch.isin(node_index, neighbor_nodes)])

                    avaiable_freq = ~torch.isin(torch.arange(self._num_freq_ch+2).to(self._device), neighbor_freq)
                    avaiable_freq[0] = False
                    avaiable_freq[-1] = False
                    avaiable_freq_count = freq_count * avaiable_freq
                    avaiable_freq_count[~avaiable_freq] = -1e6
                    freq = torch.argmax(avaiable_freq_count)
                    if freq == 0:
                        # if can't assign freq(neighbor node use all freq), just disconnect edge
                        freq = -1

                    freq_count[freq] += 1
                    HEDGE[2][target_node_index] = freq

                    # remove node (disconnect edge)
                    mask = (degree_edge_index[0] != target_node_index) & (degree_edge_index[1] != target_node_index)
                    degree_edge_index = degree_edge_index[:,mask]

                # cal cir
                HEDGE[2][HEDGE[2]==-1] = 0
                freq_alloc = HEDGE[2]
                #not_alloc_index = freq_alloc==-1
                freq_alloc = torch.nn.functional.one_hot(freq_alloc, num_classes=self._num_freq_ch+1)[:,1:]
                g.edge_index = saved_edge_index

                cir = self.cal_cir(g, freq_alloc=freq_alloc)
                cir_result.append(cir)

            # result save
            output = []
            for cir in cir_result:
                num_success = sum(cir>=22).item()
                total_link = cir.size(0)
                output.append([num_success, total_link, num_success/total_link])
            output = pd.DataFrame(output)
            result.append([output.mean(axis=0)[2], threshold]+freq_count.tolist())
        pd.DataFrame(result).to_csv('result_TRUNK.csv', index=False)    
        end_time = time.time()
        print(end_time-start_time)

    def plot_freq_alloc_hedge(self, g, freq_alloc):

        tx = g.node_tx
        rx = g.node_rx
        
        # Generates all nodes available on Trunk / IAB
        nodes = torch.cat((tx,rx))
        nodes = torch.unique(nodes)
        nodes = torch.sort(nodes).values
        pos = g.node_pos[nodes].to('cpu')
        pos = {idx: pos for idx, pos in zip(nodes.tolist(), pos.tolist())}

        # Create NetworkX
        G = nx.DiGraph()
        G.add_nodes_from(nodes.tolist())    # Add the Nodes
        node_labels = {node: str(node) for node in G.nodes()}  # Create labels with node identifiers
        edges = []
        
        # Add the edges
        edges = [(x, y) for x, y in zip(tx.tolist(), rx.tolist())]
        G.add_edges_from(edges)    

        
        edge_colors = []
        # Set the colors for each edges
        for x in freq_alloc:
            if torch.any(x):
                edge_color = self._graph_color_map[torch.argmax(x).item()]
            else:
                edge_color = 'black'
            edge_colors.append(edge_color)

        color_map = {edge: color for edge, color in zip(edges, edge_colors)}

        fig = plt.figure()
        
        # Draw the NetworkX on the fig
        nx.draw_networkx_nodes(G, pos = pos, node_color="lightblue", node_size=100)
        nx.draw_networkx_edges(G, pos = pos, connectionstyle="arc3,rad=0.1", arrows=True, edge_color=[color_map[edge] for edge in G.edges()])
        nx.draw_networkx_labels(G, pos = pos, labels=node_labels, font_size=6)  # Draw node labels
        plt.show()
        print(1)

    def evaluate_results(self, use_wandb=False):
        success = 0
        total = 0
        cir = []
        suctot_ratio = []
        idx = 0
        acc = []
        min = []
        max = []
        for g in self._eval_graph_dataloader:
            start = time.time()
            buf = self.roll_out(g)
            buf.cal_reward(cir_thresh=self._cir_thresh)
            c, succ, tot = buf.get_performance(cir_thresh=self._cir_thresh)
            success += succ
            total += tot
            suctot_ratio.append([succ,tot])
            acc = succ/tot

            if (idx == 0):
                min = [succ, tot, acc, idx]
                max = [succ, tot, acc, idx]
            if acc > max[2] :
                max = [succ, tot, acc, idx]
            if acc < min[2]:
                min =   [succ, tot, acc, idx]
            cir.append(c)
            idx += 1
            if self._use_graph:
                networkx_plot = self.plot_freq_alloc(g, buf, show_graph = self._use_graph)
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set(title='ecdf', xlabel='cir', ylabel='prob')
            ax.ecdf(c)
            end = time.time()
            elapsed_time = end - start

            print(f"{succ},{tot},{elapsed_time}")

            

        success_ratio = success / total
        print(f"success ratio: {success_ratio}")
        cir = torch.concatenate(cir, dim=0).numpy()

    def run_greedy_IAB(self):
        
        result = []
        threshold_list = [-200+10*n for n in range(14)]
        time_record = []
        for threshold in threshold_list:
            cir_result = []
            start_time = time.time()
            for g in self._train_graph_dataset[:100]:
                
                g.to(self._device)

                # pruning edge
                saved_edge_index = g.edge_index

                mask = g.edge_attr >= threshold
                edge_index = g.edge_index[:, mask]

                # make undirected graph
                reversed_edge_index = edge_index[[1,0], :]
                undirected_edge_index = torch.unique(torch.cat([edge_index, reversed_edge_index], dim=1), dim=1)
                
                net_map = g.net_map
                num_nodes = g.num_nodes

                edge_index = undirected_edge_index
                degree_edge_index = edge_index

                node_index = torch.arange(num_nodes).to(self._device)
                node_degree = degree(edge_index[0], num_nodes, dtype=torch.long)

                sorted_index = torch.argsort(node_degree, descending=True)
            
                freq_alloc = torch.zeros(num_nodes, dtype=torch.long).to(self._device)
                freq_count = torch.zeros(self._num_freq_ch+2, dtype=torch.long).to(self._device)

                HEDGE = torch.stack([node_index, node_degree, freq_alloc], dim=1).T.to(self._device) 
                # 0 node_index_sort, 1 node_degree_sort, 2 is_empty, 3 freq_alloc
                
                for _ in range(torch.unique(net_map[:,1]).size(0)):
                    node_degree = degree(degree_edge_index[0], num_nodes, dtype=torch.long)
                    sorted_index = torch.argsort(node_degree, descending=True)

                    target_node_index = sorted_index[HEDGE[2][sorted_index]==0][0]

                    target_map_index = net_map[net_map[:,0] == target_node_index][0][1]
                    target_map = net_map[net_map[:,1] == target_map_index][:,0]


                    tmp_result = []
                    for f in range(self._num_freq_ch):
                        tmp_freq_alloc = HEDGE[2]
                        f += 1
                        tmp_freq_alloc[target_map] = f
                        tmp_freq_alloc = torch.nn.functional.one_hot(tmp_freq_alloc, num_classes=self._num_freq_ch+1)[:,1:]
                        tmp_cir = self.cal_cir(g, freq_alloc=tmp_freq_alloc)
                        tmp_num_success = sum(tmp_cir>=8).item()
                        tmp_result.append((tmp_num_success, f))
                    tmp_result.sort(reverse=True)
                    freq = tmp_result[0][1]


                    freq_count[freq] += 6
                    HEDGE[2][target_map] = freq

                    # remove node (disconnect edge)
                    mask = (~torch.isin(degree_edge_index[0], target_map)) & (~torch.isin(degree_edge_index[1], target_map))
                    degree_edge_index = degree_edge_index[:,mask]
                    
                HEDGE[2][HEDGE[2]==-1] = 0
                freq_alloc = HEDGE[2]
                freq_alloc = torch.nn.functional.one_hot(freq_alloc, num_classes=self._num_freq_ch+1)[:,1:]
                g.edge_index = saved_edge_index

                cir = self.cal_cir(g, freq_alloc=freq_alloc)
                cir_result.append(cir)
            
            end_time = time.time()

            # result save
            output = []
            for cir in cir_result:
                num_success = sum(cir>=8).item()
                total_link = cir.size(0)
                output.append([num_success, total_link, num_success/total_link])
            output = pd.DataFrame(output)
            result.append([output.mean(axis=0)[2], threshold]+freq_count.tolist()+[end_time-start_time])
        pd.DataFrame(result).to_csv('result_greedy_IAB.csv', index=False)

    def iter_evaluate(self):
        performance = []
        time_spend = []

        
        for g in self._eval_graph_dataset:
            start_time = time.time()
            g_batch = Batch.from_data_list([g] * self._train_graph_batch_size)
            buf = self.roll_out(g_batch)
            ptr = buf._ptr
            cir, _, _ = buf.get_performance(cir_thresh=self._cir_thresh)
            success = (cir >= self._cir_thresh).to(self._device)
            ratio = []
            for s in range(buf._batch_size):
                suc = success[ptr[s]:ptr[s+1]]
                suc = torch.sum(suc.int())
                tot = ptr[s+1] - ptr[s]
                ratio.append(suc/tot)
            ratio = torch.stack(ratio).to(torch.float)

            max_rate = torch.max(ratio)
            end_time = time.time()
            performance.append([ratio[0].item(), max_rate.item()])
            time_spend.append(end_time-start_time)
        performance = np.array(performance)
        time_spend = np.array(time_spend)
        output = pd.DataFrame(np.hstack((performance, time_spend.reshape(-1,1))))
        output.to_csv('iter_result.csv',index=False)



if __name__ == '__main__':

    device = 'cuda:0'
    en = Engine(params_file='config.yaml', device=device)
    en.load_model()
    en.train(use_wandb=False, save_model=True)
    en.iter_evaluate()




