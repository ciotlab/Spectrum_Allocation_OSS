import networkx as nx
import numpy as np
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import csv
import dill
from tqdm import tqdm
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader


class Network:
    def __init__(self, G, type, net_link_dict=None):
        self._G = G  # Interference graph
        self._type = type  # trunk, IAB
        self._net_link_dict = {n: [n] for n in self._G.nodes} if net_link_dict is None else net_link_dict  # Dictionary mapping network (AP) to links

    def to_pyg(self):
        node_power_attn = torch.tensor([self._G.nodes[node]['power_attn'] for node in self._G.nodes()], dtype=torch.float)
        node_channel_req = torch.randint(1,2, node_power_attn.size())

        node_tx_power = torch.tensor([self._G.nodes[node]['tx_power'] for node in self._G.nodes()], dtype=torch.float)
        edge_index = torch.tensor(list(self._G.edges()), dtype=torch.long).t().contiguous()
        edge_power_attn = torch.tensor([self._G[u][v]['power_attn'] for u, v in self._G.edges()], dtype=torch.float)
        
        # Take the TX, RX, and the position of each node with the masks
        node_tx= torch.tensor([self._G.nodes[node]['tx'] for node in self._G.nodes()], dtype=torch.int32)
        node_rx= torch.tensor([self._G.nodes[node]['rx'] for node in self._G.nodes()], dtype=torch.int32)
        
        node_position = torch.tensor(ng.node_pos, dtype = torch.float)

        # Check if it is an IAB network or a Trunk network
        if self._type == "IAB":
            # Gather data from _node_link_dict
            # node_channel_req = 
            nodes = torch.tensor(list(self._net_link_dict.keys()))
            ul_values = torch.tensor([value['UL'] for value in self._net_link_dict.values() if 'UL' in value],dtype = torch.int32)
            dl_values = torch.tensor([value['DL'] for value in self._net_link_dict.values() if 'DL' in value], dtype= torch.int32)
            net_values = torch.cat((ul_values, dl_values), dim=1)
            net_values, _ = torch.sort(net_values, dim=1)
            # Sort unique values and get indices
            sorted_indices = torch.argsort(nodes)

            # Map each value in the tensor to its corresponding index
            nodes = torch.searchsorted(nodes[sorted_indices], nodes)
            # Gather the index of the connection according to its AP
            net_index = torch.arange(net_values.size(0)).unsqueeze(1).expand(-1, net_values.size(1))
            net_index = torch.gather(nodes, 0, net_index.flatten())   
            
            # Stack index and value on the 2nd dimension
            net_data = torch.stack((net_values.flatten(),net_index), dim = 1)          
            net_map = net_data
        else:
            net_map = torch.tensor([])
        data = Data(x=node_power_attn, node_tx_power=node_tx_power, edge_index=edge_index, edge_attr=edge_power_attn, node_tx=node_tx, 
                    node_rx=node_rx, node_pos=node_position, net_map=net_map, node_channel_req=node_channel_req)       
        return data

    def random_allocation(self, num_freq):
        for n in self._net_link_dict:
            freq = np.random.randint(low=0, high=num_freq)
            freq_alloc = np.full(shape=(num_freq,), fill_value=False)
            freq_alloc[freq] = True
            links = (self._net_link_dict[n]['UL'] + self._net_link_dict[n]['DL'] if self._type == 'IAB'
                     else self._net_link_dict[n])
            for l in links:
                self._G.nodes[l]['freq_alloc'] = freq_alloc

    def cal_cir(self):
        for lt in self._G.nodes:
            freq_alloc = self._G.nodes[lt]['freq_alloc']
            tx_power = self._G.nodes[lt]['tx_power']
            power_attn = self._G.nodes[lt]['power_attn']
            rx_power = (tx_power + power_attn) * np.ones(freq_alloc.shape)
            sum_interf = np.zeros(freq_alloc.shape)
            for ls in self._G.predecessors(lt):
                interf_freq_alloc = self._G.nodes[ls]['freq_alloc']
                interf_tx_power = self._G.nodes[ls]['tx_power']
                interf_power_attn = self._G.edges[ls, lt]['power_attn']
                interf_rx_power = interf_tx_power + interf_power_attn
                sum_interf[interf_freq_alloc] += np.power(10, interf_rx_power * 0.1)
            tmp = np.full(freq_alloc.shape, -np.inf)
            np.log10(sum_interf, out=tmp, where=sum_interf > 0.0)
            sum_interf = 10 * tmp
            cir = rx_power[freq_alloc] - sum_interf[freq_alloc]
            self._G.nodes[lt]['cir'] = cir

    def draw_cir_ecdf(self):
        cir = []
        for l in self._G.nodes:
            cir.extend(self._G.nodes[l]['cir'])
        cir = np.array(cir)
        plt.ecdf(cir)
        plt.show()


class NetworkGroup:
    def __init__(self, network_list, type):
        self._network_list = network_list
        self._type = type

    def __getitem__(self, idx):
        return self._network_list[idx]

    def add_network(self, network):
        self._network_list.append(network)

    def save(self, file_name):
        path = Path(__file__).parents[0] / 'network' / file_name
        with open(path, 'wb') as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, file_name):
        path = Path(__file__).parents[0] / 'network' / file_name
        with open(path, 'rb') as f:
            obj = dill.load(f)
        return obj


class InterfGraphDataset(InMemoryDataset):
    def __init__(self, file_name):
        super().__init__()
        self._file_name = Path(__file__).parents[0] / 'network' / file_name
        self.load(str(self._file_name))

class NetworkGenerator:
    def __init__(self, params_file='config.yaml', parabolic_gain_file='parabolic_gain.csv'):
        self._config = {}
        conf_dir = Path(__file__).parents[0]
        with open(conf_dir / params_file, 'r') as f:
            self._config = yaml.safe_load(f)
        with open(conf_dir / parabolic_gain_file, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            self._parabolic_gain = []
            for d in reader:
                self._parabolic_gain.append(np.array([eval(d[0]), eval(d[1])]))
            self._parabolic_gain = np.stack(self._parabolic_gain, axis=0)
        self._omni_gain = 4.91

        # Variable for Node Position
        self.node_pos = []
    def generate_dataset(self, num_network, file_name_prefix):
        nt_list = []
        na_list = []
        for _ in tqdm(range(num_network)):
            nt, na = self.generate_network()
            nt_list.append(nt.to_pyg())
            na_list.append(na.to_pyg())
        save_dir = Path(__file__).parents[0] / 'network'
        InterfGraphDataset.save(nt_list, str(save_dir / (file_name_prefix + '_trunk')))
        InterfGraphDataset.save(na_list, str(save_dir / (file_name_prefix + '_IAB')))

    def generate_network_group(self, num_network):
        nt_list = []
        na_list = []
        for _ in tqdm(range(num_network)):
            nt, na = self.generate_network()
            nt_list.append(nt)
            na_list.append(na)
        ntg = NetworkGroup(nt_list, type='trunk')
        nag = NetworkGroup(na_list, type='IAB')
        return ntg, nag

    def generate_network(self):
        tG = ng.generate_topology(show_graph=False)
        self.node_pos = self.get_node_pos(tG)    # Take all of the node position
        iGt, iGa, net_link_dict = ng.get_interference_graph(tG)
        nt = Network(iGt, type='trunk', net_link_dict=None)
        na = Network(iGa, type='IAB', net_link_dict=net_link_dict)
        return nt, na

    
    def generate_topology(self, show_graph=False):
        G = nx.Graph()
        self._deploy_base_node(G)
        self._deploy_combat_unit(G)
        if show_graph:
            self._show_network_graph(G)
        return G

    def _deploy_base_node(self, G):
        top_unit_level = self._config['top_unit_level']
        region = np.array(self._config['AOR'][top_unit_level])
        region = region * np.array([1, 0.8])
        base_dist_min = self._config['node_dist_min'][top_unit_level]
        base_dist_max = self._config['node_dist_max']
        base_node_pos = np.ndarray([0, 2])
        pos_candidate = np.random.rand(self._config['node_base_trial_cnt'], 2) * region
        for pos in pos_candidate:
            dist = np.linalg.norm(base_node_pos - pos, axis=1)
            if dist.shape[0] == 0 or (base_dist_min < np.min(dist) < base_dist_max):
                node_idx = base_node_pos.shape[0]
                G.add_node(node_idx, pos=pos, level=0, n_base_link=0, n_child_link=0)
                base_node_pos = np.concatenate((base_node_pos, pos[np.newaxis, :]), axis=0)
        G.graph['base_node_pos'] = base_node_pos
        for n1, d1 in G.nodes.data():
            for n2, d2 in G.nodes.data():
                dist = np.linalg.norm(d1['pos'] - d2['pos'])
                if (d1['n_base_link'] < self._config['max_base_link']
                        and d2['n_base_link'] < self._config['max_base_link']
                        and n1 < n2 and dist <= base_dist_max):
                    G.add_edge(n1, n2, type='trunk', dist=dist)
                    d1['n_base_link'] += 1
                    d2['n_base_link'] += 1

    def _deploy_combat_unit(self, G):
        top_unit_level = self._config['top_unit_level']
        outer_AOR = np.array(self._config['AOR'][top_unit_level])
        pending_unit_list = []
        # Deploy initial unit
        node_idx = G.number_of_nodes()
        level = top_unit_level
        max_comm_dist = self._config['comm_distance']['trunk']
        max_sub_unit = self._config['max_sub_unit'][level]
        comm_ability = self._config['comm_ability'][level]
        comm_node_idx = None
        pos_low = pos_high = np.array(self._config['AOR'][level]) * np.array([0.5, 0.1])
        self._deploy_single_combat_unit(G, node_idx, level, max_comm_dist, max_sub_unit, comm_ability, pos_low, pos_high,
                                        comm_node_idx)
        pending_unit_list.append(node_idx)
        # Deploy units
        while pending_unit_list:
            parent_idx = pending_unit_list[0]
            parent_level = G.nodes[parent_idx]['level']
            parent_max_sub_unit = G.nodes[parent_idx]['max_sub_unit']
            parent_n_sub_unit = G.nodes[parent_idx]['n_sub_unit']
            if parent_level >= self._config['bottom_unit_level'] or parent_n_sub_unit >= parent_max_sub_unit:
                pending_unit_list.pop(0)
                continue
            node_idx += 1
            level = parent_level + 1
            # Corps, Division, Brigade, Batallion (level 1 to 4) has upstream trunk link
            max_comm_dist = (self._config['comm_distance']['trunk'] * 0.5 if level in [1, 2, 3, 4]
                             else self._config['comm_distance']['IAB'])
            max_sub_unit = self._config['max_sub_unit'][level]
            comm_ability = self._config['comm_ability'][level]
            # Corps, Division, Brigade (level 1 to 3) are connected to base node
            comm_node_idx = None if level in [1, 2, 3] else parent_idx
            parent_AOR = np.array(self._config['AOR'][parent_level])
            parent_pos = G.nodes[parent_idx]['pos']
            if level in [1, 2, 3]:  # Corps, Division, Brigade
                ratio = self._config['deploy_region']['level_1_3'][parent_n_sub_unit]
            elif level in [4, 5, 6]:  # Battalion, Company, Platoon
                ratio = self._config['deploy_region']['level_4_6'][parent_n_sub_unit]
            else:  # Squad
                ratio = self._config['deploy_region']['level_7']
            pos_low = parent_pos + parent_AOR * np.array(ratio['low'])
            pos_high = parent_pos + parent_AOR * np.array(ratio['high'])
            pos_low = np.clip(a=pos_low, a_min=np.array([0, 0]), a_max=outer_AOR)
            pos_high = np.clip(a=pos_high, a_min=np.array([0, 0]), a_max=outer_AOR)
            self._deploy_single_combat_unit(G, node_idx, level, max_comm_dist, max_sub_unit, comm_ability, pos_low,
                                            pos_high, comm_node_idx)
            G.nodes[parent_idx]['n_sub_unit'] += 1
            pending_unit_list.append(node_idx)

    def _deploy_single_combat_unit(self, G, node_idx, level, max_comm_dist, max_sub_unit, comm_ability,
                                   pos_low, pos_high, comm_node_idx):
        pos = None
        if comm_node_idx is None:  # Base unit
            while pos is None:
                p = np.random.uniform(low=pos_low, high=pos_high)
                base_dist = np.linalg.norm(G.graph['base_node_pos'] - p, axis=1)
                dist = np.min(base_dist)
                if dist <= max_comm_dist:
                    pos = p
                    comm_node_idx = np.argmin(base_dist)
                elif np.all(pos_low == pos_high):
                    raise Exception("Communication link cannot be established.")
        else:
            comm_node_pos = G.nodes[comm_node_idx]['pos']
            while pos is None:
                p = np.random.uniform(low=pos_low, high=pos_high)
                dist = np.linalg.norm(comm_node_pos - p)
                if dist <= max_comm_dist:
                    pos = p
                elif np.all(pos_low == pos_high):
                    raise Exception("Communication link cannot be established.")
        G.add_node(node_idx, pos=pos, level=level, n_sub_unit=0, max_sub_unit=max_sub_unit,
                   n_child_link=0, comm_ability=comm_ability)
        G.nodes[comm_node_idx]['n_child_link'] += 1
        if level in [1, 2, 3, 4]:
            G.add_edge(comm_node_idx, node_idx, type='trunk')

        else:
            G.add_edge(comm_node_idx, node_idx, type='IAB', AP=comm_node_idx)


    def _show_network_graph(self, G):
        pos = nx.get_node_attributes(G, name='pos')
        nx.draw(G, with_labels=False, font_weight='normal', pos=pos, node_size=3)
        plt.axis('on')
        plt.show()
        
    
    

    @staticmethod
    def hata_model(freq, hb, hr, d):
        # COST hata path loss model (rural area)
        # freq: frequency (MHz), hb: height of bs antenna (m), hr: height of mobile antenna (m), d: distance (km)
        d[d <= 0] = 0.000000001
        a = (1.1 * np.log10(freq) - 0.7) * hr - (1.56 * np.log10(freq) - 0.8)  # antenna height correction factor
        cm = 0  # constant offset
        lb = 46.3 + 33.9 * np.log10(freq) - 13.82 * np.log10(hb) - a + (44.9 - 6.55 * np.log10(hb)) * np.log10(d) + cm
        return lb

    @staticmethod
    def free_space_path_loss(freq, d):
        # freq: frequency array (MHz), d: distance (km)
        d[d <= 0] = 0.000000001
        c = 299792458.0  # m/s
        pl = 20 * (np.log10(4*np.pi*(d*1000)) + np.log10(freq*1000000) - np.log10(c))
        return pl

    def parabolic_gain(self, azimuth):
        # gain of parabolic antenna
        # azimuth (radian)
        a = ((azimuth * 180.0 / np.pi) + 180) % 360 - 180
        gain = np.interp(a, xp=self._parabolic_gain[:, 0], fp=self._parabolic_gain[:, 1])
        return gain

    def get_interference_graph(self, nG):  # nG: network graph
        iGt = nx.DiGraph()  # trunk interference graph
        iGa = nx.DiGraph()  # IAB interference graph

        # Node construction
        t_link_idx = 0
        a_link_idx = 0
        net_link_dict = {}
        tx_pos_trunk, rx_pos_trunk, tx_pos_iab, rx_pos_iab = [], [], [], []
        for u, v, a in nG.edges(data=True):
            pos_u, pos_v = nG.nodes[u]['pos'], nG.nodes[v]['pos']
            link_type = a['type']
            if link_type == 'trunk':
                iGt.add_node(t_link_idx, tx=u, rx=v, type=link_type, pair=t_link_idx + 1)
                iGt.add_node(t_link_idx + 1, tx=v, rx=u, type=link_type, pair=t_link_idx)
                t_link_idx += 2
                tx_pos_trunk += [pos_u, pos_v]
                rx_pos_trunk += [pos_v, pos_u]
            if link_type == 'IAB':
                AP = a['AP']
                if AP not in net_link_dict:
                    net_link_dict[AP] = {'UL': [], 'DL': []}
                iGa.add_node(a_link_idx, tx=u, rx=v, AP=AP, type=link_type, pair=a_link_idx + 1)
                direction = 'DL' if u == AP else 'UL'
                net_link_dict[AP][direction].append(a_link_idx)
                iGa.add_node(a_link_idx + 1, tx=v, rx=u, AP=AP, type=link_type, pair=a_link_idx)
                direction = 'DL' if v == AP else 'UL'
                net_link_dict[AP][direction].append(a_link_idx + 1)
                a_link_idx += 2
                tx_pos_iab += [pos_u, pos_v]
                rx_pos_iab += [pos_v, pos_u]
        tx_pos_trunk, rx_pos_trunk = np.stack(tx_pos_trunk, axis=0), np.stack(rx_pos_trunk, axis=0)
        tx_pos_iab, rx_pos_iab = np.stack(tx_pos_iab, axis=0), np.stack(rx_pos_iab, axis=0)

        # Compute interference matrix (Trunk)
        vec_trunk = rx_pos_trunk[:, np.newaxis, :] - tx_pos_trunk[np.newaxis, :, :]
        dist_trunk = np.linalg.norm(vec_trunk, axis=2)
        freq_trunk = float(self._config['freq']['trunk'])  # MHz
        pl_trunk = self.free_space_path_loss(freq_trunk, dist_trunk)
        angle = np.arctan2(vec_trunk[:, :, 1], vec_trunk[:, :, 0])
        tx_angle_diff = angle - np.diag(angle)[np.newaxis, :]
        rx_angle_diff = angle - np.diag(angle)[:, np.newaxis]
        tx_gain_trunk = self.parabolic_gain(tx_angle_diff)
        rx_gain_trunk = self.parabolic_gain(rx_angle_diff)
        power_attn_trunk = tx_gain_trunk + rx_gain_trunk - pl_trunk
        power_attn_trunk = np.clip(a=power_attn_trunk, a_min=None, a_max=0)

        # Compute interference matrix (IAB)
        vec_iab = rx_pos_iab[:, np.newaxis, :] - tx_pos_iab[np.newaxis, :, :]
        dist_iab = np.linalg.norm(vec_iab, axis=2)
        freq_iab = float(self._config['freq']['IAB'])  # MHz
        hb = self._config['ant_height']['hb']  # meter
        hr = self._config['ant_height']['hr']  # meter
        pl_iab = self.hata_model(freq_iab, hb, hr, dist_iab)
        tx_gain_iab = self._omni_gain
        rx_gain_iab = self._omni_gain
        power_attn_iab = tx_gain_iab + rx_gain_iab - pl_iab
        power_attn_iab = np.clip(a=power_attn_iab, a_min=None, a_max=0)

        # Link construction (trunk)
        tx_power_trunk = 10 * np.log10(self._config['tx_power']['trunk'])
        for ls in iGt.nodes:  # interference source
            for lt in iGt.nodes:  # interference target
                if ls == lt:
                    iGt.nodes[lt]['power_attn'] = power_attn_trunk[lt, ls]  # dB
                    iGt.nodes[lt]['tx_power'] = tx_power_trunk  # dB
                else:
                    iGt.add_edge(ls, lt, power_attn=power_attn_trunk[lt, ls])

        # Link construction (IAB)
        if not self._config['iab_sync']:
            tx_power_iab = 10 * np.log10(self._config['tx_power']['IAB'])
            for sn in net_link_dict:  # interference source network
                for tn in net_link_dict:  # interference target network
                    if sn == tn:
                        for l in net_link_dict[tn]['UL'] + net_link_dict[tn]['DL']:
                            iGa.nodes[l]['power_attn'] = power_attn_iab[l, l]
                            iGa.nodes[l]['tx_power'] = tx_power_iab  # dB
                    else:
                        # Select the link in source network causing the largest interference
                        attn_dict = {}
                        lt = net_link_dict[tn]['UL'][0]
                        for ls in net_link_dict[sn]['UL'] + net_link_dict[sn]['DL']:
                            attn_dict[ls] = power_attn_iab[lt, ls]
                        ls = max(attn_dict, key=attn_dict.get)  # uplink interference source
                        for lt in net_link_dict[tn]['UL'] + net_link_dict[tn]['DL']:
                            iGa.add_edge(ls, lt, power_attn=power_attn_iab[lt, ls])
        else:
            tx_power_iab = 10 * np.log10(self._config['tx_power']['IAB'])
            for sn in net_link_dict:  # interference source network
                for tn in net_link_dict:  # interference target network
                    if sn == tn:
                        for l in net_link_dict[tn]['UL'] + net_link_dict[tn]['DL']:
                            iGa.nodes[l]['power_attn'] = power_attn_iab[l, l]
                            iGa.nodes[l]['tx_power'] = tx_power_iab  # dB
                    else:
                        # Select the link in source network causing the largest interference
                        attn_dict = {}
                        lt = net_link_dict[tn]['UL'][0]
                        for ls in net_link_dict[sn]['UL']:
                            attn_dict[ls] = power_attn_iab[lt, ls]
                        uls = max(attn_dict, key=attn_dict.get)  # uplink interference source
                        dls = iGa.nodes[uls]['pair']  # downlink interference source
                        for lt in net_link_dict[tn]['UL'] + net_link_dict[tn]['DL']:
                            ls = uls if lt in net_link_dict[tn]['UL'] else dls
                            iGa.add_edge(ls, lt, power_attn=power_attn_iab[lt, ls])

        return iGt, iGa, net_link_dict

    def get_node_pos(self, G):
        node_pos = nx.get_node_attributes(G, 'pos')
        node_pos = np.array(list(node_pos.values()))
        return node_pos
if __name__ == '__main__':
    ng = NetworkGenerator()
    ng.generate_network()
    # Test network generation
    # nG = ng.generate_topology(show_graph=True)
    # iG = ng.get_interference_graph(nG)

    # # Test random allocation
    # ntg, nag = ng.generate_network_group(num_network=1)
    # ntg[0].random_allocation(5)
    # ntg[0].cal_cir()
    # nag[0].random_allocation(7)
    # nag[0].cal_cir()
    # ntg[0].draw_cir_ecdf()

    # # Test save/load network group
    # ntg.save('trunk.pkl')
    # nag.save('IAB.pkl')
    # ntg = NetworkGroup.load('trunk.pkl')
    # nag = NetworkGroup.load('IAB.pkl')

    # Test save dataset
    ng.generate_dataset(num_network=200, file_name_prefix='test_37_nonsync_rand')

    # # Test load dataset
    # dt = InterfGraphDataset('test_trunk')
    # da = InterfGraphDataset('test_IAB')
    # loader = DataLoader(dt, batch_size=32, shuffle=True)
    # for data in loader:
    #     pass
