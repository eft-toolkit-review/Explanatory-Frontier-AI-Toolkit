"""
paper: Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph
Reference From: https://github.com/pliang279/MFN & https://github.com/A2Zadeh/CMU-MultimodalSDK
"""

import copy
from torch import Tensor
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain, combinations
import logging

logger = logging.getLogger('MMSA')

__all__ = ['A2_Graph_MFN_A']

class DynamicFusionGraph(nn.Module):
    """Fuse modalities with learned edge weights; return fused features, node features, and weights."""

    def __init__(self, pattern_model, in_dimensions, out_dimension, efficacy_model, device, set_efficacies_as_one):
        """
        Args:
            pattern_model - nn.Module, a nn.Sequential model which will be used as core of the models inside the DFG
            in_dimensions - List, input dimensions of each modality
            out_dimension - int, output dimension of the pattern models
            efficacy_model - the core of the efficacy model
    	"""
        super(DynamicFusionGraph, self).__init__()
        self.num_modalities = len(in_dimensions)
        self.in_dimensions = in_dimensions
        self.out_dimension = out_dimension
        self.set_efficacies_as_one = set_efficacies_as_one

        # in this part we sort out number of connections, how they will be connected etc.
        # powerset = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)] for three modal fusion.
        self.powerset = list(
            chain.from_iterable(combinations(range(self.num_modalities), r) for r in range(self.num_modalities + 1)))[
                        1:]

        # initializing the models inside the DFG
        self.input_shapes = {tuple([key]): value for key, value in zip(range(self.num_modalities), in_dimensions)}

        self.networks = {}
        self.total_input_efficacies = 0  # total_input_efficacies: for alpha list size = [batch_size, total_input_efficacies].

        # loop over n-modal node (n >= 2)
        for key in self.powerset[self.num_modalities:]:
            # connections coming from the unimodal components
            unimodal_dims = 0
            for modality in key:
                unimodal_dims += in_dimensions[modality]
            multimodal_dims = ((2 ** len(key) - 2) - len(key)) * out_dimension
            self.total_input_efficacies += 2 ** len(key) - 2
            # for the network that outputs key component, what is the input dimension
            final_dims = unimodal_dims + multimodal_dims
            self.input_shapes[key] = final_dims
            pattern_copy = copy.deepcopy(pattern_model)

            # final_model: transform the input to the node into out_dimension dim.
            final_model = nn.Sequential(
                *[nn.Linear(self.input_shapes[key], list(pattern_copy.children())[0].in_features), pattern_copy]).to(
                device)
            self.networks[key] = final_model
        # finished construction weights, now onto the t_network which summarizes the graph
        self.total_input_efficacies += 2 ** self.num_modalities - 1

        self.t_in_dimension = unimodal_dims + (2 ** self.num_modalities - (self.num_modalities) - 1) * out_dimension
        pattern_copy = copy.deepcopy(pattern_model)
        # self.t_network: generate top level representation τ
        self.t_network = nn.Sequential(
            *[nn.Linear(self.t_in_dimension, list(pattern_copy.children())[0].in_features), pattern_copy]).to(device)
        # self.efficacy_model: generate the alpha list using the singleton vertex input.
        # (in 3 modal [batch_size, dim_l+dim_v+dim_a] -> [batch_size, 19])
        if not set_efficacies_as_one:
            self.efficacy_model = nn.Sequential(
                *[nn.Linear(sum(in_dimensions), list(efficacy_model.children())[0].in_features), efficacy_model,
                  nn.Linear(list(efficacy_model.children())[-1].out_features, self.total_input_efficacies)]).to(device)
        else:
            print("set efficies as 1", "\n no e")
            self.efficacy_model = OnesLike()

    def __call__(self, in_modalities):
        return self.fusion(in_modalities)

    def fusion(self, in_modalities):

        outputs = {}
        for modality, index in zip(in_modalities, range(len(in_modalities))):
            outputs[tuple([index])] = modality
        efficacies = self.efficacy_model(torch.cat([x for x in in_modalities], dim=1))

        efficacy_index = 0
        for key in self.powerset[self.num_modalities:]:
            small_power_set = list(chain.from_iterable(combinations(key, r) for r in range(len(key) + 1)))[1:-1]

            this_input = torch.cat([outputs[x].cuda() * efficacies[:, efficacy_index + y].view(-1, 1).cuda() for x, y in
                                    zip(small_power_set, range(len(small_power_set)))], dim=1)

            outputs[key] = self.networks[key](this_input)
            efficacy_index += len(small_power_set)

        small_power_set.append(tuple(range(self.num_modalities)))
        t_input = torch.cat([outputs[x].cuda() * efficacies[:, efficacy_index + y].view(-1, 1).cuda() for x, y in
                             zip(small_power_set, range(len(small_power_set)))], dim=1)
        t_output = self.t_network(t_input)

        return t_output, outputs, efficacies

    def forward(self, x):
        print("Not yet implemented for nn.Sequential")
        exit(-1)

class OnesLike(nn.Module):
    """Return ones of shape (batch_size, 19) for fixed trimodal fusion weights."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x) -> Tensor:
        b = x.shape[0]
        return torch.ones((b, 19))

class Concat(nn.Module):
    """Concatenate input tensors along the specified dimension."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: list) -> Tensor:
        return torch.concat(x, dim=self.dim)

class MultiOutputConcat(Concat):
    """Return concatenated tensors and placeholders matching the fusion graph output."""

    def forward(self, x: list) -> Tensor:
        return super().forward(x), 0, torch.ones(1)

class Select(nn.Module):
    """Select an input tensor by index."""

    def __init__(self, select_index: int) -> None:
        super().__init__()
        self.index = select_index

    def forward(self, x: list) -> Tensor:
        return x[self.index]

class MultiSelect(nn.Module):
    """Select input tensors in the specified index order."""

    def __init__(self, select_index: list) -> None:
        super().__init__()
        self.index = select_index

    def forward(self, x: list) -> Tensor:
        out = []
        for i in self.index:
            out.append(x[i])
        return out

# Assume abl is a string such as "la", "v", "lav"
class A2_Graph_MFN_A(nn.Module):
    def __init__(self, args):
        super(A2_Graph_MFN_A, self).__init__()
        self.abl: str = args.abl
        self.d_l, self.d_a, self.d_v = args.feature_dims
        # 256 32 256
        self.dh_l, self.dh_a, self.dh_v = args.hidden_dims
        # for ablations
        self.noW = args.noW
        self.noE = args.noE
        self.noG = args.noG
        self.noM = args.noM
        self.noS = args.noS
        if (self.noW + self.noE + self.noG + self.noM + self.noS > 1):
            warnings.warn("The ablations config will remove more than 1 component.", UserWarning, stacklevel=2)
        logger.info(
            f"self.noW:{self.noW} , self.noE:{self.noE} , self.noG:{self.noG} , self.noM:{self.noM} , self.noS:{self.noS}")

        self.num_modalities = len(self.abl)
        if self.abl == "l":
            total_h_dim = self.dh_l
        if self.abl == "a":
            total_h_dim = self.dh_a
        if self.abl == "v":
            total_h_dim = self.dh_v
        if self.abl == "la":
            total_h_dim = self.dh_l + self.dh_a
        if self.abl == "lv":
            total_h_dim = self.dh_l + self.dh_v
        if self.abl == "av":
            total_h_dim = self.dh_a + self.dh_v
        if self.abl == "lav":
            total_h_dim = self.dh_l + self.dh_a + self.dh_v

        self.mem_dim = args.memsize
        self.inner_node_dim = args.inner_node_dim
        self.singleton_l_size, self.singleton_a_size, self.singleton_v_size = args.hidden_dims

        output_dim = 1

        gammaInShape = total_h_dim + self.mem_dim if self.noG else self.inner_node_dim + self.mem_dim
        if self.noM:
            self.mem_dim = 0
        if self.num_modalities ==1:
            final_out = total_h_dim
        else:
            final_out = total_h_dim if self.noM else total_h_dim + self.mem_dim
        h_att2 = args.NNConfig["shapes"]
        h_gamma1 = args.gamma1Config["shapes"]
        h_gamma2 = args.gamma2Config["shapes"]
        h_out = args.outConfig["shapes"]
        att2_dropout = args.NNConfig["drop"]
        gamma1_dropout = args.gamma1Config["drop"]
        gamma2_dropout = args.gamma2Config["drop"]
        out_dropout = args.outConfig["drop"]

        if self.abl == "l":
            total_h_dim = self.dh_l
        if self.abl == "a":
            total_h_dim = self.dh_a
        if self.abl == "v":
            total_h_dim = self.dh_v
        if self.abl == "la":
            total_h_dim = self.dh_l + self.dh_a
        if self.abl == "lv":
            total_h_dim = self.dh_l + self.dh_v
        if self.abl == "av":
            total_h_dim = self.dh_a + self.dh_v
        if self.abl == "lav":
            total_h_dim = self.dh_l + self.dh_a + self.dh_v

        # Construct LSTMCells according to the ablation configuration

        self.lstm_l = nn.LSTMCell(self.d_l, self.dh_l)
        self.l_transform = self.build_lstm_transform(self.dh_l * 2, self.singleton_l_size)

        self.lstm_a = nn.LSTMCell(self.d_a, self.dh_a)
        self.a_transform = self.build_lstm_transform(self.dh_a * 2, self.singleton_a_size)

        self.lstm_v = nn.LSTMCell(self.d_v, self.dh_v)
        self.v_transform = self.build_lstm_transform(self.dh_v * 2, self.singleton_v_size)
        # Initialize the dynamic fusion graph.
        if self.noG:
            print("no g", "*" * 10)
            # GMFN no G completely removes the DFG and performs multimodal fusion by a simple concatenation of LSTM outputs
            self.graph_mfn = MultiOutputConcat(dim=1)
        else:

            pattern_model = nn.Sequential(nn.Linear(100, self.inner_node_dim)).to(args.device)
            efficacy_model = nn.Sequential(nn.Linear(100, self.inner_node_dim)).to(
                args.device)  # Note : actually here inner_node_dim can change arbitrarily
            self.graph_mfn = DynamicFusionGraph(pattern_model,
                                                [self.singleton_l_size, self.singleton_a_size, self.singleton_v_size],
                                                self.inner_node_dim, efficacy_model, args.device, self.noE).to(  args.device)


        if self.noG:
            self.att2_fc1 = nn.Linear(total_h_dim,
                                      h_att2)  # Note: might (inner_node_dim = self.mem_dim) is a common choice.
        else:
            self.att2_fc1 = nn.Linear(self.inner_node_dim, h_att2)
        self.att2_fc2 = nn.Linear(h_att2, self.mem_dim)
        self.att2_dropout = nn.Dropout(att2_dropout)
        if self.noM:
            print("no m", "*" * 10)
            self.MSM = self.forward_one
            self.mem = 1
            self.out_concant = nn.Sequential(
                MultiSelect([0, 1, 2]),
                Concat(dim=1)
            )
        if not self.noM:
            self.gamma1_fc1 = nn.Linear(gammaInShape, h_gamma1)
            self.gamma1_fc2 = nn.Linear(h_gamma1, self.mem_dim)
            self.gamma1_dropout = nn.Dropout(gamma1_dropout)

            self.gamma2_fc1 = nn.Linear(gammaInShape, h_gamma2)
            self.gamma2_fc2 = nn.Linear(h_gamma2, self.mem_dim)
            self.gamma2_dropout = nn.Dropout(gamma2_dropout)
            self.MSM = self.forward_msm
            self.out_concant = Concat(dim=1)

        self.out_fc1 = nn.Linear(final_out, h_out)
        self.out_fc2 = nn.Linear(h_out, output_dim)
        self.out_dropout = nn.Dropout(out_dropout)

    def build_lstm_transform(self, in_ch, out_ch):
        if self.noW:  # no W
            print("no w", "*" * 10, nn.Sequential(
                Select(1),
                nn.ReLU()
            ))
            return nn.Sequential(
                Select(1),
                nn.ReLU()
            )
        return nn.Sequential(
            Concat(dim=1),
            nn.Linear(in_ch, out_ch),
            nn.ReLU()
        )

    def forward_msm(self, attended):
        t1 = self.att2_fc1(attended)
        t2 = self.att2_dropout(F.relu(t1))
        t3 = self.att2_fc2(t2)
        cHat = torch.tanh(t3)
        both = torch.cat([attended, self.mem], dim=1)
        gamma1 = torch.sigmoid(self.gamma1_fc2(self.gamma1_dropout(F.relu(self.gamma1_fc1(both)))))
        gamma2 = torch.sigmoid(self.gamma2_fc2(self.gamma2_dropout(F.relu(self.gamma2_fc1(both)))))
        self.mem = gamma1 * self.mem + gamma2 * cHat
        return self.mem

    def forward_one(self, _):
        pass
    def forward(self, text_x, audio_x, video_x):
        '''
        Input arguments:
            text_x: Text input tensor with shape (batch_size, sequence_len, text_in)
            audio_x: Audio input tensor with shape (batch_size, sequence_len, audio_in)
            video_x: Video input tensor with shape (batch_size, sequence_len, video_in)
        '''

        # Convert inputs from batch-first to time-first layout (t, n, d)

        # Align all modalities to the same sequence length (use minimum)
        min_len = min(text_x.shape[1], audio_x.shape[1], video_x.shape[1])
        text_x = text_x[:, :min_len, :]
        audio_x = audio_x[:, :min_len, :]
        video_x = video_x[:, :min_len, :]

        text_x = text_x.permute(1, 0, 2)
        audio_x = audio_x.permute(1, 0, 2)
        video_x = video_x.permute(1, 0, 2)

        # Get the number of time steps and batch size
        n = text_x.size()[1]  # batch size
        t = text_x.size()[0]  # sequence length

        self.h_l = torch.zeros(n, self.dh_l).to(text_x.device)  # Text hidden state
        self.h_a = torch.zeros(n, self.dh_a).to(text_x.device)  # Audio hidden state
        self.h_v = torch.zeros(n, self.dh_v).to(text_x.device)  # Video hidden state
        self.c_l = torch.zeros(n, self.dh_l).to(text_x.device)  # Text cell state
        self.c_a = torch.zeros(n, self.dh_a).to(text_x.device)  # Audio cell state
        self.c_v = torch.zeros(n, self.dh_v).to(text_x.device)  # Video cell state

        self.mem = torch.zeros(n, self.mem_dim).to(text_x.device)  # Initialize the cross-modal memory vector

        all_h_ls = []
        all_h_as = []
        all_h_vs = []
        all_mems = []
        all_efficacies = []  # Store attention efficacies at each time step

        for i in range(t):
            # Save previous hidden states for unimodal memory modeling
            prev_h_l = self.h_l
            prev_h_a = self.h_a
            prev_h_v = self.h_v

            # Update each modality independently with its LSTM at the current time step
            new_h_l, new_c_l = self.lstm_l(text_x[i], (self.h_l, self.c_l))
            new_h_a, new_c_a = self.lstm_a(audio_x[i], (self.h_a, self.c_a))
            new_h_v, new_c_v = self.lstm_v(video_x[i], (self.h_v, self.c_v))

            # Model modality updates using previous and current hidden states
            l_singleton_input = self.l_transform([prev_h_l, new_h_l])
            a_singleton_input = self.a_transform([prev_h_a, new_h_a])
            v_singleton_input = self.v_transform([prev_h_v, new_h_v])

            l_singleton_input = l_singleton_input if 'l' in self.abl else torch.zeros_like(l_singleton_input)
            a_singleton_input = a_singleton_input if 'a' in self.abl else torch.zeros_like(a_singleton_input)
            v_singleton_input = v_singleton_input if 'v' in self.abl else torch.zeros_like(v_singleton_input)

            if not self.noS: # Temporal memory enabled
                attended, _, efficacies = self.graph_mfn([l_singleton_input, a_singleton_input, v_singleton_input])
                all_efficacies.append(efficacies.cpu().detach().tolist())  # Store attention weights at the current time step
                self.MSM(attended)  # Update memory state
                all_mems.append(self.mem)  # Record current memory

            self.h_l, self.c_l = new_h_l, new_c_l
            self.h_a, self.c_a = new_h_a, new_c_a
            self.h_v, self.c_v = new_h_v, new_c_v
            all_h_ls.append(self.h_l)
            all_h_as.append(self.h_a)
            all_h_vs.append(self.h_v)

        if self.noS:
            attended, _, efficacies = self.graph_mfn([l_singleton_input, a_singleton_input, v_singleton_input])
            all_efficacies.append(efficacies.cpu().detach().tolist())
            self.mem = self.MSM(attended)
            all_mems.append(self.mem)

        # Extract modality outputs and memory
        # Get the final hidden state of each modality and the memory module (batch_size x hidden_dim)
        last_h_l = all_h_ls[-1]  # Text modality (language)
        last_h_a = all_h_as[-1]  # Audio modality (acoustic)
        last_h_v = all_h_vs[-1]  # Video modality (visual)
        last_mem = all_mems[-1]  # Multimodal memory output (usually retained during ablation)

        # Modality ablation control (ablation-aware concatenation)
        # Replace disabled modality outputs with zero tensors to preserve dimensions
        if 'l' not in self.abl:
            last_h_l = torch.zeros_like(last_h_l)
        if 'a' not in self.abl:
            last_h_a = torch.zeros_like(last_h_a)
        if 'v' not in self.abl:
            last_h_v = torch.zeros_like(last_h_v)

        # Feature concatenation: construct the final vector from the selected modalities
        # Concatenate according to self.abl (retain l, a, v, or their combinations)
        if self.abl == "l":
            last_hs = last_h_l
        elif self.abl == "a":
            last_hs = last_h_a
        elif self.abl == "v":
            last_hs = last_h_v
        elif self.abl == "la":
            last_hs = self.out_concant([last_h_l, last_h_a, last_mem])
        elif self.abl == "lv":
            last_hs = self.out_concant([last_h_l, last_h_v, last_mem])
        elif self.abl == "av":
            last_hs = self.out_concant([last_h_a, last_h_v, last_mem])
        elif self.abl == "lav":
            last_hs = self.out_concant([last_h_l, last_h_a, last_h_v, last_mem])

        # Output layers (two fully connected layers, ReLU, and dropout)
        # 1. First fully connected layer
        out_fc1_result = self.out_fc1(last_hs)
        # 2. ReLU activation
        relu_result = F.relu(out_fc1_result)
        # 3. Dropout layer to prevent overfitting
        dropout_result = self.out_dropout(relu_result)
        # 4. Second fully connected layer producing final predictions
        output = self.out_fc2(dropout_result)

        # Return results
        # Include the main task output M and modality fusion weights (efficacies)
        res = {
            'M': output,
            'Efficacies': all_efficacies
        }
        return res
