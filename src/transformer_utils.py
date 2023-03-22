import torch
import torch.nn as nn
import math
import pickle
import numpy as np
import json
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from utility import top_bottom


def right_area(joint, player, multi_point, first_d):
    a = joint[player][15][0] + joint[player][16][0]
    b = multi_point[2][0] + multi_point[17][0]
    if first_d == 1:
        return True if a < b else False
    else:
        return True if a > b else False


def check_pos_and_score(d_list, joint_list, mp, top_bot_score):
    if d_list.index(1) < d_list.index(2):
        first_d = 1
        index = d_list.index(1) - 3
    else:
        first_d = 2
        index = d_list.index(2) - 3

    top, bot = top_bottom(joint_list[index])
    player = top if first_d == 1 else bot
    pi = 0 if first_d == 1 else 1
    right = right_area(joint_list[index], player, mp, first_d)
    if right:
        if top_bot_score[pi] % 2 == 0:
            return True, top_bot_score
        else:
            top_bot_score[pi] += 1
            return False, top_bot_score
    else:
        if top_bot_score[pi] % 2 == 1:
            return True, top_bot_score
        else:
            top_bot_score[pi] += 1
            return False, top_bot_score


def get_original_data(path):
    data_x = []
    with open(path, 'r') as score_json:
        frame_dict = json.load(score_json)

    score_x = []

    for i in range(len(frame_dict['frames'])):
        joint = np.array(frame_dict['frames'][i]['joint'])

        top, bot = top_bottom(joint)
        if top != 1:
            t = []
            t.append(joint[bot])
            t.append(joint[top])
            joint = np.array(t)

        score_x.append(joint)

    score_x = np.array(score_x)
    data_x.append(score_x)

    return data_x


def get_data(path, sc_root='model_weights/scaler_12.pickle'):
    sc = pickle.load(open(sc_root, 'rb'))
    c_count = 0
    n_count = 0
    data_x = []
    # vid_name = path.split('/')[-3]
    with open(path, 'r') as score_json:
        frame_dict = json.load(score_json)

    score_x = []

    for i in range(len(frame_dict['frames'])):
        temp_x = []
        joint = np.array(frame_dict['frames'][i]['joint'])

        top, bot = top_bottom(joint)
        if top != 1:
            c_count += 1
            t = []
            t.append(joint[bot])
            t.append(joint[top])
            joint = np.array(t)
        else:
            n_count += 1

        for p in range(2):
            temp_x.append(joint[p][5:])  # ignore head part
        temp_x = np.array(temp_x)  # 2, 12, 2
        temp_x = np.reshape(temp_x, [1, -1])
        temp_x = sc.transform(temp_x)
        temp_x = np.reshape(temp_x, [2, 12, 2])

        score_x.append(temp_x)

    score_x = np.array(score_x)
    data_x.append(score_x)

    return data_x


# model --------------------------------------------------------
class coordinateEmbedding(nn.Module):
    def __init__(self, in_channels: int, emb_size: int):
        super().__init__()
        half_emb = int(emb_size / 2)
        self.projection1 = nn.Linear(in_channels, half_emb)
        self.projection1_2 = nn.Linear(half_emb, half_emb)
        self.projection2 = nn.Linear(in_channels, half_emb)
        self.projection2_2 = nn.Linear(half_emb, half_emb)

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], -1)
        p1 = x.select(2, 0)
        p2 = x.select(2, 1)
        p1 = self.projection1(p1)
        p2 = self.projection2(p2)
        p1 = self.projection1_2(p1)
        p2 = self.projection2_2(p2)
        projected = torch.cat((p1, p2), 2)
        return projected


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)  # 0, 1, 2, 3, 4, 5  10,1
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)  # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class Optimus_Prime(nn.Module):
    def __init__(self, num_tokens, dim_model, num_heads, num_encoder_layers, dim_feedforward, dropout_p=0):
        super().__init__()
        # INFO
        self.dim_model = dim_model

        # LAYERS
        self.positional_encoder = PositionalEncoding(dim_model=dim_model, dropout_p=dropout_p, max_len=600)

        self.xy_embedding = coordinateEmbedding(in_channels=24, emb_size=dim_model)

        encoder_layers = TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout_p)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)

        self.decoder1 = nn.Linear(dim_model, dim_model)
        self.decoder2 = nn.Linear(dim_model, num_tokens)

    def forward(self, src, src_pad_mask=None):
        src = self.xy_embedding(src)
        src = self.positional_encoder(src) * math.sqrt(self.dim_model)
        src = src.permute(1, 0, 2)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        output = self.transformer_encoder(src, src_key_padding_mask=src_pad_mask)
        output = F.relu(self.decoder1(output))
        output = self.decoder2(output)
        return output

    def create_src_pad_mask(self, matrix: torch.tensor, PAD_array=np.zeros((1, 2, 12, 2))) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        src_pad_mask = []
        PAD_array = torch.tensor(PAD_array).squeeze(0).to(device)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                a = matrix[i][j]
                src_pad_mask.append(torch.equal(a, PAD_array))
        src_pad_mask = torch.tensor(src_pad_mask).unsqueeze(0).reshape(matrix.shape[0], -1).to(device)
        return src_pad_mask


def build_model(path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Optimus_Prime(
        num_tokens=4, dim_model=512, num_heads=8, num_encoder_layers=8, dim_feedforward=1024,
        dropout_p=0
    ).to(device)

    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def predict(model, input_sequence):
    model.eval()
    # Get source mask
    src_pad_mask = model.create_src_pad_mask(input_sequence)
    pred = model(input_sequence, src_pad_mask=src_pad_mask)
    pred_indices = torch.max(pred.detach(), 2).indices.squeeze(-1)

    return pred_indices


# kp = [[474, 433], [1446, 415], [382, 708], [1534, 704], [269, 1040], [1648, 1040]]
#
# get_data('E:/test_videos/outputs/p_test/json/score_0_285_546.json', kp)
