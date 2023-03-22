import copy, cv2, json, time, numpy as np
import os
import shutil
from PIL import Image
import torch, torchvision
from torchvision.transforms import transforms
from torchvision.transforms import functional as F
import scene_utils, transformer_utils
from shot_recognition import check_hit_frame, add_result, add_result2
from utility import check_dir, get_path, parse_time, top_bottom, correction, extension, type_classify, count_percentage, \
    counts_three, to_float
from transformer_utils import coordinateEmbedding, PositionalEncoding, Optimus_Prime
from scene_utils import scene_classifier


def generate_player_strategy(base, vid_name):
    paths = get_path(f"{base}/outputs/{vid_name}/scores")
    shot_dict = {
        '長球': '0',
        '切球': '1',
        '殺球': '2',
        '挑球': '3',
        '小球': '4',
        '平球': '5',
        '撲球': '6',
    }
    blue_win_shots_and_nums = {}
    blue_loss_shots_and_nums = {}
    red_win_shots_and_nums = {}
    red_loss_shots_and_nums = {}
    for path in paths:
        with open(f"{path}/info.json", 'r', encoding="utf-8") as f:
            frame_dict = json.load(f)
        game_num = path.split('/')[-1].split('_')[1]
        score_num = path.split('/')[-1].split('_')[-1]
        blue_shots, red_shots = [], []
        winner = frame_dict['winner']
        blue_serve_first = frame_dict['blue serve first']
        shot_list = frame_dict['shot list']
        if len(shot_list) < 6:
            continue
        for i in range(6):
            if (blue_serve_first and len(shot_list) % 2 == 1) or (not blue_serve_first and len(shot_list) % 2 == 0):
                if i % 2 == 0:
                    red_shots.append(shot_list[i - 6][0].split(' ')[-1])
                else:
                    blue_shots.append(shot_list[i - 6][0].split(' ')[-1])
            else:
                if i % 2 == 0:
                    blue_shots.append(shot_list[i - 6][0].split(' ')[-1])
                else:
                    red_shots.append(shot_list[i - 6][0].split(' ')[-1])
        blue_key = shot_dict[blue_shots[-3]] + shot_dict[blue_shots[-2]] + shot_dict[blue_shots[-1]]
        red_key = shot_dict[red_shots[-3]] + shot_dict[red_shots[-2]] + shot_dict[red_shots[-1]]
        if winner:
            if blue_key in blue_win_shots_and_nums.keys():
                blue_win_shots_and_nums[blue_key].append((game_num, score_num))
            else:
                blue_win_shots_and_nums[blue_key] = [(game_num, score_num)]
            if red_key in red_loss_shots_and_nums.keys():
                red_loss_shots_and_nums[red_key].append((game_num, score_num))
            else:
                red_loss_shots_and_nums[red_key] = [(game_num, score_num)]
        else:
            if blue_key in blue_loss_shots_and_nums.keys():
                blue_loss_shots_and_nums[blue_key].append((game_num, score_num))
            else:
                blue_loss_shots_and_nums[blue_key] = [(game_num, score_num)]
            if red_key in red_win_shots_and_nums.keys():
                red_win_shots_and_nums[red_key].append((game_num, score_num))
            else:
                red_win_shots_and_nums[red_key] = [(game_num, score_num)]

    blue_win_keys = counts_three(blue_win_shots_and_nums)
    blue_loss_keys = counts_three(blue_loss_shots_and_nums)
    red_win_keys = counts_three(red_win_shots_and_nums)
    red_loss_keys = counts_three(red_loss_shots_and_nums)

    bwk_list, blk_list, rwk_list, rlk_list = [], [], [], []
    for i in range(3):
        index = - i - 1
        blue_win_key = blue_win_keys[index][0]
        red_win_key = red_win_keys[index][0]
        blue_loss_key = blue_loss_keys[index][0]
        red_loss_key = red_loss_keys[index][0]
        output_highlights(f"{base}/outputs/{vid_name}", blue_win_shots_and_nums[blue_win_key], True, i + 1)
        output_highlights(f"{base}/outputs/{vid_name}", red_win_shots_and_nums[red_win_key], False, i + 1)
        bwk_list.append(code_to_name(blue_win_key))
        blk_list.append(code_to_name(blue_loss_key))
        rwk_list.append(code_to_name(red_win_key))
        rlk_list.append(code_to_name(red_loss_key))

    hl_info_dict = {
        'blue win key': bwk_list,
        'blue loss key': blk_list,
        'red win key': rwk_list,
        'red loss key': rlk_list,
    }
    hl_info_save_path = f"{base}/outputs/{vid_name}/highlights.json"
    with open(hl_info_save_path, 'w', encoding="utf-8") as f:
        json.dump(hl_info_dict, f, indent=2, ensure_ascii=False)

    return True


def code_to_name(code):
    shots = []
    trans_dict = {
        '0': '長球',
        '1': '切球',
        '2': '殺球',
        '3': '挑球',
        '4': '小球',
        '5': '平球',
        '6': '撲球',
    }
    shots.append(trans_dict[code[0]])
    shots.append(trans_dict[code[1]])
    shots.append(trans_dict[code[2]])
    return shots


def output_highlights(base, num_list, blue, i):
    img_list = []
    player = 'blue' if blue else 'red'
    print(f'Generating {player} Highlights')
    for nums in num_list:
        game_num = nums[0]
        score_num = nums[1]
        cap = cv2.VideoCapture(f'{base}/scores/game_{game_num}_score_{score_num}/video.mp4')
        if not cap.isOpened():
            print('Error while trying to read video. Please check path again')
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        FPS = int(cap.get(5))
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                img_list.append(frame)
            else:
                break

    out = cv2.VideoWriter(f"{base}/{player}_highlights_{i}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), FPS,
                          (frame_width, frame_height))
    for frame in img_list:
        out.write(frame)
    out.release()
    print('Done!')

    return True


def check_type(last_type, wait_list):
    sum = 0
    if last_type == 1:
        for pair in wait_list:
            sum += pair[0]
        if sum <= 3:
            return True
        else:
            return False
    else:
        for pair in wait_list:
            sum += pair[0]
        if sum >= 3:
            return True
        else:
            return False


# check if player is in court
def in_court(court_info, court_points, joint):
    l_a = court_info[0]
    l_b = court_info[1]
    r_a = court_info[2]
    r_b = court_info[3]
    ankle_x = (joint[15][0] + joint[16][0]) / 2
    ankle_y = (joint[15][1] + joint[16][1]) / 2
    top = ankle_y > court_points[0][1]
    bottom = ankle_y < court_points[5][1]
    lmp_x = (ankle_y - l_b) / l_a
    rmp_x = (ankle_y - r_b) / r_a
    left = ankle_x > lmp_x
    right = ankle_x < rmp_x

    if left and right and top and bottom:
        return True
    else:
        return False


# get the index of the in court players
def score_rank(court_info, court_points, joints):
    indexes = []
    for i in range(len(joints)):
        if in_court(court_info, court_points, joints[i]):
            indexes.append(i)
    if len(indexes) < 2:
        return False
    else:
        return indexes


# check if up court and bot court got player
def check_pos(court_mp, indices, boxes):
    for i in range(len(indices)):
        combination = 1
        if boxes[indices[0]][1] < court_mp < boxes[indices[combination]][3]:
            return True, [0, combination]
        elif boxes[indices[0]][3] > court_mp > boxes[indices[combination]][1]:
            return True, [0, combination]
        else:
            combination += 1
    return False, [0, 0]


# get and set the court information
def get_court_info(frame_height, court_kp_model, court_kp_model_old, img):
    with torch.no_grad():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        img = F.to_tensor(img)
        img = img.unsqueeze(0)
        img = img.to(device)
        output = court_kp_model(img)
        output_old = court_kp_model_old(img)
    scores = output[0]['scores'].detach().cpu().numpy()
    high_scores_idxs = np.where(scores > 0.7)[0].tolist()
    post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs],
                                        output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy()
    keypoints = []
    for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
        keypoints.append([list(map(int, kp[:2])) for kp in kps])
    scores_old = output_old[0]['scores'].detach().cpu().numpy()
    high_scores_idxs_old = np.where(scores_old > 0.7)[0].tolist()
    post_nms_idxs_old = torchvision.ops.nms(output_old[0]['boxes'][high_scores_idxs_old],
                                            output_old[0]['scores'][high_scores_idxs_old], 0.3).cpu().numpy()
    keypoints_old = []
    for kps in output_old[0]['keypoints'][high_scores_idxs_old][post_nms_idxs_old].detach().cpu().numpy():
        keypoints_old.append([list(map(int, kp[:2])) for kp in kps])
    try:
        true_court_points = copy.deepcopy(keypoints[0])
    except:
        return False, False, False, False

    multi_points = extension(correction(np.array(keypoints[0]))).tolist()
    print(multi_points)
    keypoints_old[0][0][0] -= 80
    keypoints_old[0][0][1] -= 80
    keypoints_old[0][1][0] += 80
    keypoints_old[0][1][1] -= 80
    keypoints_old[0][2][0] -= 80
    keypoints_old[0][3][0] += 80
    keypoints_old[0][4][0] -= 80
    keypoints_old[0][4][1] = min(keypoints_old[0][4][1] + 80, frame_height - 40)
    keypoints_old[0][5][0] += 80
    keypoints_old[0][5][1] = min(keypoints_old[0][5][1] + 80, frame_height - 40)
    court_points = keypoints_old[0]

    l_a = (true_court_points[0][1] - true_court_points[4][1]) / (true_court_points[0][0] - true_court_points[4][0])
    l_b = true_court_points[0][1] - l_a * true_court_points[0][0]
    r_a = (true_court_points[1][1] - true_court_points[5][1]) / (true_court_points[1][0] - true_court_points[5][0])
    r_b = true_court_points[1][1] - r_a * true_court_points[1][0]
    mp_y = (true_court_points[2][1] + true_court_points[3][1]) / 2
    court_info = [l_a, l_b, r_a, r_b, mp_y]

    return multi_points, true_court_points, court_points, court_info


# update the score using the serving player of next score
def update_score(base, vid_name, game, score, shuttle_direction, blue_red_score, flip, win_loss_dicts, move_dir_list,
                 win_loss_movement_dict):
    with open(f"{base}/outputs/{vid_name}/scores/game_{game}_score_{score}/info.json", 'r',
              encoding="utf-8") as score_json:
        dict = json.load(score_json)
        shot_list = dict['shot list']
        bsv = dict['blue serve first']
    if shuttle_direction is not None:
        if 1 in shuttle_direction and 2 in shuttle_direction:
            winner = True if shuttle_direction.index(1) < shuttle_direction.index(2) else False
        elif 1 in shuttle_direction and 2 not in shuttle_direction:
            winner = True
        else:
            winner = False
        winner = winner if not flip else not winner
    else:
        winner = True if blue_red_score[0] > blue_red_score[1] else False

    dict['winner'] = winner # True implies blue win
    if winner:
        dict['blue red score'][0] += 1
        blue_red_score[0] += 1
        for i in range(len(move_dir_list)):
            if bsv:
                if i % 2 == 0:
                    win_loss_movement_dict[3][move_dir_list[i][0]] += 1
                else:
                    win_loss_movement_dict[0][move_dir_list[i][0]] += 1
            else:
                if i % 2 == 0:
                    win_loss_movement_dict[0][move_dir_list[i][0]] += 1
                else:
                    win_loss_movement_dict[3][move_dir_list[i][0]] += 1
        for i in range(len(shot_list)):
            if bsv:
                if i % 2 == 0:
                    win_loss_dicts[0][shot_list[i][0].split(' ')[-1]] += 1
                else:
                    win_loss_dicts[3][shot_list[i][0].split(' ')[-1]] += 1
            else:
                if i % 2 == 0:
                    win_loss_dicts[3][shot_list[i][0].split(' ')[-1]] += 1
                else:
                    win_loss_dicts[0][shot_list[i][0].split(' ')[-1]] += 1
    else:
        dict['blue red score'][1] += 1
        blue_red_score[1] += 1
        for i in range(len(move_dir_list)):
            if bsv:
                if i % 2 == 0:
                    win_loss_movement_dict[2][move_dir_list[i][0]] += 1
                else:
                    win_loss_movement_dict[1][move_dir_list[i][0]] += 1
            else:
                if i % 2 == 0:
                    win_loss_movement_dict[1][move_dir_list[i][0]] += 1
                else:
                    win_loss_movement_dict[2][move_dir_list[i][0]] += 1
        for i in range(len(shot_list)):
            if bsv:
                if i % 2 == 0:
                    win_loss_dicts[1][shot_list[i][0].split(' ')[-1]] += 1
                else:
                    win_loss_dicts[2][shot_list[i][0].split(' ')[-1]] += 1
            else:
                if i % 2 == 0:
                    win_loss_dicts[2][shot_list[i][0].split(' ')[-1]] += 1
                else:
                    win_loss_dicts[1][shot_list[i][0].split(' ')[-1]] += 1

    with open(f"{base}/outputs/{vid_name}/scores/game_{game}_score_{score}/info.json", 'w', encoding="utf-8") as f:
        json.dump(dict, f, indent=2, ensure_ascii=False)

    return blue_red_score, win_loss_dicts, win_loss_movement_dict


class video_resolver:
    def __init__(self, vid_path, output_base='E:/test_videos', isExit=False):
        self.base = output_base
        self.vid_path = vid_path
        self.vid_name = vid_path.split('/')[-1].split('.')[0]
        if not isExit:
            print("Video haven't been resolved")
            self.start_time = time.time()
            self.transform = transforms.Compose([transforms.ToTensor()])
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.paths = [f"{self.base}/outputs",
                          f"{self.base}/outputs/eliminated",
                          f"{self.base}/outputs/{self.vid_name}",
                          f"{self.base}/outputs/{self.vid_name}/scores",
                          f"{self.base}/outputs/eliminated/{self.vid_name}"]
            for path in self.paths:
                check_dir(path)

            self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True, num_keypoints=17)
            self.model.to(self.device).eval()
            self.scene_model = scene_utils.build_model('model_weights/scene_classifier.pt', self.device)
            self.bsp_model = transformer_utils.build_model('model_weights/weights/clean_seq_labling_ultimate_2.pt')

            self.court_kp_model = torch.load('model_weights/court_kpRCNN.pth')
            self.court_kp_model.to(self.device).eval()

            self.court_kp_model_old = torch.load('model_weights/court_kpRCNN_old.pth')
            self.court_kp_model_old.to(self.device).eval()

            self.court_points = None
            self.true_court_points = None
            self.multi_points = None
            self.court_info = None

            self.cap = cv2.VideoCapture(vid_path)

            if not self.cap.isOpened():
                print('Error while trying to read video. Please check path again')
        else:
            print("Video resolved")

    def draw_key_points(self, outputs, image, flip):
        edges = [(0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (11, 12), (5, 7),
                 (7, 9), (5, 11), (11, 13), (13, 15), (6, 12), (12, 14), (14, 16), (5, 6)]
        c_edges = [[0, 1], [0, 5], [1, 2], [1, 6], [2, 3], [2, 7], [3, 4], [3, 8], [4, 9],
                   [5, 6], [5, 10], [6, 7], [6, 11], [7, 8], [7, 12], [8, 9], [8, 13], [9, 14],
                   [10, 11], [10, 15], [11, 12], [11, 16], [12, 13], [12, 17], [13, 14], [13, 18],
                   [14, 19], [15, 16], [15, 20], [16, 17], [16, 21], [17, 18], [17, 22], [18, 19],
                   [18, 23], [19, 24], [20, 21], [20, 25], [21, 22], [21, 26], [22, 23], [22, 27],
                   [23, 24], [23, 28], [24, 29], [25, 26], [25, 30], [26, 27], [26, 31], [27, 28],
                   [27, 32], [28, 29], [28, 33], [29, 34], [30, 31], [31, 32], [32, 33], [33, 34]]

        b = outputs[0]['boxes'].cpu().detach().numpy()
        j = outputs[0]['keypoints'].cpu().detach().numpy()
        in_court_indices = score_rank(self.court_info, self.court_points, j)
        filtered_joint = []
        if in_court_indices == False:
            return image, None
        fit, combination = check_pos(self.court_info[4], in_court_indices, b)
        filtered_joint.append(j[in_court_indices[combination[0]]].tolist())
        filtered_joint.append(j[in_court_indices[combination[1]]].tolist())
        # filtered_joint = np.array(filtered_joint)
        pos = top_bottom(filtered_joint)
        # top: blue, bot: red
        top_color_1 = (255, 0, 0)
        bot_color_1 = (0, 0, 255)
        top_color_joint = (115, 47, 14)
        bot_color_joint = (35, 47, 204)
        if fit:
            for i in range(2):
                p = pos[i]
                if not flip:
                    color = top_color_1 if i == 0 else bot_color_1
                    color_joint = top_color_joint if i == 0 else bot_color_joint
                else:
                    color = bot_color_1 if i == 0 else top_color_1
                    color_joint = bot_color_joint if i == 0 else top_color_joint
                keypoints = np.array(filtered_joint[p])  # 17, 3
                keypoints = keypoints[:, :].reshape(-1, 3)
                overlay = image.copy()
                # draw the court
                for e in c_edges:
                    cv2.line(overlay, (int(self.multi_points[e[0]][0]), int(self.multi_points[e[0]][1])),
                             (int(self.multi_points[e[1]][0]), int(self.multi_points[e[1]][1])),
                             (53, 195, 242), 2, lineType=cv2.LINE_AA)
                # for kps in [self.court_points]:
                for kps in [self.multi_points]:
                    for idx, kp in enumerate(kps):
                        cv2.circle(overlay, tuple(kp), 2, (5, 135, 242), 10)
                alpha = 0.4
                image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

                for p in range(keypoints.shape[0]):
                    cv2.circle(image, (int(keypoints[p, 0]), int(keypoints[p, 1])), 3, color_joint, thickness=-1,
                               lineType=cv2.FILLED)

                for ie, e in enumerate(edges):
                    # get different colors for the edges
                    # rgb = matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0])
                    # rgb = rgb * 255
                    # join the keypoint pairs to draw the skeletal structure
                    cv2.line(image, (int(keypoints[e, 0][0]), int(keypoints[e, 1][0])),
                             (int(keypoints[e, 0][1]), int(keypoints[e, 1][1])),
                             color, 2, lineType=cv2.LINE_AA)
            return image, filtered_joint
        else:
            return image, None

    def resolve(self):
        b_total_shot_dict = {
            '長球': 0,
            '切球': 0,
            '殺球': 0,
            '挑球': 0,
            '小球': 0,
            '平球': 0,
            '撲球': 0,
        }
        r_total_shot_dict = {
            '長球': 0,
            '切球': 0,
            '殺球': 0,
            '挑球': 0,
            '小球': 0,
            '平球': 0,
            '撲球': 0,
        }
        b_total_move_dict = {
            'DLBL': 0,
            'DLBR': 0,
            'DLFL': 0,
            'DLFR': 0,
            'DSBL': 0,
            'DSBR': 0,
            'DSFL': 0,
            'DSFR': 0,
            'LLB': 0,
            'LLF': 0,
            'LSB': 0,
            'LSF': 0,
            'TLL': 0,
            'TLR': 0,
            'TSL': 0,
            'TSR': 0,
            'NM': 0
        }
        r_total_move_dict = {
            'DLBL': 0,
            'DLBR': 0,
            'DLFL': 0,
            'DLFR': 0,
            'DSBL': 0,
            'DSBR': 0,
            'DSFL': 0,
            'DSFR': 0,
            'LLB': 0,
            'LLF': 0,
            'LSB': 0,
            'LSF': 0,
            'TLL': 0,
            'TLR': 0,
            'TSL': 0,
            'TSR': 0,
            'NM': 0
        }
        # b_win_dict, b_loss_dict, r_win_dict, r_loss_dict, b_win_loss_move_dict, r_win_loss_move_dict
        win_loss_dicts = [{
            '長球': 0,
            '切球': 0,
            '殺球': 0,
            '挑球': 0,
            '小球': 0,
            '平球': 0,
            '撲球': 0
        }, {
            '長球': 0,
            '切球': 0,
            '殺球': 0,
            '挑球': 0,
            '小球': 0,
            '平球': 0,
            '撲球': 0
        }, {
            '長球': 0,
            '切球': 0,
            '殺球': 0,
            '挑球': 0,
            '小球': 0,
            '平球': 0,
            '撲球': 0
        }, {
            '長球': 0,
            '切球': 0,
            '殺球': 0,
            '挑球': 0,
            '小球': 0,
            '平球': 0,
            '撲球': 0
        }]
        win_loss_movement_dict = [{
            'DLBL': 0,
            'DLBR': 0,
            'DLFL': 0,
            'DLFR': 0,
            'DSBL': 0,
            'DSBR': 0,
            'DSFL': 0,
            'DSFR': 0,
            'LLB': 0,
            'LLF': 0,
            'LSB': 0,
            'LSF': 0,
            'TLL': 0,
            'TLR': 0,
            'TSL': 0,
            'TSR': 0,
            'NM': 0
        }, {
            'DLBL': 0,
            'DLBR': 0,
            'DLFL': 0,
            'DLFR': 0,
            'DSBL': 0,
            'DSBR': 0,
            'DSFL': 0,
            'DSFR': 0,
            'LLB': 0,
            'LLF': 0,
            'LSB': 0,
            'LSF': 0,
            'TLL': 0,
            'TLR': 0,
            'TSL': 0,
            'TSR': 0,
            'NM': 0
        }, {
            'DLBL': 0,
            'DLBR': 0,
            'DLFL': 0,
            'DLFR': 0,
            'DSBL': 0,
            'DSBR': 0,
            'DSFL': 0,
            'DSFR': 0,
            'LLB': 0,
            'LLF': 0,
            'LSB': 0,
            'LSF': 0,
            'TLL': 0,
            'TLR': 0,
            'TSL': 0,
            'TSR': 0,
            'NM': 0
        }, {
            'DLBL': 0,
            'DLBR': 0,
            'DLFL': 0,
            'DLFR': 0,
            'DSBL': 0,
            'DSBR': 0,
            'DSFL': 0,
            'DSFR': 0,
            'LLB': 0,
            'LLF': 0,
            'LSB': 0,
            'LSF': 0,
            'TLL': 0,
            'TLR': 0,
            'TSL': 0,
            'TSR': 0,
            'NM': 0
        }]

        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))
        frame_count, saved_count = 0, 0
        time_rate = 0.1
        FPS = self.cap.get(5)
        frame_rate = round(int(FPS) * time_rate)
        total_frame_count = int(self.cap.get(7))
        total_saved_count = int(total_frame_count / frame_rate)

        wait_list, joint_list, joint_img_list = [], [], []
        last_type = 0
        game, score = 1, 0
        zero_count, one_count = 0, 0
        last_score = 0
        blue_red_score, res_game_info = [0, 0], []
        start_recording, flip = True, False
        start_frame, end_frame = 0, 0
        prev_shot_list, prev_move_dir_list = False, False

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                if frame_count % frame_rate == 0:
                    sceneImg = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    sceneImg = scene_utils.preprocess(sceneImg, self.device)
                    # slice video into score videos
                    p = scene_utils.predict(self.scene_model, sceneImg)
                    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    if len(wait_list) < 5:
                        wait_list.append((p, pil_image, frame))
                    else:
                        tup = wait_list.pop(0)
                        wait_list.append((p, pil_image, frame))
                        p = tup[0]
                        pil_image = tup[1]
                        frame = tup[2]
                        if p != last_type:
                            correct = check_type(last_type, wait_list)
                            if not correct:
                                p = 0 if p == 1 else 1
                            if p == 0:
                                if one_count > 25 and len(joint_list) / one_count > 0.6:  # 25 is changable
                                    if not start_recording:
                                        start_recording = True
                                        end_frame = frame_count
                                    eli_repeat = 0
                                    store_path = f"{self.base}/outputs/{self.vid_name}/scores/game_{game}_score_{score}"
                                    eli_path = f"{self.base}/outputs/eliminated/{self.vid_name}/game_{game}_score_{score}"
                                    check_dir(store_path)

                                    start_time = parse_time(FPS, start_frame)
                                    end_time = parse_time(FPS, end_frame)
                                    framesDict = {'frames': joint_list}
                                    joint_save_path = f"{store_path}/score_{score}_joint.json"
                                    with open(joint_save_path, 'w', encoding="utf-8") as f:
                                        json.dump(framesDict, f, indent=2, ensure_ascii=False)

                                    joint_list = torch.tensor(np.array(transformer_utils.get_data(joint_save_path,
                                                                                                  'model_weights/scaler_ultimate_2.pickle')),
                                                              dtype=torch.float32).to(self.device)
                                    orig_joint_list = np.squeeze(
                                        np.array(transformer_utils.get_original_data(joint_save_path)), axis=0)

                                    shuttle_direction = transformer_utils.predict(self.bsp_model, joint_list).tolist()
                                    print(shuttle_direction)
                                    d_zero_count = 0
                                    for d in shuttle_direction:
                                        if d == 0:
                                            d_zero_count += 1
                                    if d_zero_count / len(shuttle_direction) < 0.9:
                                        shot_list, move_dir_list = check_hit_frame(shuttle_direction, orig_joint_list,
                                                                                   self.true_court_points,
                                                                                   self.multi_points)
                                        print(shot_list, move_dir_list)

                                        out = cv2.VideoWriter(f"{store_path}/video.mp4",
                                                              cv2.VideoWriter_fourcc(*'mp4v'), int(FPS / frame_rate),
                                                              (frame_width, frame_height))
                                        try:
                                            success = add_result2(out, joint_img_list, shot_list, move_dir_list)
                                        except:
                                            print('No Shots...')
                                            continue
                                        out.release()
                                        if prev_shot_list != False and prev_move_dir_list != False:
                                            if score != 0:
                                                blue_red_score, win_loss_dicts, win_loss_movement_dict = update_score(
                                                    self.base, self.vid_name, game,
                                                    score - 1,
                                                    shuttle_direction,
                                                    blue_red_score, flip,
                                                    win_loss_dicts, prev_move_dir_list, win_loss_movement_dict)
                                            if score == 0 and game != 1:
                                                blue_red_score, win_loss_dicts, win_loss_movement_dict = update_score(
                                                    self.base, self.vid_name,
                                                    game - 1,
                                                    last_score - 1,
                                                    shuttle_direction,
                                                    blue_red_score, flip,
                                                    win_loss_dicts, prev_move_dir_list, win_loss_movement_dict)
                                                if blue_red_score[0] == 1:
                                                    res_game_info[-1]['blue red score'][0] += 1
                                                else:
                                                    res_game_info[-1]['blue red score'][1] += 1
                                                blue_red_score = [0, 0]
                                        prev_move_dir_list = move_dir_list
                                        prev_shot_list = shot_list

                                        if 2 in shuttle_direction and 1 in shuttle_direction:
                                            up_serve_first = True if shuttle_direction.index(1) < shuttle_direction.index(
                                                2) else False
                                        elif 2 in shuttle_direction:
                                            up_serve_first = False
                                        elif 1 in shuttle_direction:
                                            up_serve_first = True

                                        # current point bsv
                                        if not flip and up_serve_first:
                                            bsv = True
                                        elif flip and not up_serve_first:
                                            bsv = True
                                        else:
                                            bsv = False
                                        b_shot_dict = {
                                            '長球': 0,
                                            '切球': 0,
                                            '殺球': 0,
                                            '挑球': 0,
                                            '小球': 0,
                                            '平球': 0,
                                            '撲球': 0,
                                        }
                                        r_shot_dict = {
                                            '長球': 0,
                                            '切球': 0,
                                            '殺球': 0,
                                            '挑球': 0,
                                            '小球': 0,
                                            '平球': 0,
                                            '撲球': 0,
                                        }
                                        b_move_dict = {
                                            'DLBL': 0,
                                            'DLBR': 0,
                                            'DLFL': 0,
                                            'DLFR': 0,
                                            'DSBL': 0,
                                            'DSBR': 0,
                                            'DSFL': 0,
                                            'DSFR': 0,
                                            'LLB': 0,
                                            'LLF': 0,
                                            'LSB': 0,
                                            'LSF': 0,
                                            'TLL': 0,
                                            'TLR': 0,
                                            'TSL': 0,
                                            'TSR': 0,
                                            'NM': 0
                                        }
                                        r_move_dict = {
                                            'DLBL': 0,
                                            'DLBR': 0,
                                            'DLFL': 0,
                                            'DLFR': 0,
                                            'DSBL': 0,
                                            'DSBR': 0,
                                            'DSFL': 0,
                                            'DSFR': 0,
                                            'LLB': 0,
                                            'LLF': 0,
                                            'LSB': 0,
                                            'LSF': 0,
                                            'TLL': 0,
                                            'TLR': 0,
                                            'TSL': 0,
                                            'TSR': 0,
                                            'NM': 0
                                        }
                                        for shot in shot_list:
                                            type = shot[0]
                                            if not flip:
                                                if shot[3]:
                                                    b_shot_dict[type.split(' ')[-1]] += 1
                                                    b_total_shot_dict[type.split(' ')[-1]] += 1
                                                else:
                                                    r_shot_dict[type.split(' ')[-1]] += 1
                                                    r_total_shot_dict[type.split(' ')[-1]] += 1
                                            else:
                                                if shot[3]:
                                                    r_shot_dict[type.split(' ')[-1]] += 1
                                                    r_total_shot_dict[type.split(' ')[-1]] += 1
                                                else:
                                                    b_shot_dict[type.split(' ')[-1]] += 1
                                                    b_total_shot_dict[type.split(' ')[-1]] += 1
                                        for move in move_dir_list:
                                            if not flip:
                                                if move[1]:
                                                    b_total_move_dict[move[0]] += 1
                                                    b_move_dict[move[0]] += 1
                                                else:
                                                    r_total_move_dict[move[0]] += 1
                                                    r_move_dict[move[0]] += 1
                                            else:
                                                if move[1]:
                                                    r_total_move_dict[move[0]] += 1
                                                    r_move_dict[move[0]] += 1
                                                else:
                                                    b_total_move_dict[move[0]] += 1
                                                    b_move_dict[move[0]] += 1

                                        info_dict = {
                                            'game': game,
                                            'score': score,
                                            'time': [start_time, end_time],
                                            'winner': None,
                                            'blue red score': blue_red_score,
                                            'shuttle direction': shuttle_direction,
                                            'shot list': shot_list,
                                            'blue shot dict': b_shot_dict,
                                            'red shot dict': r_shot_dict,
                                            'blue serve first': bsv,
                                            'move direction list': move_dir_list,
                                            'blue move dict': b_move_dict,
                                            'red move dict': r_move_dict
                                        }
                                        info_save_path = f"{store_path}/info.json"
                                        with open(info_save_path, 'w', encoding="utf-8") as f:
                                            json.dump(info_dict, f, indent=2, ensure_ascii=False)
                                        if success:
                                            print(f'Finish score_{score}')
                                            score += 1
                                    else:  # all 0
                                        zero_count = backup_z
                                        out = cv2.VideoWriter(f"{store_path}/video.mp4",
                                                              cv2.VideoWriter_fourcc(*'mp4v'), int(FPS / frame_rate),
                                                              (frame_width, frame_height))
                                        for img in joint_img_list:
                                            out.write(img)
                                        out.release()
                                        try:
                                            shutil.move(store_path, eli_path)
                                        except:
                                            eli_repeat += 1
                                            os.rename(store_path, f'{store_path}-{eli_repeat}')
                                            shutil.move(f'{store_path}-{eli_repeat}', eli_path)
                                print('clear')
                                one_count = 0
                                joint_list = []
                                joint_img_list = []
                            last_type = p
                        if p == 1:
                            # check if next game starts
                            if zero_count != 0 and 996 < zero_count < 1956 and 15 < max(blue_red_score) and game < 3:
                                res_game_info.append({f'score count': score, 'blue red score': blue_red_score})
                                last_score = score
                                game += 1
                                score = 0
                                flip = not flip
                                blue_red_score = [0, 0]
                            if game == 3 and zero_count != 0 and 480 < zero_count < 1094 and 9 < max(
                                    blue_red_score) < 21:
                                flip = not flip

                            backup_z = zero_count
                            zero_count = 0

                            # get the court info when first meet the right shooting angle
                            if self.court_points is None or self.court_points is False:
                                self.multi_points, self.true_court_points, self.court_points, self.court_info = get_court_info(
                                    frame_height, self.court_kp_model, self.court_kp_model_old,
                                    img=wait_list[2][2])
                                print(self.true_court_points)
                                if self.court_points == False:
                                    continue
                                else:
                                    self.court_kp_model = None
                                    self.court_kp_model_old = None
                            if start_recording:
                                start_frame = frame_count
                                start_recording = False
                            one_count += 1
                            image = self.transform(pil_image)
                            image = image.unsqueeze(0).to(self.device)
                            with torch.no_grad():
                                outputs = self.model(image)
                            output_image, player_joints = self.draw_key_points(outputs, frame, flip)

                            if player_joints is not None:
                                for points in player_joints:
                                    for i, joints in enumerate(points):
                                        points[i] = joints[0:2]
                                joint_list.append({
                                    'joint': player_joints,
                                })
                            joint_img_list.append(output_image)
                        else:
                            zero_count += 1
                    saved_count += 1
                    print(saved_count, ' / ', total_saved_count, ' ', self.vid_name, 'score: ', score, 'last score: ',
                          last_score, 'p: ', p)
                frame_count += 1
            else:
                break
        self.cap.release()
        cv2.destroyAllWindows()

        print(f"Time cost: {round(time.time() - self.start_time, 1)}")
        print(f'Frame count:{frame_count}')
        print(f'Save count:{saved_count}')
        print(f'Score: {score}')

        # last score
        res_game_info.append({f'score count': score, 'blue red score': blue_red_score})
        print(res_game_info, len(res_game_info))
        if score != 0:
            blue_red_score, win_loss_dicts, win_loss_movement_dict = update_score(self.base, self.vid_name,
                                                                                 len(res_game_info), score - 1,
                                                                                 None, blue_red_score, flip,
                                                                                 win_loss_dicts, prev_move_dir_list,
                                                                                 win_loss_movement_dict)
        else:
            blue_red_score, win_loss_dicts, win_loss_movement_dict = update_score(self.base, self.vid_name,
                                                                                 len(res_game_info) - 1, last_score - 1,
                                                                                 None, blue_red_score, flip,
                                                                                 win_loss_dicts, prev_move_dir_list,
                                                                                 win_loss_movement_dict)
        print(blue_red_score)

        # whole game
        with open(f"{self.base}/outputs/{self.vid_name}/game_info.json", 'w', encoding="utf-8") as f:
            json.dump({'games': res_game_info,
                       'blue win shots': win_loss_dicts[0],
                       'blue loss shots': win_loss_dicts[1],
                       'red win shots': win_loss_dicts[2],
                       'red loss shots': win_loss_dicts[3],
                       'blue win moves': win_loss_movement_dict[0],
                       'blue loss moves': win_loss_movement_dict[1],
                       'red win moves': win_loss_movement_dict[2],
                       'red loss moves': win_loss_movement_dict[3],
                       'blue total shots': b_total_shot_dict,
                       'red total shots': r_total_shot_dict,
                       'blue total moves': b_total_move_dict,
                       'red total moves': r_total_move_dict}, f, indent=2, ensure_ascii=False)

        generate_player_strategy(self.base, self.vid_name)

        return True

    def get_total_info(self):
        with open(f"{self.base}/outputs/{self.vid_name}/game_info.json", 'r', encoding="utf-8") as f:
            frame_dict = json.load(f)
        blue_total_shots = copy.deepcopy(frame_dict['blue win shots'])
        red_total_shots = copy.deepcopy(frame_dict['red win shots'])
        for k in frame_dict['blue win shots'].keys():
            blue_total_shots[k] += frame_dict['blue loss shots'][k]
            red_total_shots[k] += frame_dict['red loss shots'][k]
            if (frame_dict['blue win shots'][k] + frame_dict['blue loss shots'][k]) != 0:
                frame_dict['blue win shots'][k] = np.round(frame_dict['blue win shots'][k] / (
                            frame_dict['blue win shots'][k] + frame_dict['blue loss shots'][k]) * 100, 2)
            else:
                frame_dict['blue win shots'][k] = np.round(frame_dict['blue win shots'][k], 2)
            if (frame_dict['red win shots'][k] + frame_dict['red loss shots'][k]) != 0:
                frame_dict['red win shots'][k] = np.round(frame_dict['red win shots'][k] / (
                            frame_dict['red win shots'][k] + frame_dict['red loss shots'][k]) * 100, 2)
            else:
                frame_dict['red win shots'][k] = np.round(frame_dict['red win shots'][k], 2)

        games = frame_dict['games']
        for game in games[:-1]:
            game['blue red score'][np.argmax(game['blue red score'])] += 1
            game['score count'] += 1.0
        for game in games:
            for i in range(2):
                game['blue red score'][i] = float(game['blue red score'][i])
            game['score count'] = float(game['score count'])

        for k in frame_dict['blue win shots'].keys():
            frame_dict['blue win shots'][k] = float(frame_dict['blue win shots'][k])
            frame_dict['red win shots'][k] = float(frame_dict['red win shots'][k])
        print(blue_total_shots, frame_dict['blue total shots'])
        print(red_total_shots, frame_dict['red total shots'])
        selected_dict = {
            'games': games,
            'blue total shots': count_percentage(frame_dict['blue total shots']),
            'red total shots': count_percentage(frame_dict['red total shots']),
            'blue win shots': count_percentage(frame_dict['blue win shots']),
            'blue loss shots': count_percentage(frame_dict['blue loss shots']),
            'red win shots': count_percentage(frame_dict['red win shots']),
            'red loss shots': count_percentage(frame_dict['red loss shots']),
            'blue win moves': count_percentage(frame_dict['blue win moves']),
            'blue loss moves': count_percentage(frame_dict['blue loss moves']),
            'red win moves': count_percentage(frame_dict['red win moves']),
            'red loss moves': count_percentage(frame_dict['red loss moves']),
            'blue total moves': count_percentage(frame_dict['blue total moves']),
            'red total moves': count_percentage(frame_dict['red total moves']),
        }
        return selected_dict

    def get_respective_score_info(self):
        g1, g2, g3 = [], [], []
        paths = get_path(f"{self.base}/outputs/{self.vid_name}/scores")
        for path in paths:
            num = path.split('/')[-1].split('_')[1]
            if num == '1':
                g1.append(path)
            elif num == '2':
                g2.append(path)
            elif num == '3':
                g3.append(path)
        g1 = sorted(g1, key=lambda i: int(i.split('/')[-1].split('_')[-1]))
        g2 = sorted(g2, key=lambda i: int(i.split('/')[-1].split('_')[-1]))
        g3 = sorted(g3, key=lambda i: int(i.split('/')[-1].split('_')[-1]))
        paths = g1 + g2 + g3
        g1, g2, g3 = [], [], []
        for p in paths:
            num = p.split('/')[-1].split('_')[1]
            with open(f"{p}/info.json", 'r', encoding="utf-8") as f:
                frame_dict = json.load(f)
            selected_dict = {
                'game': float(frame_dict['game']),
                'blue red score': frame_dict['blue red score'],
                'winner': frame_dict['winner'],
                'blue serve first': frame_dict['blue serve first'],
                'shot list': frame_dict['shot list'],
                'blue shot dict': to_float(frame_dict['blue shot dict']),
                'red shot dict': to_float(frame_dict['red shot dict']),
                'move direction list': frame_dict['move direction list'],
                'blue move dict': to_float(count_percentage(frame_dict['blue move dict'])),
                'red move dict': to_float(count_percentage(frame_dict['red move dict'])),
            }
            if num == '1':
                g1.append(selected_dict)
            elif num == '2':
                g2.append(selected_dict)
            elif num == '3':
                g3.append(selected_dict)
        scores_dict = {'g1': g1, 'g2': g2, 'g3': g3} if g3 else {'g1': g1, 'g2': g2}
        return scores_dict

    def get_highlights_info(self):
        with open(f"{self.base}/outputs/{self.vid_name}/highlights.json", 'r', encoding="utf-8") as f:
            frame_dict = json.load(f)
        bwk = frame_dict['blue win key']
        blk = frame_dict['blue loss key']
        rwk = frame_dict['red win key']
        rlk = frame_dict['red loss key']

        return [bwk, blk, rwk, rlk]
