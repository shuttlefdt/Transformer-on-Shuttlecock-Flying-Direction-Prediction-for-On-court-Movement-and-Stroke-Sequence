import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from utility import zone, cal_move_direction, top_bottom


# [top_back, top_mid, top_front, bot_front, bot_mid, bot_back]
def add_result(base, vid_path, shot_list, move_dir_list, court_points):
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print('Error while trying to read video. Please check path again')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    FPS = cap.get(5)
    save_path = f"{base}{vid_path.split('/')[-1].split('.')[0]}_added.mp4"
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (frame_width, frame_height))
    count = 1
    i = 0
    imax = len(shot_list)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            bound = shot_list[i][2]
            if bound >= count:
                text = shot_list[i][0] + ' ' + move_dir_list[i][0]
                cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_im = Image.fromarray(cv2_im)
                draw = ImageDraw.Draw(pil_im)
                font = ImageFont.truetype("../font/msjh.ttc", 50, encoding="utf-8")
                draw.text((900, 50), text, (255, 255, 255), font=font)
                cv2_text_im = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
                out.write(cv2_text_im)
                count += 1
            elif count > bound and i < imax - 1:
                i += 1
                text = shot_list[i][0]
                cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_im = Image.fromarray(cv2_im)
                draw = ImageDraw.Draw(pil_im)
                font = ImageFont.truetype("../font/msjh.ttc", 50, encoding="utf-8")
                draw.text((900, 50), text, (255, 255, 255), font=font)
                cv2_text_im = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
                out.write(cv2_text_im)
                count += 1
            else:
                out.write(frame)
                count += 1
        else:
            break
    return True


def add_result2(out, joint_img_list, shot_list, move_dir_list):
    count = 1
    index = 0
    imax = len(shot_list)
    for i in range(len(joint_img_list)):
        bound = shot_list[index][2]
        if bound >= count:
            text = shot_list[index][0] + ' ' + move_dir_list[index][0]
            cv2_im = cv2.cvtColor(joint_img_list[i], cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im)
            draw = ImageDraw.Draw(pil_im)
            font = ImageFont.truetype("../font/msjh.ttc", 50, encoding="utf-8")
            draw.text((900, 50), text, (255, 255, 255), font=font)
            cv2_text_im = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
            out.write(cv2_text_im)
            count += 1
        elif count > bound and index < imax - 1:
            index += 1
            text = shot_list[index][0]
            cv2_im = cv2.cvtColor(joint_img_list[i], cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im)
            draw = ImageDraw.Draw(pil_im)
            font = ImageFont.truetype("../font/msjh.ttc", 50, encoding="utf-8")
            draw.text((900, 50), text, (255, 255, 255), font=font)
            cv2_text_im = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
            out.write(cv2_text_im)
            count += 1
        else:
            out.write(joint_img_list[i])
            count += 1

    return True


def correct_seq(seq):
    def find_last_index(search_list, search_item):
        return len(search_list) - 1 - search_list[::-1].index(search_item)

    if 2 in seq and 1 in seq:
        front_zero = min(seq.index(2), seq.index(1))
        back_zero = max(find_last_index(seq, 2), find_last_index(seq, 1)) + 1
    elif 2 in seq:
        front_zero = seq.index(2)
        back_zero = find_last_index(seq, 2) + 1
    elif 1 in seq:
        front_zero = seq.index(1)
        back_zero = find_last_index(seq, 1) + 1

    temp = seq[front_zero:back_zero]
    while (True):
        try:
            index = temp.index(0)
            temp[index] = temp[index - 1]
        except:
            break
    f_zero = [0 for i in range(front_zero)]
    b_zero = [0 for i in range(len(seq) - back_zero)]
    return f_zero + temp + b_zero


def check_hit_frame(direction_list, joint_list, court_points, multi_points):
    # joint_list seq len, 2, 12, 2
    multi_points = np.array(multi_points)
    multi_points = np.reshape(multi_points, (7, 5, 2))
    bounds = get_area_bound(court_points)  # 前中後場
    shot_list = []
    move_dir_list = []
    got_first = False
    last_d = 0
    direction_list = correct_seq(direction_list)
    print(direction_list)
    for i in range(len(direction_list)):
        d = direction_list[i]
        if not got_first:
            if d == 1:
                top, bot = top_bottom(joint_list[i])
                first_y = (joint_list[i][top][-1][1] + joint_list[i][top][-2][1]) / 2
                first_x_bot = (joint_list[i][bot][-1][0] + joint_list[i][bot][-2][0]) / 2
                first_y_bot = (joint_list[i][bot][-1][1] + joint_list[i][bot][-2][1]) / 2
                first_coord_bot = np.array([first_x_bot, first_y_bot])
                first_zone = zone(first_coord_bot, multi_points)
                first_i = i
                got_first = True
                last_d = 1
            elif d == 2:
                top, bot = top_bottom(joint_list[i])
                first_y = (joint_list[i][top][-1][1] + joint_list[i][top][-2][1]) / 2
                first_x_top = (joint_list[i][top][-1][0] + joint_list[i][top][-2][0]) / 2
                first_y_top = (joint_list[i][top][-1][1] + joint_list[i][top][-2][1]) / 2
                first_coord_top = np.array([first_x_top, first_y_top])
                first_zone = zone(first_coord_top, multi_points)

                first_i = i
                got_first = True
                last_d = 2
            continue
        if d != last_d and last_d == 1:
            if d == 0:
                d = 2
                change = True
            else:
                change = False
            top, bot = top_bottom(joint_list[i])
            second_y = (joint_list[i][bot][-1][1] + joint_list[i][bot][-2][1]) / 2
            second_x_top = (joint_list[i][top][-1][0] + joint_list[i][top][-2][0]) / 2
            second_y_top = (joint_list[i][top][-1][1] + joint_list[i][top][-2][1]) / 2
            second_x_bot = (joint_list[i][bot][-1][0] + joint_list[i][bot][-2][0]) / 2
            second_y_bot = (joint_list[i][bot][-1][1] + joint_list[i][bot][-2][1]) / 2
            second_coord_fm = np.array([second_x_bot, second_y_bot])
            second_zone = zone(second_coord_fm, multi_points)
            second_i = i

            shot, top_serve = shot_recog(first_y, second_y, d, bounds)
            move_dir = cal_move_direction(first_zone[0], second_zone[0])
            move_dir_list.append((move_dir, False))  # True for top
            first_coord_fm = np.array([second_x_top, second_y_top])
            first_zone = zone(first_coord_fm, multi_points)

            shot_list.append((shot, first_i, second_i, top_serve))

            first_i = second_i
            last_d = d
            if change:
                last_d = 0
            first_y = second_y
        if d != last_d and last_d == 2:
            if d == 0:
                d = 1
                change = True
            else:
                change = False
            top, bot = top_bottom(joint_list[i])
            second_y = (joint_list[i][top][-1][1] + joint_list[i][top][-2][1]) / 2

            second_x_top = (joint_list[i][top][-1][0] + joint_list[i][top][-2][0]) / 2
            second_y_top = (joint_list[i][top][-1][1] + joint_list[i][top][-2][1]) / 2
            second_x_bot = (joint_list[i][bot][-1][0] + joint_list[i][bot][-2][0]) / 2
            second_y_bot = (joint_list[i][bot][-1][1] + joint_list[i][bot][-2][1]) / 2

            second_coord_fm = np.array([second_x_top, second_y_top])
            second_zone = zone(second_coord_fm, multi_points)
            second_i = i

            shot, top_serve = shot_recog(first_y, second_y, d, bounds)
            move_dir = cal_move_direction(first_zone[0], second_zone[0])
            move_dir_list.append((move_dir, True))
            first_coord_fm = np.array([second_x_bot, second_y_bot])
            first_zone = zone(first_coord_fm, multi_points)

            shot_list.append((shot, first_i, second_i, top_serve))
            first_i = second_i
            last_d = d
            if change:
                last_d = 0
            first_y = second_y
    return shot_list, move_dir_list


# [[554, 513], [1366, 495], [462, 708], [1454, 704], [349, 1000], [1568, 999]]
def get_area_bound(court_points):
    top = round((court_points[0][1] + court_points[1][1]) / 2)
    mid = round((court_points[2][1] + court_points[3][1]) / 2)
    bot = round((court_points[4][1] + court_points[5][1]) / 2)
    top_sliced_area = (mid - top) / 10
    bot_sliced_area = (bot - mid) / 10
    top_back = (top, top + 4 * top_sliced_area)
    top_mid = (top + 4 * top_sliced_area, top + 6 * top_sliced_area)
    top_front = (top + 6 * top_sliced_area, mid)

    bot_back = (bot - 4 * bot_sliced_area, bot)
    bot_mid = (bot - 6 * bot_sliced_area, bot - 4 * bot_sliced_area)
    bot_front = (mid, bot - 6 * bot_sliced_area)

    bounds = [top_back, top_mid, top_front, bot_front, bot_mid, bot_back]
    return bounds


def check_pos(coord, bounds, pos):
    if pos == 'top':
        if coord < bounds[0][1]:
            return 'back'
        if coord > bounds[1][0] and coord < bounds[1][1]:
            return 'mid'
        if coord > bounds[2][0]:
            return 'front'
    if pos == 'bot':
        if coord < bounds[3][1]:
            return 'front'
        if coord > bounds[4][0] and coord < bounds[4][1]:
            return 'mid'
        if coord > bounds[5][0]:
            return 'back'
    return None


def shot_recog(first_coord, second_coord, d, bounds):
    bounds = bounds
    if d == 1:  # last d == 2
        pos_bot = check_pos(first_coord, bounds, 'bot')
        pos_top = check_pos(second_coord, bounds, 'top')
        serve = 'bot'
    if d == 2:  # last d == 1
        pos_top = check_pos(first_coord, bounds, 'top')
        pos_bot = check_pos(second_coord, bounds, 'bot')
        serve = 'top'
    print(pos_top, pos_bot, serve)
    shot, top_serve = check_shot(pos_top, pos_bot, serve)
    return shot, top_serve


def check_shot(pos_top, pos_bot, serve):
    if serve == 'top':
        if pos_top == 'front' and pos_bot == 'front':
            return '↓ 小球', True  # True stands for top player's
        if pos_top == 'front' and pos_bot == 'mid':
            return '↓ 撲球', True
        if pos_top == 'front' and pos_bot == 'back':
            return '↓ 挑球', True
        if pos_top == 'mid' and pos_bot == 'front':
            return '↓ 小球', True
        if pos_top == 'mid' and pos_bot == 'mid':
            return '↓ 平球', True
        if pos_top == 'mid' and pos_bot == 'back':
            return '↓ 挑球', True
        if pos_top == 'back' and pos_bot == 'front':
            return '↓ 切球', True
        if pos_top == 'back' and pos_bot == 'mid':
            return '↓ 殺球', True
        if pos_top == 'back' and pos_bot == 'back':
            return '↓ 長球', True
    if serve == 'bot':
        if pos_top == 'front' and pos_bot == 'front':
            return '↑ 小球', False
        if pos_top == 'front' and pos_bot == 'mid':
            return '↑ 小球', False
        if pos_top == 'front' and pos_bot == 'back':
            return '↑ 切球', False
        if pos_top == 'mid' and pos_bot == 'front':
            return '↑ 撲球', False
        if pos_top == 'mid' and pos_bot == 'mid':
            return '↑ 平球', False
        if pos_top == 'mid' and pos_bot == 'back':
            return '↑ 殺球', False
        if pos_top == 'back' and pos_bot == 'front':
            return '↑ 挑球', False
        if pos_top == 'back' and pos_bot == 'mid':
            return '↑ 挑球', False
        if pos_top == 'back' and pos_bot == 'back':
            return '↑ 長球', False

# def get_data(path):
#     inputs = []
#     joint_list = []
#
#     with open(path, 'r') as mp_json:
#         frame_dict = json.load(mp_json)
#
#     for i in range(len(frame_dict['frames'])):
#         temp_x = []
#         if i == 0:
#             temp_f = []
#             former = np.array(frame_dict['frames'][i]['joint'])
#             for p in range(2):
#                 temp_f.append(former[p][5:])
#             temp_f = np.array(temp_f)
#             joint_list.append(frame_dict['frames'][i]['joint'])
#             continue
#         joint_list.append(frame_dict['frames'][i]['joint'])
#         joint = frame_dict['frames'][i]['joint']
#         for p in range(2):
#             temp_x.append(joint[p][5:])  # ignore head part
#         temp_x = np.array(temp_x)
#         dif_x = temp_f - temp_x
#         temp_f = temp_x
#         inputs.append(dif_x)
#
#     inputs = np.array(inputs)
#     joint_list = np.array(joint_list)
#     return inputs, joint_list