import os
import numpy as np


# shot_list = [('↑ 挑球', 1, 5, False), ('↓ 長球', 5, 13, True), ('↑ 長球', 13, 19, False), ('↓ 長球', 19, 30, True), ('↑ 長球', 30, 37, False), ('↓ 切球', 37, 43, True), ('↑ 挑球', 43, 53, False), ('↓ 殺球', 53, 58, True)]

def counts(dict):
    max_key = ''
    max_len = 0
    for k in dict.keys():
        if len(dict[k]) > max_len:
            max_key = k
            max_len = len(dict[k])

    return max_key, max_len

def counts_three(dict):
    key_len_list = sorted(dict.items(), key=lambda x: x[1])[-3:]
    return key_len_list


def to_float(dict):
    for k in dict.keys():
        dict[k] = float(dict[k])
    return dict


def count_percentage(dict):
    total = 0
    for k in dict.keys():
        total += dict[k]
    for k in dict.keys():
        if total != 0:
            dict[k] = np.round(dict[k] / total * 100, 2)
        else:
            dict[k] = np.round(dict[k], 2)
    return dict

def shot_match(player_shot_list, top):
    # offense_rally = [
    #     [3, 4, 1],
    #     [3, 4, 2],
    #     [2, 4, 1],
    #     [2, 4, 2],
    #     [1, 4, 1],
    #     [1, 4, 2],
    #     [0, 0, 1],
    #     [0, 0, 2],
    #     [5, 4, 1],
    #     [5, 4, 2]
    # ]
    offense_rally = ['341','342','241','242','141','142','001','002','541','542']
    if len(player_shot_list) < 4:
        return None, None
    trim_shots = ''.join(player_shot_list[-4:])
    for pattern in offense_rally:
        if pattern in trim_shots:
            index = trim_shots.index(pattern) + 2
            if index == 3 and top:
                return True, True                 # offensive, top
            elif index == 3 and not top:
                return True, False
            elif index == 2 and top:
                return True, False
            elif index == 2 and not top:
                return True, True
    return False, None


def type_classify(shot_list):
    shot_dict = {
        '長球':'0',
        '殺球':'1',
        '切球':'2',
        '小球':'3',
        '挑球':'4',
        '平球':'5',
        '撲球':'6'
    }
    shots = []
    for shot in shot_list:
        shots.append(shot_dict[shot[0].split(' ')[-1]])
    type, pos = shot_match(shots, shot_list[-1][3])

    return type, pos


def cal_move_direction(c1, c2):
    c1 = np.array(c1)
    c2 = np.array(c2)
    d = c2 - c1

    if d[0] == 0:
        if d[1] == 0:
            return 'NM'
        if d[1] > 0:
            if d[1] == 1:
                return 'LSB'
            else:
                return 'LLB'
        else:
            if d[1] == -1:
                return 'LSF'
            else:
                return 'LLF'
    if d[1] == 0:
        if d[0] > 0:
            if d[0] == 1:
                return 'TSR'
            else:
                return 'TLR'
        else:
            if d[1] == -1:
                return 'TSL'
            else:
                return 'TLL'
    if d[0] > 0 and d[1] > 0:
        if d[0] == 1:
            if d[1] == 1:
                return 'DSBR'
            else:
                return 'DLBR'
        else:
            return 'DLBR'

    if d[0] < 0 and d[1] < 0:
        if d[0] == -1:
            if d[1] == -1:
                return 'DSFL'
            else:
                return 'DLFL'
        else:
            return 'DLFL'

    if d[0] < 0 and d[1] > 0:
        if d[0] == -1:
            if d[1] == 1:
                return 'DSBL'
            else:
                return 'DLBL'
        else:
            return 'DLBL'

    if d[0] > 0 and d[1] < 0:
        if d[0] == 1:
            if d[1] == -1:
                return 'DSFR'
            else:
                return 'DLFR'
        else:
            return 'DLFR'

    return False


def cal_area(jp, p1, p2):
    a = (p1[1] - p2[1]) / (p1[0] - p2[0])
    b = p1[1] - a * p1[0]
    j_hat = (jp[1] - b) / a
    if j_hat > jp[0]:
        return True        # left
    else:
        return False       # right


def zone(test, a):
    if test[1] < a[3][0][1]:              # top player
        if test[1] < a[1][0][1]:          # top 1st row
            if test[0] < a[1][2][0]:      # left side
                left = cal_area(test, a[0][1], a[1][1])
                if left:
                    return [3, 2], True
                else:
                    return [2, 2], True
            else:                         # right side
                left = cal_area(test, a[0][3], a[1][3])
                if left:
                    return [1, 2], True
                else:
                    return [0, 2], True

        elif test[1] < a[2][0][1]:        # top 2nd row
            if test[0] < a[2][2][0]:      # left side
                left = cal_area(test, a[1][1], a[2][1])
                if left:
                    return [3, 1], True
                else:
                    return [2, 1], True
            else:                         # right side
                left = cal_area(test, a[1][3], a[2][3])
                if left:
                    return [1, 1], True
                else:
                    return [0, 1], True
        else:                             # top 3th row
            if test[0] < a[3][2][0]:      # left side
                left = cal_area(test, a[2][1], a[3][1])
                if left:
                    return [3, 0], True
                else:
                    return [2, 0], True
            else:                         # right side
                left = cal_area(test, a[2][3], a[3][3])
                if left:
                    return [1, 0], True
                else:
                    return [0, 0], True
    else:                                 # bot player
        if test[1] < a[4][0][1]:          # bot 1st row
            if test[0] < a[4][2][0]:      # left side
                left = cal_area(test, a[3][1], a[4][1])
                if left:
                    return [0, 0], False
                else:
                    return [1, 0], False
            else:                         # right side
                left = cal_area(test, a[3][3], a[4][3])
                if left:
                    return [2, 0], False
                else:
                    return [3, 0], False

        elif test[1] < a[5][0][1]:        # bot 2nd row
            if test[0] < a[5][2][0]:      # left side
                left = cal_area(test, a[4][1], a[5][1])
                if left:
                    return [0, 1], False
                else:
                    return [1, 1], False
            else:                         # right side
                left = cal_area(test, a[4][3], a[5][3])
                if left:
                    return [2, 1], False
                else:
                    return [3, 1], False
        else:                             # bot 3th row
            if test[0] < a[6][2][0]:      # left side
                left = cal_area(test, a[5][1], a[6][1])
                if left:
                    return [0, 2], False
                else:
                    return [1, 2], False
            else:                         # right side
                left = cal_area(test, a[5][3], a[6][3])
                if left:
                    return [2, 2], False
                else:
                    return [3, 2], False


def correction(court_kp):
    ty = np.round((court_kp[0][1] + court_kp[1][1]) / 2)
    my = (court_kp[2][1] + court_kp[3][1]) / 2
    by = np.round((court_kp[4][1] + court_kp[5][1]) / 2)
    court_kp[0][1] = ty
    court_kp[1][1] = ty
    court_kp[2][1] = my
    court_kp[3][1] = my
    court_kp[4][1] = by
    court_kp[5][1] = by
    return court_kp


def extension(court_kp):
    tlspace = np.array(
        [np.round((court_kp[0][0] - court_kp[2][0]) / 3), np.round((court_kp[2][1] - court_kp[0][1]) / 3)], dtype=int)
    trspace = np.array(
        [np.round((court_kp[3][0] - court_kp[1][0]) / 3), np.round((court_kp[3][1] - court_kp[1][1]) / 3)], dtype=int)
    blspace = np.array(
        [np.round((court_kp[2][0] - court_kp[4][0]) / 3), np.round((court_kp[4][1] - court_kp[2][1]) / 3)], dtype=int)
    brspace = np.array(
        [np.round((court_kp[5][0] - court_kp[3][0]) / 3), np.round((court_kp[5][1] - court_kp[3][1]) / 3)], dtype=int)

    p2 = np.array([court_kp[0][0] - tlspace[0], court_kp[0][1] + tlspace[1]])
    p3 = np.array([court_kp[1][0] + trspace[0], court_kp[1][1] + trspace[1]])
    p4 = np.array([p2[0] - tlspace[0], p2[1] + tlspace[1]])
    p5 = np.array([p3[0] + trspace[0], p3[1] + trspace[1]])

    p8 = np.array([court_kp[2][0] - blspace[0], court_kp[2][1] + blspace[1]])
    p9 = np.array([court_kp[3][0] + brspace[0], court_kp[3][1] + brspace[1]])
    p10 = np.array([p8[0] - blspace[0], p8[1] + blspace[1]])
    p11 = np.array([p9[0] + brspace[0], p9[1] + brspace[1]])

    kp = np.array([court_kp[0], court_kp[1],
                   p2, p3, p4, p5,
                   court_kp[2], court_kp[3],
                   p8, p9, p10, p11,
                   court_kp[4], court_kp[5]], dtype=int)

    ukp = []

    for i in range(0, 13, 2):
        sub2 = np.round((kp[i] + kp[i + 1]) / 2)
        sub1 = np.round((kp[i] + sub2) / 2)
        sub3 = np.round((kp[i + 1] + sub2) / 2)
        ukp.append(kp[i])
        ukp.append(sub1)
        ukp.append(sub2)
        ukp.append(sub3)
        ukp.append(kp[i + 1])
    ukp = np.array(ukp, dtype=int)
    return ukp


def check_dir(path):
    isExit = os.path.exists(path)
    if not isExit:
        os.mkdir(path)


def get_path(base):
    paths = []
    with os.scandir(base) as entries:
        for entry in entries:
            paths.append(base + '/' + entry.name)
            pass
    return paths


def parse_time(FPS, frame_count):
    start_sec = int(frame_count / FPS)

    ssec = start_sec % 60
    smin = start_sec // 60
    if smin >= 60:
        smin = smin % 60
    shr = start_sec // 3600

    if ssec < 10:
        start_sec = '0' + str(start_sec)
    if smin < 10:
        smin = '0' + str(smin)
    if shr < 10:
        shr = '0' + str(shr)

    return f'{shr}:{smin}:{ssec}'


def top_bottom(joint):
    a = joint[0][-1][1] + joint[0][-2][1]
    b = joint[1][-1][1] + joint[1][-2][1]
    if a > b:
        top = 1
        bottom = 0
    else:
        top = 0
        bottom = 1
    return top, bottom

    # a = joint[0][15][1] + joint[0][16][1]
    # b = joint[1][15][1] + joint[1][16][1]
    # if a > b:
    #     top = 1
    #     bottom = 0
    # else:
    #     top = 0
    #     bottom = 1
    # return top, bottom