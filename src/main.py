import os
import json
from utility import get_path
from pipeline import video_resolver
from transformer_utils import coordinateEmbedding, PositionalEncoding, Optimus_Prime
from scene_utils import scene_classifier


def main():
    paths = get_path('../test_videos/inputs')
    # paths = get_path('short_vid')
    vid_paths = []
    for path in paths:
        if path.split('/')[-1].split('.')[-1] == 'mp4':
            vid_paths.append(path)
    for vid_path in vid_paths:
        vid_name = vid_path.split('/')[-1].split('.')[0]
        isExit = os.path.exists(f'../test_videos/outputs/{vid_name}')
        vpr = video_resolver(vid_path, output_base='../test_videos',
                             isExit=isExit)  # output base is where "outputs" dir is
        if not isExit:
            _ = vpr.resolve()
        # boolean, boolean, [blue win key, blue loss key, red win key, red loss key]
        blue_highlight, red_highlight, keys = vpr.get_highlights_info()

        total_info = vpr.get_total_info()

        scores_dict = vpr.get_respective_score_info()
        # print(scores_dict['g1'][0])
        # print(total_info)
        return_info_dict = {
            'players': {'blue': '...', 'red': '...'},
            'highlights info': [blue_highlight, red_highlight, keys],
            'total info': total_info,
            # 'respective scores': scores_dict,
        }
        joint_save_path = f'../test_videos/outputs/{vid_name}/return_info.json'
        try:
            with open(joint_save_path, 'w', encoding="utf-8") as f:
                json.dump(return_info_dict, f, indent=2, ensure_ascii=False)
        except:
            print(return_info_dict)


if __name__ == '__main__':
    main()

