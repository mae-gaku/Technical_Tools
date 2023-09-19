import cv2
import tqdm
import glob

def frame_to_movie(dir, bname, fps):
    path_list = sorted(glob.glob(f"{dir}/*.jpg"))

    video_height = 1440 # img.shape[0]
    video_width = 2560 #img.shape[1]

    out = cv2.VideoWriter(f"{dir}/{bname}", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (video_width, video_height))
    
    for path in tqdm(path_list):
        frame = cv2.imread(path)
        out.write(frame)
        # os.remove(path)

    out.release()


if __name__ == '__main__':
    DIR = "/home/gaku/yolov5-face/images"
    BNAME = "face.mp4"
    FPS = 10
    frame_to_movie(DIR, BNAME, FPS)