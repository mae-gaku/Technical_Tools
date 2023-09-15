import cv2
from PIL import Image
import tkinter as tk
import tkinter.filedialog


def video2frames(path):
    """
    動画をフレームに分解し、それをPillow画像リストとして返す。
    @ path: 動画のパス
    return: Pillow画像リスト、フレームレート
    """
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not cap.isOpened():
        return None, None
    
    # Pillowの画像リスト
    images = []

    while True:
        ret, frame = cap.read()
        if ret:
            # BGR を RGB に変換
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Pillow画像リストにドンドン追加
            pillow_image = Image.fromarray(rgb_image)
            images.append(pillow_image)

        else:
            return images, fps


def make_gif(images, save_path, fps):
    """
    Pillow画像リストと保存先パスを受け取りGIFファイルを生成。
    @ images   : Pillow画像リスト
    @ save_path: 保存GIFファイル名
    @ fps      : フレームレート
    """
    # 各フレームの表示時間(ミリ秒)
    dur = int(1000.0 / fps) 

    images[0].save(save_path, save_all=True, append_images=images[1:], duration=dur, loop=0)
    print("frame rate = ", fps)
    print("duration = ", dur)


def main():
    # ダイアログを開いてファイルを選択
    root = tk.Tk()
    root.withdraw()
    target_file = tkinter.filedialog.askopenfilename(filetypes=[("動画", "*.mp4")])
    
    # 出力されるGIFの保存先
    save_path = target_file.rsplit(".", 1)[-2] + ".gif"

    # 変換開始
    images, fps = video2frames(target_file)
    make_gif(images, save_path, fps)


if __name__ == "__main__":
    main()