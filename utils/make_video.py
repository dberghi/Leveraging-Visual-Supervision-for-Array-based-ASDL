#!/usr/bin/python

import csv, os, cv2, subprocess, shutil
import numpy as np
import core.config as conf


base_path = conf.input['project_path']
frames = base_path + 'frames/' # it will temporally create a frames/ folder

sequence = 'interactive4_t3-cam06.mp4'
info = 'gcc_GT_GT'
lr = 0.0005

# ========== ========== ========== ==========
# # GENERATE VIDEO WITH VERTICAL LINE
# ========== ========== ========== ==========

def generate_video(video_path, csv_path):
    frame_array = []
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        size = (W, H)
        cap.release()

        for row in reader:
            if row[0] == sequence[:-4]:

                if float(row[3]) > 0.5:
                    time_frame = int(np.round(float(row[1])*fps)) +1
                    print(base_path + 'frames/%05d.jpg' %time_frame)

                    frame = cv2.imread(base_path + 'frames/%05d.jpg' %time_frame)
                    x = int(float(row[2])*W)
                    
                    frame = cv2.line(frame, (x, 0), (x, H), (0, 255, 0), 5)
                    cv2.imwrite(base_path + 'frames/%05d.jpg' %(time_frame), frame)


        

def main():
    video_path = base_path + 'data/TragicTalkers/videos/test/' + sequence[:-10] + '/' + sequence
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # MAKE FRAMES DIRECTORY
    if os.path.exists(frames):
        shutil.rmtree(frames)
    os.mkdir(frames)

    # EXTRACT VIDEO FRAMES
    command = ("ffmpeg -i %s -threads 10 -deinterlace -q:v 1 -vf fps=%d %s%s" % (video_path, fps, frames, '%05d.jpg'))
    subprocess.call(command, shell=True, stdout=None)


    generate_video(video_path, base_path + 'output/forward/%s/%f/test_forward.csv' %(info,lr))


    # GENERATE SILENT VIDEO
    command = ('ffmpeg -r %d -start_number 1 -i %s"%%05d".jpg -c:v libx264 -vf fps=%d -pix_fmt yuv420p %ssilent.mp4' % (
    fps, frames, fps, base_path))
    subprocess.call(command, shell=True, stdout=None)

    # EXTRACT AUDIO
    command = ('ffmpeg -i %s %saudio.mp3' % (video_path, base_path))
    subprocess.call(command, shell=True, stdout=None)

    # MERGE VIDEO AND AUDIO
    command = ('ffmpeg -i %ssilent.mp4 -i %saudio.mp3 %scheck_%s' % (base_path, base_path, base_path, sequence))
    subprocess.call(command, shell=True, stdout=None)

    # DELETE silent.mnp4 and audio.mp3 and frames/
    if os.path.exists("%ssilent.mp4" % base_path):
        os.remove("%ssilent.mp4" % base_path)
    if os.path.exists("%saudio.mp3" % base_path):
        os.remove("%saudio.mp3" % base_path)
    if os.path.exists(frames):
        shutil.rmtree(frames)


if __name__ == "__main__":
    main()