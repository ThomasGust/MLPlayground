from VideoStyleTransfer1.utils import VideoStylizer
import shutil

video_stylizer = VideoStylizer()


video_stylizer.stitch_video(video_data_dir_path='data_stylized', video_output_name='StylizedDanceTest', fps=24,
                            video_size_x=1028, video_size_y=580, format='mp4')
