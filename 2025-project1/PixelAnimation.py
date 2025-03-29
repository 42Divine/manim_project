from manimlib import *
import cv2
import numpy as np
import os
from noise import pnoise2
def trans_image(scene,img_path 
                ,location=ORIGIN
                ,brightness_threshold = 30
                ,delta_x=ORIGIN):
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
        
      # 创建图片对象并获取其大小
    image = ImageMobject(img_path)
    image.scale(0.7) 
    image.move_to(location) # 这个缩放值要记住
    image_width = image.get_width()
    image_height = image.get_height()
        
        # 2. 确定区域大小
    block_size = max(2, min(width, height) // 50)
        
        # 3. 收集区域点和颜色
    points = []
    colors = []
    image_x=image.get_x()
    image_y=image.get_y()
    for y in range(0, height-block_size+1, block_size):
        for x in range(0, width-block_size+1, block_size):
                # 获取区域
            block = img[y:y+block_size, x:x+block_size]
                
                # 计算区域平均颜色
            color = np.mean(block, axis=(0,1))
                
                # 计算亮度 (R*0.299 + G*0.587 + B*0.114)
            brightness = 0.299*color[0] + 0.587*color[1] + 0.114*color[2]
                
                # 只添加非黑色点
            if brightness > brightness_threshold:
                    # 转换坐标以匹配图片大小
                manim_x = (x - width/2) * image_width/width+image_x
                manim_y = -(y - height/2) * image_height/height+image_y
                    
                points.append([manim_x, manim_y, 0])
                colors.append(color/255.0)  # 归一化颜色值
    

        # 4. 创建粒子
    len_points=len(points)
    delta_1=delta_x[0]
    delta_2=delta_x[1]
    particles = VGroup(*[
        Dot(
            radius=0.03,
            color=rgb_to_color(colors[i]),
        ).move_to([
            # 使用极坐标生成星团效果
            delta_1 + points[(len_points//2-i-1)%len_points][0] + image_width/2 * np.cos(np.random.uniform(0, 2 * np.pi)),
            delta_2 + points[(len_points//2-i-1)%len_points][1] + image_height/2 * np.cos(np.random.uniform(0, 2 * np.pi)),
            0
        ])
        for i in range(len(points))
    ])
    particles_fadein=[FadeInFromPoint(particle,point=particle.get_center()+UP*np.random.uniform(-1, 1)+RIGHT*np.random.uniform(-1, 1)) for particle in particles]
        # 5. 动画序列
    scene.play(*particles_fadein,run_time=1.5,rate_func=smooth)    
        # 创建动画
    animations = []
    for i in range(len(particles)):
        delay = random.uniform(0, 0.3)
        # 创建一个随机的控制点，使粒子形成弧形路径
        #control_point1 = np.array([
        #    points[i][0] + random.uniform(-2, 2),  # x坐标随机偏移
        #    points[i][1] + random.uniform(-1, 1),  # y坐标随机偏移
        #    0
        #])
        #control_point2 = np.array([
        #    points[i][0] + random.uniform(-2, 2),
        #    points[i][1] + random.uniform(-1, 1),
        #    0
        #])
        
        #path = CubicBezier(
        #    particles[i].get_center(),
        #    control_point1,
        #    control_point2,
        #    points[i]
        #)
        start_pos = particles[i].get_center()
        end_pos = points[i]
            
            # 计算旋转轴：中垂线
        mid_point = (start_pos + end_pos) / 2
        direction = end_pos - start_pos
        normal = np.array([-direction[1], direction[0], 0]) 
        anim = Rotating(particles[i],about_point=mid_point,axis=normal,angle=180*DEGREES)
        animations.append(anim)
        
    # 播放动画
    scene.play(
        *animations,
        run_time=2,
        rate_func=smooth
    )
        # 淡出淡入
    scene.play(
            *[
                UpdateFromAlphaFunc(
                    particle,
                    lambda m, a: m.set_opacity(1 - a)
                )
                for particle in particles
            ],
            FadeIn(image),
            run_time=0.8
        )
    return image
class PixelAnimation1(Scene):
    def construct(self):
        # 1. 读取图片并创建图片对象
        image1=trans_image(self,"images/126616.png",location=LEFT*3+2*UP,brightness_threshold=30,delta_x=DOWN)
        image2=trans_image(self,"images/10.png",location=2.9*LEFT+1.7*DOWN,brightness_threshold=10)
        self.wait(1)

class PixelAnimation2(Scene):
    def construct(self):
        # 1. 读取图片并创建图片对象
        image1=trans_image(self,"images/126616.png",location=LEFT*3+2*UP,brightness_threshold=30,delta_x=DOWN)
        image2=trans_image(self,"images/10.png",location=2.9*LEFT+1.7*DOWN,brightness_threshold=10)
        self.wait(1)
if __name__ == "__main__":
    os.system("manimgl {} PixelAnimation2 -c black --uhd -w".format(__file__))