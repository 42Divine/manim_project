from manimlib import *
import random
import numpy as np
import os
class AnimatedStreamLinesbywjj(VGroup):
    def __init__(
        self,
        stream_lines,
        lag_range: float = 4,
        rate_multiple: float = 1.0,
        line_anim_config: dict = dict(
        ),shift = 0.5 * RIGHT + 0.5 * UP,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.stream_lines = stream_lines
        self.shift = shift
        for line in stream_lines:
            line.anim = FadeIn(
                line,
                shift=self.shift,
                run_time=line.virtual_time / rate_multiple,
                **line_anim_config,
            )
            line.anim.begin()
            line.time = -lag_range * np.random.random()
            self.add(line.anim.mobject)

        self.add_updater(lambda m, dt: m.update(dt))

    def update(self, dt: float) -> None:
        stream_lines = self.stream_lines
        for line in stream_lines:
            line.time += dt
            adjusted_time = max(line.time, 0) % line.anim.run_time
            line.anim.update(adjusted_time / line.anim.run_time)
    def streamlinefinish(self):
        self.clear_updaters()
        stream_lines = self.stream_lines
        for line in stream_lines:
            line.anim.finish()
class ContinuousFlow(Scene):
    def construct(self):
        pass
class qq_temp1(Scene):
    def construct(self):
        # 配置参数
        frame = self.camera.frame
        frame.reorient(0,70,0,(0,0,1),height=6)
        frame.rotate(angle=10*DEG,axis=IN)
        wall_width = 3.0       # 顶部横梁宽度
        wall_height = 2.0      # 侧墙高度
        door_width = 0.8
        door_height = 1.2
        wall_thickness = 0.2   # 墙体厚度
              # 门宽度
        x_intersect = -door_width / 2
    
    # 计算交点y坐标的分子和分母
        numerator = door_height * (3 * door_width + wall_width)
        denominator = wall_width - door_width
    
    # 计算y坐标（含异常处理防止除零错误）
        if denominator == 0:
            raise ValueError("Wall width and door width cannot be equal (lines are parallel)")
    
        y_intersect = (numerator / denominator) - (wall_height / 2)
        Intersection_dot = np.array([x_intersect,y_intersect,0])
        dot = Dot().move_to(Intersection_dot)
        # 矩形门
        wall_points = [
            [-wall_width/2, wall_height/2, 0],          # 左上
            [wall_width/2, wall_height/2, 0],            # 右上
            [wall_width/2 , -wall_height/2, 0], # 右下（内缩形成右侧立柱）
            [door_width/2, -wall_height/2, 0], # 左下（内缩形成左侧立柱）
            [door_width/2 , -wall_height/2+door_height, 0],
            [-door_width/2 , -wall_height/2+door_height, 0],
            [-door_width/2, -wall_height/2, 0],
            [-wall_width/2 , -wall_height/2, 0],
        ]
        # 创建凹形墙（单Polygon实现）
        wall = Polygon(*wall_points, color=GREEN_A, fill_opacity=1,stroke_color=GREEN,
            stroke_opacity=0.9,
            stroke_width=8,)
        # 创建门（与缺口匹配）
        door = Polygon([door_width/2, -wall_height/2, 0], 
            [door_width/2 , -wall_height/2+door_height, 0],
            [-door_width/2 , -wall_height/2+door_height, 0],
            [-door_width/2, -wall_height/2, 0],
            color=BLUE_A,
            fill_opacity=1
        ) # 精确对齐顶部
        recline1 = Line([door_width/2, -wall_height/2, 0], 
            [door_width/2 , -wall_height/2+door_height, 0],).set_color(YELLOW_A)
        recline2 = Line([-door_width/2 , -wall_height/2+door_height, 0],
            [-door_width/2, -wall_height/2, 0],).set_color(YELLOW_A)
        rectex = VGroup(Tex("A").next_to(recline1.get_start(),UR,buff=0.1).set_color(RED_A),
                        Tex("B").next_to(recline1.get_end(),UR,buff=0.1).set_color(RED_A),
                        Tex("C").next_to(recline2.get_start(),UL,buff=0.1).set_color(RED_A),
                        Tex("D").next_to(recline2.get_end(),UL,buff=0.1).set_color(RED_A),
                        Tex(r"\alpha").next_to(np.array([-wall_width/2, wall_height/2, 0]),DR,buff=0.1))
        rectex[0].add_updater(lambda mob:mob.next_to(recline1.get_start(),IN,buff=0.1))
        rectex[1].add_updater(lambda mob:mob.next_to(recline1.get_end(),OUT+RIGHT+DOWN,buff=0.2))
        rectex[2].add_updater(lambda mob:mob.next_to(recline2.get_start(),OUT+LEFT,buff=0.1))
        rectex[3].add_updater(lambda mob:mob.next_to(recline2.get_end(),IN+LEFT,buff=0.1))
        rectangle_door = VGroup(wall,door,recline1,recline2,rectex)
        # 设置旋转轴心（门顶部中点）
        rectangle_door.shift(wall_height/2*UP)
        rectangle_door.rotate(axis=RIGHT,angle=PI/2,about_point=ORIGIN)
        ###梯形门
        
        dot.shift(wall_height/2*UP)
        dot.rotate(axis=RIGHT,angle=PI/2,about_point=ORIGIN)
        door_vg = VGroup(door,recline1,recline2)

        numberplane = NumberPlane(y_range=(-8,8,1),background_line_style = dict(
            stroke_color=BLUE_B,
            stroke_width=2,
            stroke_opacity=0.5,
        ),faded_line_ratio=2)
        self.add(numberplane,rectangle_door)
        point1 = door.get_left()
        self.play(
            Rotate(door_vg,angle=PI/2,about_point=point1,axis=IN,rate_func=smooth,run_time=2),
            )
        k=self.get_time()
        door_vg.add_updater(lambda mob:mob.rotate(angle=1.5*np.cos(self.get_time()-k)*DEG,about_point=point1,axis=IN))
        
        self.wait(11)

        door_vg.clear_updaters()
        DashedLine1_left = DashedLine(start=recline1.get_end(),end=np.array([recline1.get_start()[0],recline1.get_start()[1],dot.get_center()[2]])).set_color("#FF0000")
        DashedLine2_left = DashedLine(start=recline2.get_start(),end=np.array([recline2.get_start()[0],recline2.get_start()[1],dot.get_center()[2]])).set_color("#FF0000")
        
        self.play(
                  Write(DashedLine1_left),Write(DashedLine2_left)
                  ,run_time=2)

        door_vg.add(DashedLine1_left,DashedLine2_left)

        k1=self.get_time()
        
        door_vg.add_updater(lambda mob:mob.rotate(angle=1.3*np.sin(self.get_time()-k1)*DEG,about_point=point1,axis=IN))
        self.wait(16)
class qq_temp2(Scene):
    def construct(self) -> None:
        # 配置参数
        frame = self.camera.frame
        frame.reorient(0,70,0,(0,0,1),height=6)
        frame.rotate(angle=10*DEG,axis=IN)
        wall_width = 3.0       # 顶部横梁宽度
        wall_height = 2.0      # 侧墙高度
        door_width = 0.8
        door_height = 1.2
        wall_thickness = 0.2   # 墙体厚度
              # 门宽度
        x_intersect = -door_width / 2
    
    # 计算交点y坐标的分子和分母
        numerator = door_height * (3 * door_width + wall_width)
        denominator = wall_width - door_width
    
    # 计算y坐标（含异常处理防止除零错误）
        if denominator == 0:
            raise ValueError("Wall width and door width cannot be equal (lines are parallel)")
    
        y_intersect = (numerator / denominator) - (wall_height / 2)
        Intersection_dot = np.array([x_intersect,y_intersect,0])
        dot = Dot().move_to(Intersection_dot)
        # 矩形门
        wall_points = [
            [-wall_width/2, wall_height/2, 0],          # 左上
            [wall_width/2, wall_height/2, 0],            # 右上
            [wall_width/2 , -wall_height/2, 0], # 右下（内缩形成右侧立柱）
            [door_width/2, -wall_height/2, 0], # 左下（内缩形成左侧立柱）
            [door_width/2 , -wall_height/2+door_height, 0],
            [-door_width/2 , -wall_height/2+door_height, 0],
            [-door_width/2, -wall_height/2, 0],
            [-wall_width/2 , -wall_height/2, 0],
        ]
        # 创建凹形墙（单Polygon实现）
        ###梯形门
        wall1 = Polygon(*wall_points, color=GREEN_A, fill_opacity=1,stroke_color=GREEN,
            stroke_opacity=0.9,
            stroke_width=8,)
        # 创建门（与缺口匹配）
        door1 = Polygon([wall_width/4+door_width/4, -wall_height/2, 0], 
            [door_width/2 , -wall_height/2+door_height, 0],
            [-door_width/2 , -wall_height/2+door_height, 0],
            [-door_width/2, -wall_height/2, 0],
            color=BLUE_A,
            fill_opacity=1
        ) # 精确对齐顶部
        trapeline1 = Line([wall_width/4+door_width/4, -wall_height/2, 0], 
            [door_width/2 , -wall_height/2+door_height, 0],).set_color(YELLOW_A)
        trapeline2 = Line([-door_width/2 , -wall_height/2+door_height, 0],
            [-door_width/2, -wall_height/2, 0],).set_color(YELLOW_A)
        trapetex1 = VGroup(Tex("A").next_to(trapeline1.get_start(),UR,buff=0.1).set_color(RED_A),
                        Tex("B").next_to(trapeline1.get_end(),UR,buff=0.1).set_color(RED_A),
                        Tex("C").next_to(trapeline2.get_start(),UL,buff=0.1).set_color(RED_A),
                        Tex("D").next_to(trapeline2.get_end(),UL,buff=0.1).set_color(RED_A),
                        Tex(r"\alpha").next_to(np.array([-wall_width/2, wall_height/2, 0]),DR,buff=0.1))
        
        trapetex1[0].add_updater(lambda mob:mob.next_to(trapeline1.get_start(),IN,buff=0.1))
        trapetex1[1].add_updater(lambda mob:mob.next_to(trapeline1.get_end(),OUT+RIGHT+DOWN,buff=0.2))
        trapetex1[2].add_updater(lambda mob:mob.next_to(trapeline2.get_start(),UL,buff=0.1))
        trapetex1[3].add_updater(lambda mob:mob.next_to(trapeline2.get_end(),UL,buff=0.1))
        trapetangle_door = VGroup(wall1,door1,trapeline1,trapeline2,trapetex1)
        # 设置旋转轴心（门顶部中点）
        trapetangle_door.shift(wall_height/2*UP)
        trapetangle_door.rotate(axis=RIGHT,angle=PI/2,about_point=ORIGIN)
        dot.shift(wall_height/2*UP)
        dot.rotate(axis=RIGHT,angle=PI/2,about_point=ORIGIN)

        door_vg1 = VGroup(door1,trapeline1,trapeline2)
        numberplane = NumberPlane(y_range=(-8,8,1),background_line_style = dict(
            stroke_color=BLUE_B,
            stroke_width=2,
            stroke_opacity=0.5,
        ),faded_line_ratio=2)
        self.add(numberplane,trapetangle_door)
        point2 = door1.get_left()
        self.play(
            Rotate(door_vg1,angle=PI/2,about_point=point2,axis=IN,rate_func=smooth,run_time=2),
            )
        k=self.get_time()
        
        door_vg1.add_updater(lambda mob:mob.rotate(angle=1.5*np.cos(self.get_time()-k)*DEG,about_point=point2,axis=IN))
        self.wait(11)
        door_vg1.clear_updaters()

        DashedLine1 = DashedLine(start=trapeline1.get_end(),end=dot.get_center()).set_color("#FF0000")
        DashedLine2 = DashedLine(start=trapeline2.get_start(),end=dot.get_center()).set_color("#FF0000")
        self.play(Write(DashedLine1),Write(DashedLine2)
                  ,run_time=2)
        door_vg1.add(DashedLine1,DashedLine2)
        k1=self.get_time()
        door_vg1.add_updater(lambda mob:mob.rotate(angle=1.3*np.sin(self.get_time()-k1)*DEG,about_point=point2,axis=IN))
        self.wait(16)
if __name__ == "__main__":
    
    os.system("manimgl {} qq_temp1 -c #f5dbec --uhd -w".format(__file__))