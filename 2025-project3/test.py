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
        stream_lines = self.stream_lines
        for line in stream_lines:
            line.anim = None
class decimal_number_tex(Tex):
    def __init__(
        self,
        number: float,
        font_size: int = 48,
        alignment: str = R"\centering",
        template: str = "",
        additional_preamble: str = "",
        tex_to_color_map: dict = dict(),
        t2c: dict = dict(),
        use_labelled_svg: bool = True,
        num_decimal_places: int = 2,
        **kwargs
    ):
        self.num_decimal_places = num_decimal_places
        # Combine multi-string arg, but mark them to isolate
        rounded_num = np.round(number, self.num_decimal_places)

        tex_string = str(rounded_num)

        # Prevent from passing an empty string.
        if not tex_string.strip():
            tex_string = R"\\"

        self.font_size = font_size
        self.tex_string = tex_string
        self.alignment = alignment
        self.template = template
        self.additional_preamble = additional_preamble
        self.tex_to_color_map = dict(**t2c, **tex_to_color_map)

        super().__init__(
            tex_string,
            use_labelled_svg=use_labelled_svg,
            **kwargs
        )
def get_special_dot(
    color=YELLOW,
    radius=0.05,
    glow_radius_multiple=3,
    glow_factor=1.5
):
    return Group(
        TrueDot(radius=radius).make_3d(),
        GlowDot(radius=radius * glow_radius_multiple, glow_factor=glow_factor)
    ).set_color(color)
class introduce(Scene):
    def construct(self):
        frame = self.camera.frame
        vector = Arrow(stroke_width=2,end=RIGHT+UP,buff=0,thickness=4,fill_opacity=0.8,fill_color=BLUE_A)
        self.wait(1)
        self.play(Write(vector))
        self.wait()
        vector1 = vector.copy()
        vector2 = vector.copy()
        self.play(AnimationGroup(Rotate(vector1,120*DEG,about_edge=DL)
                                 ,Rotate(vector2,240*DEG,about_edge=DL)
                                 ,run_time=1.2,lag_ratio=0.2,rate_func=rush_into
                                 ))
        self.wait()
        self.play(AnimationGroup(
            vector.animate.put_start_and_end_on(start=DOWN+1*LEFT,end=1*RIGHT).set_color(GREEN_A),
            vector1.animate.put_start_and_end_on(start=DOWN+5*LEFT,end=3*LEFT).set_color(BLUE_A),
            vector2.animate.put_start_and_end_on(start=DOWN+3*RIGHT,end=5*RIGHT).set_color(RED_A),
            run_time=2,lag_ratio=0.2,rate_func=rush_into
        ))
        vector_copy=vector.copy()
        vector1_copy=vector1.copy()
        vector2_copy = vector2.copy()
        Text_vg = VGroup(Text("数量积",font="KaiTi").set_color(GREEN_A).next_to(vector,UP),
                         Text("内积",font="KaiTi").set_color(BLUE_A).next_to(vector1,UP),
                         Text("点积",font="KaiTi").set_color(RED_A).next_to(vector2,UP),
                         )
        self.wait()
        self.play(AnimationGroup(
            Rotate(vector_copy,-40*DEG,about_edge=DL),
            Write(Text_vg[0]),
            run_time=2,lag_ratio=0.2,rate_func=rush_into
        ))
        self.wait()
        self.play(AnimationGroup(
            Rotate(vector1_copy,-60*DEG,about_edge=DL),
            Write(Text_vg[1]),
            run_time=2,lag_ratio=0.2,rate_func=rush_into
        ))
        self.wait()
        self.play(AnimationGroup(
            Rotate(vector2_copy,-60*DEG,about_edge=DL),
            Write(Text_vg[2]),
            run_time=2,lag_ratio=0.2,rate_func=rush_into
        ))
        self.wait()
        #定义汇总的表达式a*b
        eq = Tex(r"\vec{a}\cdot \vec{b}",font_size=72).move_to(3*UP).set_color_by_gradient("#f7cee7","#f5dbec","#e9cefa")
        self.play(AnimationGroup(
            AnimationGroup(
                ShowCreationThenDestruction(Line(start=Text_vg[0].get_center(),end=eq.get_center()).set_color(Text_vg[0].get_color())),
                ShowCreationThenDestruction(Line(start=Text_vg[1].get_center(),end=eq.get_center()).set_color(Text_vg[1].get_color())),
                ShowCreationThenDestruction(Line(start=Text_vg[2].get_center(),end=eq.get_center()).set_color(Text_vg[2].get_color())),
            ),
            Write(eq,run_time=1.5),lag_ratio=0.3
        ))
        self.wait()
        ####替换点
        epsilon = 1e-5
        evolution_time = 30
        n_points = 10
        states = [
            [10, 10, 10 + n * epsilon]
            for n in range(n_points)
        ]
        FadeOut
        colors = color_gradient([BLUE_E, BLUE_A], len(states))
        dot = get_special_dot(color=colors[-1], radius=0.25/3).move_to(eq[2])
        self.play(FadeIn(dot),FadeOut(eq[2]))
        self.wait()
        self.remove(eq[2])
        self.play(AnimationGroup(
            AnimationGroup(frame.animate.reorient(0,70,0,dot.get_center()),run_time=4),
            AnimationGroup(
                FadeOut(eq,scale=0.1),
                FadeOutbyrotate(VGroup(vector1,vector1_copy,Text_vg[1]),about_point=vector1.get_center(),angle=-TAU),
                FadeOutbyrotate(VGroup(vector,vector_copy,Text_vg[0]),about_point=vector.get_center(),angle=TAU),
                FadeOutbyrotate(VGroup(vector2,vector2_copy,Text_vg[2]),about_point=vector2.get_center(),angle=-TAU),run_time=2
            ),lag_ratio=0.2
        ))
        self.wait()
class scene1(Scene):
    def construct(self) -> None:
        epsilon = 1e-5
        evolution_time = 30
        n_points = 10
        states = [
            [10, 10, 10 + n * epsilon]
            for n in range(n_points)
        ]
        colors = color_gradient([BLUE_E, BLUE_A], len(states))
        dot = get_special_dot(color=colors[-1], radius=0.25/3)
        dot1 = get_special_dot(color=GREEN_A, radius=0.25/3)
        self.add(dot)
        self.wait()
        self.play(dot.animate.shift(LEFT),
                  dot1.animate.shift(RIGHT),
                  )
        self.wait()
        signal = Group(Tex(r"\times",fill_color="#f7cee7").move_to(dot.get_center()+dot1.get_center()),
                        Tex(r"=",fill_color="#f5dbec").move_to(2*RIGHT),
                        Tex(r"n",fill_color="#e9cefa").move_to(3*RIGHT),
                        get_special_dot(color="#e9cefa", radius=0.25/3).move_to(3*RIGHT),
                        Tex(r"?",fill_color="#e9cefa").move_to(3*RIGHT))
        self.play(FadeIn(signal[0],scale=0.1))
        self.wait()
        self.play(FadeIn(signal[1],scale=0.1))
        self.wait()
        self.play(FadeIn(signal[2]))
        self.wait()
        self.play(FadeOut(signal[2],shift=DOWN),FadeIn(signal[3],shift=DOWN),rate_func=rush_from)
        self.wait()
        self.play(FadeOut(signal[3],shift=DOWN),FadeIn(signal[4],shift=DOWN),rate_func=rush_from)
        self.wait()
        self.play(AnimationGroup(FadeOutbyrotate(signal[0],about_edge=ORIGIN,shift=LEFT+3*UP),
                                 FadeOutbyrotate(signal[1],about_edge=ORIGIN,shift=RIGHT+3*UP),
                                 FadeOutbyrotate(signal[4],about_edge=ORIGIN,shift=LEFT+3*DOWN),
                                 lag_ratio=0.2,run_time=3))
        self.remove(signal)
        self.wait()
        ###考虑一维度情形
        k = Line(8*LEFT,8*RIGHT,stroke_width=8).set_color([BLUE_A,TEAL_A,GREEN_A])
        self.play(Write(k))
        self.wait()
        numberline = NumberLine(include_numbers = True)
        numberline.numbers.shift(0.2*DOWN)
        self.play(FadeIn(numberline.numbers,shift=UP),FadeIn(numberline.ticks,shift=DOWN))
        self.wait()
        vec1 = Arrow(thickness=4,stroke_width=2,start=dot.get_center()+2*UP,end=dot.get_center()+0.05*UP,fill_color=dot[0].get_color(),buff=0,fill_opacity=0.8)
        vec2 = Arrow(thickness=4,stroke_width=2,start=dot1.get_center()+2*UP,end=dot1.get_center()+0.05*UP,fill_color=dot1[0].get_color(),buff=0,fill_opacity=0.8)
        decimal_number1 = decimal_number_tex(dot.get_center()[0],num_decimal_places=1,fill_color=dot[0].get_color()).next_to(vec1,UP)
        decimal_number2 = decimal_number_tex(dot1.get_center()[0],num_decimal_places=1,fill_color=dot1[0].get_color()).next_to(vec2,UP)
        self.play(FadeIn(vec1,shift=LEFT),
                  FadeIn(vec2,shift=RIGHT),
                  FadeIn(decimal_number1,shift=DOWN),
                  FadeIn(decimal_number2,shift=DOWN),
                  )
        vec1.add_updater(lambda mob:mob.become(Arrow(thickness=4,stroke_width=2,start=dot.get_center()+2*UP,end=dot.get_center()+0.05*UP,fill_color=dot[0].get_color(),buff=0,fill_opacity=0.8)))
        vec2.add_updater(lambda mob:mob.become(Arrow(thickness=4,stroke_width=2,start=dot1.get_center()+2*UP,end=dot1.get_center()+0.05*UP,fill_color=dot1[0].get_color(),buff=0,fill_opacity=0.8)))
        decimal_number1.add_updater(lambda mob:mob.become(decimal_number_tex(dot.get_center()[0],num_decimal_places=1,fill_color=dot[0].get_color()).next_to(vec1,UP)))
        decimal_number2.add_updater(lambda mob:mob.become(decimal_number_tex(dot1.get_center()[0],num_decimal_places=1,fill_color=dot1[0].get_color()).next_to(vec2,UP)))
        self.wait()
        self.play(dot1.animate.shift(3*LEFT),run_time=3)
        self.play(dot.animate.shift(2*RIGHT),run_time=2)
        self.play(dot1.animate.shift(6*RIGHT),run_time=2)
        self.wait()
        vec1.clear_updaters()
        vec2.clear_updaters()
        decimal_number1.clear_updaters()
        decimal_number2.clear_updaters()
        ######展示自然的乘法
        dot_copy = dot.copy()
        dot1_copy = dot1.copy()
        vg = Group(dot_copy,Tex(r"\times",fill_color="#f7cee7"),dot1_copy)
        vg.arrange(LEFT)
        vg.shift(4*LEFT+2*UP)
        self.play(FadeIn(vg[1],shift=DOWN,scale=0.2),
                  FadeIn(vg[0],shift=LEFT,scale=0.2),
                  FadeIn(vg[2],shift=RIGHT,scale=0.2),
                  )
        self.wait()
        decimal_number1_copy = decimal_number1.copy()
        decimal_number2_copy = decimal_number2.copy()
        vg1 = VGroup(decimal_number1_copy,Tex(r"\times",fill_color="#f7cee7"),decimal_number2_copy)
        vg1[1].next_to(vg[1],DOWN,buff=1)
        vg1[0].next_to(vg1[1],RIGHT)
        vg1[2].next_to(vg1[1],LEFT)
        self.play(TransformFromCopy(VGroup(decimal_number1,decimal_number2),vg1))
        self.wait()
        equality = Tex("=").set_color(YELLOW_A)
        equality.rotate(PI/2)
        equality.move_to((vg[1].get_center()+vg1[1].get_center())/2)
        self.play(FadeIn(equality,shift=RIGHT))
        self.wait()
        coorid = VGroup(Tex(r"x_1").set_color(GREEN_A).next_to(vg1[1],LEFT),
                        Tex(r"x_2").set_color(BLUE_A).next_to(vg1[1],RIGHT),
                        )
        self.play(FlashAround(Group(vg,vg1)),run_time=2)
        self.wait()
        self.play(
        AnimationGroup(
                FadeOut(vg1[2],shift=DOWN),
                FadeIn(coorid[0],shift=DOWN),
        ),
        AnimationGroup(
                FadeOut(vg1[0],shift=DOWN),
                FadeIn(coorid[1],shift=DOWN),
        ),lag_ratio=0.8
                  )
        self.wait()
        self.play(FlashAround(coorid,color=BLUE_A),run_time=1.5)
        self.wait()
        self.play(AnimationGroup(
            AnimationGroup(FadeOut(numberline.ticks,shift=UP),
                  FadeOut(numberline.numbers,shift=DOWN),run_time=2),
            AnimationGroup(
                FadeOut(vec1,shift=LEFT),
                FadeOut(vec2,shift=RIGHT),
                FadeOut(decimal_number1,shift=LEFT),
                FadeOut(decimal_number2,shift=RIGHT)
            ),
            AnimationGroup(
                FadeOut(vg,shift=LEFT),
                FadeOut(vg1[1],shift=RIGHT),
                FadeOut(coorid,shift=LEFT),
                FadeOut(equality,shift=RIGHT),run_time=4
            ),lag_ratio=0.3
        ))
        self.wait()
class scene2(Scene):
    def construct(self) -> None:
        frame = self.camera.frame
        epsilon = 1e-5
        evolution_time = 30
        n_points = 10
        states = [
            [10, 10, 10 + n * epsilon]
            for n in range(n_points)
        ]
        colors = color_gradient([BLUE_E, BLUE_A], len(states))
        dot = get_special_dot(color=colors[-1], radius=0.25/3)
        dot1 = get_special_dot(color=GREEN_A, radius=0.25/3)
        dot.shift(RIGHT)
        dot1.shift(4*RIGHT)
        line_dot1 = dot.copy()
        line_dot2 = dot1.copy()
        k = Line(8*LEFT,8*RIGHT,stroke_width=8).set_color([BLUE_A,TEAL_A,GREEN_A])
        self.add(dot,k,dot1)
        self.wait()
        ky = k.copy().rotate(PI/2)
        self.play(AnimationGroup(frame.animate.reorient(0,45,0,(0.5,1,0)),
                                 FadeIn(ky,shift=OUT,scale=2),
                                 AnimationGroup(dot.animate.move_to(np.array([1,2,0])),
                                                dot1.animate.move_to(np.array([-1,1,0])),
                                                ),lag_ratio=0.2
                                 ))
        self.wait()
        numberplane = NumberPlane(y_range=(-8,8,1),background_line_style = dict(
            stroke_color=BLUE_B,
            stroke_width=2,
            stroke_opacity=0.5,
        ),faded_line_ratio=2)
        self.play(
            AnimationGroup(
            frame.animate.reorient(0,0,0,(0,0,0)),run_time=4
        ),AnimationGroup(
            FadeOut(VGroup(k,ky),shift=OUT),
            FadeIn(numberplane,shift=OUT),run_time=2
        ),lag_ratio=0.4
        )
        self.remove(k,ky)
        self.add(numberplane)
        #####在这里添加二维点的说明
        factor = np.array([0.35,0.7,0])
        tex_dot = Tex(r"(x_1,y_1)").set_color(dot[0].get_color()).move_to(dot.get_center()+factor*dot.get_center()/np.linalg.norm(dot.get_center()))
        tex_dot1 = Tex(r"(x_2,y_2)").set_color(dot1[0].get_color()).move_to(dot1.get_center()+factor*dot1.get_center()/np.linalg.norm(dot1.get_center()))
        self.play(FadeIn(tex_dot,shift=dot.get_center()/np.linalg.norm(dot.get_center())),
                  FadeIn(tex_dot1,shift=dot1.get_center()/np.linalg.norm(dot1.get_center()))
                  )
        tex_dot.add_updater(lambda mob:mob.move_to(dot.get_center()+factor*(dot.get_center()-numberplane.c2p(0,0))/np.linalg.norm((dot.get_center()-numberplane.c2p(0,0)))))
        tex_dot1.add_updater(lambda mob:mob.move_to(dot1.get_center()+factor*(dot1.get_center()-numberplane.c2p(0,0))/np.linalg.norm((dot1.get_center()-numberplane.c2p(0,0)))))
        self.wait()
        self.play(MoveAlongPath(dot,ArcBetweenPoints(start=dot.get_center(),end=(np.array([-2,-2,0])))),
                  MoveAlongPath(dot1,ArcBetweenPoints(start=dot1.get_center(),end=(np.array([2,-2,0])))),
                  lag_ratio=0.3,run_time=3
                  )
        ##############对比一维与二维
        self.wait()
        numberplane.become(NumberPlane(background_line_style = dict(
            stroke_color=BLUE_B,
            stroke_width=2,
            stroke_opacity=0.5,
        ),faded_line_ratio=2))
        self.wait()
        k=NumberLine(x_range=(-6,6,1),include_tip=True,color="#f7cee7")
        line = Group(line_dot1,line_dot2,k)
        line.shift(np.array([0,0,-5]))
        line_dot1.move_to(k.n2p(-3))
        line_dot2.move_to(k.n2p(3))
        
        self.play(AnimationGroup(AnimationGroup(frame.animate.reorient(0,60,0,(0,0,-3)),run_time=4),
                  AnimationGroup(FadeIn(line_dot1,shift=LEFT,scale=1.5),
                                 FadeIn(line_dot2,shift=LEFT,scale=1.5),
                                 FadeIn(k,shift=OUT,scale=0.5)
                                 ,run_time=2),lag_ratio=0.4))
        self.wait()
        self.play(
            numberplane.animate.become(NumberPlane(x_range=(-4,4,1),background_line_style = dict(
            stroke_color=BLUE_B,
            stroke_width=2,
            stroke_opacity=0.5,
        ),faded_line_ratio=2)),
            run_time=2
        )
        self.wait()
        dot.add_updater(lambda mob:mob.move_to(numberplane.c2p(-4,-4)))
        dot1.add_updater(lambda mob:mob.move_to(numberplane.c2p(4,-4)))
        line_dot1.add_updater(lambda mob:mob.move_to(k.n2p(-3)))
        line_dot2.add_updater(lambda mob:mob.move_to(k.n2p(3)))
        self.play(AnimationGroup(
            AnimationGroup(frame.animate.reorient(0,0,0,(0,0,0)),run_time=4),
            AnimationGroup(
                numberplane.animate.move_to(np.array([-3.2,1.6,0])).scale(0.5),
                Group(dot,dot1,tex_dot,tex_dot1).animate.scale(0.8),
                k.animate.scale(0.75).move_to(np.array([-1.,-2.5,0])),
                run_time=2
            ),
            lag_ratio=0.2
        ))
        self.wait()
        ########介绍思路，想从一维推广到二维
        coord_dot1 = Tex(r"x_1").set_color(line_dot1[0].get_color()).next_to(line_dot1,UP)
        coord_dot2 = Tex(r"x_2").set_color(line_dot2[0].get_color()).next_to(line_dot2,UP)
        line_text = Text("一维情形",font="KaiTi").set_color(BLUE_A).next_to(coord_dot2,UP,buff=1)
        plane_text = Text("二维情形",font="KaiTi").set_color(YELLOW_A).next_to(line_text,2.7*UP,buff=1)
        eq_plane = Text("???",font="KaiTi").set_color(YELLOW_A).move_to(plane_text)
        eq_line = Tex(r"x_1x_2",tex_to_color_map={"x_1":BLUE_A,"x_2":GREEN_A}).move_to(line_text)
        eq_line1 = Text("关\n键\n性\n质",font="KaiTi",font_size=48).set_color(k.get_color()).next_to((eq_line.get_top()+eq_plane.get_bottom())/2,1.5*np.sqrt(3)*RIGHT,buff=1)
        self.play(FadeIn(line_text,shift=DOWN,scale=0.5))
        self.wait()
        self.play(FadeIn(coord_dot1,shift=DOWN,scale=0.5),
                  FadeIn(coord_dot2,shift=DOWN,scale=0.5))
        self.play(FadeInbyrotate(plane_text,rate_func=rush_from),run_time=3)
        self.wait()
        coord = Axes(x_range=[-0.25, 0.25, 0.1],
                     y_range=[0, 1, 0.2])
        s = StreamLines(
            lambda state: ( 0 , 1),
            coord,
            magnitude_range=(0.5, 5),
            solution_time = 1.5,
        ).move_to((eq_line.get_top()+eq_plane.get_bottom())/2)
        asl = AnimatedStreamLines(s)
        self.play(AnimationGroup(Transform(line_text,eq_line)))
        self.wait()
        self.play(FadeIn(asl,shift=UP,scale=1.2))
        self.wait()
        self.play(Transform(plane_text,eq_plane))
        self.remove(plane_text,line_text)
        self.add(eq_line,eq_plane)
        self.wait()
        points = []
        vector1 = 1*UP+1.35*np.sqrt(3)*RIGHT
        vector2 = -1*RIGHT+1.35*np.sqrt(3)*UP
        for x in np.arange(-5,5,0.6):
            for y in np.arange(-3,3,0.3):
                if np.linalg.norm(np.cross(np.array([x,y,0]),vector1))/np.linalg.norm(vector1)<0.4 and np.linalg.norm(np.cross(np.array([x,y,0]),vector2))/np.linalg.norm(vector2)<0.7:
                    points.append(x* RIGHT +y * UP)
        vec_field =[]
        for point in points:
            field = 0.25*UP+0.25*1.35*np.sqrt(3)*RIGHT
            result = Vector(field).shift(point)
            result.virtual_time=1
            result.set_color(BLUE_A)
            vec_field.append(result)
        draw_field = VGroup(*vec_field)
        draw_field.shift((eq_line.get_center()+eq_line1.get_center())/2+0.2*DOWN)
        streamvector = AnimatedStreamLinesbywjj(draw_field,shift=vector1*0.5)
        self.add(streamvector)
        self.wait()
        self.play(
            FadeIn(eq_line1,shift=1.35*UP+1.35*np.sqrt(3)*RIGHT)
            )
        self.wait(2)
        ############画箭头
        points = []
        vector1 = 1*UP-1.35*np.sqrt(3)*RIGHT
        vector2 = 1*RIGHT+1.35*np.sqrt(3)*UP
        for x in np.arange(-5,5,0.6):
            for y in np.arange(-3,3,0.3):
                if np.linalg.norm(np.cross(np.array([x,y,0]),vector1))/np.linalg.norm(vector1)<0.4 and np.linalg.norm(np.cross(np.array([x,y,0]),vector2))/np.linalg.norm(vector2)<0.7:
                    points.append(x* RIGHT +y * UP)
        vec_field =[]
        for point in points:
            field = 0.25*UP-0.25*1.35*np.sqrt(3)*RIGHT
            result = Vector(field).shift(point)
            result.virtual_time=1
            result.set_color(GREEN_A)
            vec_field.append(result)
        draw_field1 = VGroup(*vec_field)
        draw_field1.shift((eq_line1.get_center()+eq_plane.get_center())/2+0.2*UP)
        streamvector1 = AnimatedStreamLinesbywjj(draw_field1,shift=vector1*0.5)
        self.add(streamvector1)
        self.wait(10)
        self.play(FlashAround(k,color=GREEN_A),run_time=2)
        self.wait()
        self.play(AnimationGroup(
            AnimationGroup(frame.animate.reorient(0,0,0,k.n2p(0)+3*DOWN),run_time=4),
            AnimationGroup(MoveAlongPath(coord_dot1,ArcBetweenPoints(2*line_dot1.get_center()-coord_dot1.get_center(),coord_dot1.get_center(),).rotate(angle=-PI,about_point=line_dot1.get_center(),axis=RIGHT)),
                           MoveAlongPath(coord_dot2,ArcBetweenPoints(2*line_dot2.get_center()-coord_dot2.get_center(),coord_dot2.get_center(),).rotate(angle=-PI,about_point=line_dot2.get_center(),axis=RIGHT)),
                           run_time=2),lag_ratio=0.2
        ))
        self.wait()
class scene3(Scene):
    def construct(self) -> None:
        frame = self.camera.frame
        epsilon = 1e-5
        n_points = 10
        states = [
            [10, 10, 10 + n * epsilon]
            for n in range(n_points)
        ]
        colors = color_gradient([BLUE_E, BLUE_A], len(states))
        line_dot1 = get_special_dot(color=colors[-1], radius=0.25/3)
        line_dot2 = get_special_dot(color=GREEN_A, radius=0.25/3)
        k=NumberLine(x_range=(-6,6,1),include_tip=True,color="#f7cee7").scale(0.75)
        line = Group(line_dot1,line_dot2,k)
        line_dot1.move_to(k.n2p(-3))
        line_dot2.move_to(k.n2p(3))
        line.shift(3*UP)
        coord_dot1 = Tex(r"x_1").set_color(line_dot1[0].get_color()).next_to(line_dot1,DOWN)
        coord_dot2 = Tex(r"x_2").set_color(line_dot2[0].get_color()).next_to(line_dot2,DOWN)
        self.add(line,coord_dot1,coord_dot2)
        self.wait()
        ###########比如我们可以将x_2写成lambda\timesx_1就简写为lambda x_1
        ###########然后我们的点自然有这样的数乘运算，
        ###########那么我们的乘法从数值上似乎只满足这样的条件就足够了a*a=|a|**2（因为从运算律上还要满足交换、分配）
        dot = Dot().set_color(BLUE_A).move_to(k.n2p(0))
        origin_point = Tex("O").set_color(BLUE_A).next_to(k.n2p(0),DOWN)
        self.play(FadeIn(origin_point,shift=UP,scale=1.2),
                  FadeIn(dot,shift=DOWN,scale=1.2))
        self.wait()
        linedot1 = line_dot1.copy()
        linedot2 = line_dot2.copy()
        self.add(linedot1,linedot2)
        self.play(line_dot1.animate.shift(2*DOWN),coord_dot1.animate.shift(2*DOWN),
                  line_dot2.animate.shift(2*DOWN),coord_dot2.animate.shift(2*DOWN),run_time=2
                  )
        self.wait()
        coord_dot2_coef_x = Tex(r"x_1").set_color(line_dot1[0].get_color()).move_to(coord_dot2)
        dot2_bydot1 = line_dot1.copy().next_to(line_dot2,UP)
        coef_point = Tex(r"\lambda").set_color(line_dot2[0].get_color()).next_to(dot2_bydot1,LEFT)
        coef_num = Tex(r"\lambda").set_color(line_dot2[0].get_color()).move_to(np.array([coef_point.get_center()[0],coord_dot2_coef_x.get_center()[1],0]))
        coord_dot2_coef = VGroup(coord_dot2_coef_x,coef_num)
        dot2_withdot1 = Group(coef_point,dot2_bydot1)
        self.play(Transform(coord_dot2,coord_dot2_coef),run_time=2)
        self.remove(coord_dot2)
        self.add(coord_dot2_coef)
        self.wait()
        self.play(FadeIn(dot2_withdot1,shift=DOWN,scale=1.5),run_time=2)
        self.wait()
        coef_point1 = Tex(r"0.5").set_color(line_dot2[0].get_color()).move_to(coef_point)
        coef_num1 = Tex(r"0.5").set_color(line_dot2[0].get_color()).move_to(coef_num)
        coef_point2 = Tex(r"-0.5").set_color(line_dot2[0].get_color()).next_to(dot2_bydot1,LEFT)
        coef_num2 = Tex(r"-0.5").set_color(line_dot2[0].get_color()).move_to(np.array([coef_point2.get_center()[0],coord_dot2_coef_x.get_center()[1],0]))
        self.play(FadeOut(coef_point,shift=DOWN),FadeOut(coef_num,shift=DOWN),
                  FadeIn(coef_point1,shift=DOWN),FadeIn(coef_num1,shift=DOWN),run_time=1.5)
        self.play(linedot2.animate.move_to(k.n2p(-1.5)),run_time=0.5)
        self.wait(3)
        self.play(FadeOut(coef_point1,shift=DOWN),FadeOut(coef_num1,shift=DOWN),
                  FadeIn(coef_point2,shift=DOWN),FadeIn(coef_num2,shift=DOWN),run_time=1.5)
        self.play(linedot2.animate.move_to(k.n2p(1.5)),run_time=0.5)
        self.wait(3)
        self.play(FadeOut(coef_point2,shift=UP),FadeOut(coef_num2,shift=UP),
                  FadeIn(coef_point,shift=UP),FadeIn(coef_num,shift=UP),run_time=1)
        self.play(linedot2.animate.move_to(k.n2p(3)),run_time=0.5)
        self.wait()
        signal = VGroup(Tex(r"\times").next_to(line_dot1,RIGHT),
                        Tex(r"\times"),
                        Tex(r"\times"),
                        Tex(r"\Leftrightarrow").set_color(BLUE_A),
                        Tex("=").set_color(GREEN_A).rotate(PI/2),
                        )
        self.play(FadeIn(signal[0],shift=DOWN),
                  line_dot2.animate.next_to(signal[0],RIGHT))
        self.wait()
        signal[3].next_to(line_dot2,RIGHT,buff=1)
        temp_dot = line_dot1.copy().next_to(signal[3],RIGHT,buff=1)
        signal[1].next_to(temp_dot,RIGHT) 
        dot_temp = coef_point.copy().next_to(signal[1],RIGHT)       
        self.play(AnimationGroup(
            FadeIn(signal[3],shift=DOWN),
            AnimationGroup(
                FadeIn(temp_dot),
                FadeIn(signal[1]),
                coef_point.animate.next_to(signal[1],RIGHT),
                dot2_bydot1.animate.next_to(dot_temp,RIGHT),
            ),lag_ratio=0.3
        ))
        self.wait()
        signal[4].next_to(signal[1],DOWN,buff=0.4)
        signal[2].next_to(signal[4],DOWN,buff=0.4)
        dot_temp1 = coef_num.copy().next_to(signal[2],RIGHT)
        self.play(
            AnimationGroup(
                FadeIn(signal[4],shift=RIGHT),
                AnimationGroup(
                    FadeIn(signal[2]),
                    coord_dot1.animate.next_to(signal[2],LEFT),
                    coef_num.animate.next_to(signal[2],RIGHT),
                    coord_dot2_coef_x.animate.next_to(dot_temp1,RIGHT),
                ),lag_ratio=0.2
            ))
        self.wait()
        self.play(AnimationGroup(
            AnimationGroup(FlashAround(coef_point),FlashAround(coef_num),run_time=2),
            AnimationGroup(FadeOut(coef_point,shift=DOWN),FadeOut(coef_num,shift=DOWN)),
            AnimationGroup(dot2_bydot1.animate.next_to(signal[1],RIGHT),
                           coord_dot2_coef_x.animate.next_to(signal[2],RIGHT)),lag_ratio=0.8
        ))
        self.wait()
        temp_equality = Tex("=")
        self.play(
            AnimationGroup(
                AnimationGroup(
                AnimationGroup(SelfRotatingWithShift(signal[4],angle=3*PI/2),run_time=1.5),
                AnimationGroup(Group(temp_dot,signal[1],dot2_bydot1).animate.next_to(temp_equality,LEFT),
                VGroup(coord_dot1,signal[2],coord_dot2_coef_x).animate.next_to(temp_equality,RIGHT),
                FadeOut(line_dot1),
                FadeOut(line_dot2),
                FadeOut(signal[0]),
                FadeOut(signal[3]),run_time=1.1),lag_ratio=0.3
                ),
                AnimationGroup(
                    frame.animate.reorient(0,0,0,UP,6)
                ),lag_ratio=0.5
            )
        )
        self.wait()
        temp_rec = Rectangle(width=4.2,height=0.5)
        self.play(FlashAround(temp_rec,color=YELLOW_A))
        self.wait()
        eq_right = Tex(r"|x_1|^2").set_color(BLUE_A).next_to(signal[4],RIGHT)
        temp_vg = VGroup(coord_dot1,signal[2],coord_dot2_coef_x)
        self.play(Transform(temp_vg,eq_right))
        self.remove(temp_vg)
        self.add(eq_right)
        self.wait()
        line1 = Line(start=k.n2p(-3),end=k.n2p(0))
        brace_x_1 = BraceLabel(line1,text="|x_1|",brace_direction=UP).set_color(BLUE_A)
        self.play(FlashAround(linedot1))
        self.wait()
        self.play(AnimationGroup(
            frame.animate.shift(0.2*UP),
            FadeIn(brace_x_1,DOWN),lag_ratio=0.5
        ))
        self.wait()
        number_plane = NumberPlane(x_range=(-4,4,1),background_line_style = dict(
            stroke_color=BLUE_B,
            stroke_width=2,
            stroke_opacity=0.5,
        ),faded_line_ratio=2).scale(0.5).shift(3*LEFT)
        self.play(AnimationGroup(
            AnimationGroup(frame.animate.reorient(0,0,0,ORIGIN,8),run_time=4),
            AnimationGroup(FadeOut(k),FadeOut(brace_x_1),FadeOut(origin_point),FadeOut(dot),
                           Group(temp_dot,signal[1],dot2_bydot1,signal[4],eq_right).animate.shift(3.5*UP)),
            AnimationGroup(FadeIn(number_plane,scale=1.2),
                           linedot1.animate.move_to(number_plane.c2p(-1,2)),
                           linedot2.animate.move_to(number_plane.c2p(2,1)),
                           ),lag_ratio=0.2
        ))
        self.wait()
        eq_right1 = Tex(r"x_1^2+y_1^2").set_color(BLUE_A).next_to(signal[4],RIGHT)
        underline = Line(LEFT, RIGHT).set_color(GREEN_A)
        underline.next_to(signal[1], DOWN)
        underline.set_width(20)
        underline.shift(0.4*DOWN)
        b = Group(temp_dot,signal[1],dot2_bydot1,signal[4],eq_right1)
        self.play(b.animate.to_edge(UP),Write(underline),FadeOut(eq_right,UP))
        self.wait()
        coord_dot1_plane = Tex(r"(x_1,y_1)",font_size=32).set_color(linedot1[0].get_color()).next_to(linedot1,UL,buff=0.01)
        coord_dot2_plane = Tex(r"(x_2,y_2)",font_size=32).set_color(linedot2[0].get_color()).next_to(linedot2,UP)
        self.play(FadeIn(coord_dot1_plane,shift=DOWN),FadeIn(coord_dot2_plane,shift=DOWN))
        self.wait()
        eq_target = Tex(r"(x_1+x_2,y_1+y_2)",font_size=40).set_color(GREEN_A).shift(2.5*RIGHT+0.5*DOWN)
        eq_target_temp = VGroup(Tex(r"=",font_size=40).set_color(GREEN_A).rotate(PI/2),
                                Tex(r"(x_1,y_1)+(x_2,y_2)",font_size=40).set_color(RED_A))
        eq_target_temp[0].next_to(eq_target,UP)
        eq_target_temp[1].next_to(eq_target_temp[0],UP)
        eq_target_temp_end = Tex(r"[(x_1,y_1)+(x_2,y_2)]^2",font_size=40).set_color(RED_A).move_to(eq_target_temp[1])
        eq_target_end = Tex(r"(x_1+x_2,y_1+y_2)^2",font_size=40).set_color(GREEN_A).move_to(eq_target)
        self.play(FadeIn(eq_target_temp[1],scale=1.5))
        self.wait()
        self.play(AnimationGroup(Write(eq_target_temp[0]),
                                 Write(eq_target),lag_ratio=0.5,run_time=3))
        self.wait()
        equality = VGroup(Tex("=").set_color_by_gradient(GREEN_A,BLUE_A).rotate(PI/2),
                          Tex("=").set_color_by_gradient(BLUE_A,GREEN_A).rotate(PI/2))
        equality[0].next_to(eq_target,DOWN)
        eq_target_right = Tex(r"(x_1+x_2)^2+(y_1+y_2)^2",font_size=40).set_color(BLUE_A).next_to(equality[0],DOWN)
        equality[1].next_to(eq_target_temp[1],UP)
        eq_target_left = Tex(r"(x_1,y_1)^2+2(x_1,y_1)\times(x_2,y_2)+(x_2,y_2)^2",font_size=32).set_color(GREEN_A).next_to(equality[1],UP)
        self.play(Transform(eq_target,eq_target_end),
                  Transform(eq_target_temp[1],eq_target_temp_end),run_time=3)
        self.wait()
        self.play(FlashAround(b,color=[BLUE_A,GREEN_A]),b.animate.scale(1.1),rate_func=there_and_back,run_time=2)
        self.wait()
        self.play(FadeIn(equality[0],shift=LEFT),FadeIn(eq_target_right,scale=1.5),run_time=2)
        self.wait()
        self.play(FadeIn(equality[1],shift=RIGHT),TransformFromCopy(eq_target_temp,eq_target_left),run_time=2)
        self.wait()
        
        sr1,sr2=SurroundingRectangle(eq_target_left[:8]).set_color(RED_A),SurroundingRectangle(eq_target_left[-8:]).set_color(RED_A)
        self.play(Write(sr1),Write(sr2),run_time=2)
        self.wait()
        self.play(AnimationGroup(
            AnimationGroup(FlashAround(eq_target_right[:8]),FlashAround(eq_target_right[-8:]),run_time=2),
            AnimationGroup(FlashAround(eq_target_right[:8]),FlashAround(eq_target_right[-8:]),run_time=2),lag_ratio=0.2
        ))
        eq_target_right_end = Tex(r"x_1x_2+y_1y_2",font_size=40).set_color(BLUE_A).next_to(equality[1],DOWN)
        eq_target_left_end = Tex(r"(x_1,y_1)\times(x_2,y_2)",font_size=32).set_color(GREEN_A).move_to(eq_target_left)
        self.play(
            AnimationGroup(
                AnimationGroup(FadeOut(equality[0]),FadeOut(eq_target),FadeOut(sr1),FadeOut(sr2),FadeOut(eq_target_temp),run_time=2),
                AnimationGroup(Transform(eq_target_right,eq_target_right_end),
                               Transform(eq_target_left,eq_target_left_end),run_time=2)
            ,lag_ratio=0.3))
        self.remove(eq_target_left,eq_target_right)
        self.add(eq_target_right_end,eq_target_left_end)
        self.wait()
        vector1 = Arrow(start=number_plane.c2p(0,0),end=linedot1.get_center(),buff=0).set_color(linedot1[0].get_color())
        vector2 = Arrow(start=number_plane.c2p(0,0),end=linedot2.get_center(),buff=0).set_color(linedot2[0].get_color())
        self.play(FadeIn(vector1,shift=LEFT),FadeIn(vector2,shift=RIGHT),run_time=3)
        self.wait()
        self.play(Transform(eq_target_left_end[7],Tex(r"\cdot").move_to(eq_target_left_end[7].get_center())))
        self.wait()
class scene4(Scene):
    def construct(self) -> None:
        ####既然走到这里了，我们不妨回顾一下来路
        self.camera.frame.reorient(-90,45,90,np.array([-3,0,0]),height=6)
        frame = self.camera.frame
        default = (-2,2,1)
        radius = 0.1
        #我们首先从一维的数轴，提炼出了点的乘积的一个重要性质，dxd = |d|**2
        d1 = NumberLine((-2,2,1),include_tip=True,color=BLUE_A).move_to(np.array([-4,0,0]))
        d1.rotate(axis=np.array([2,0,1]),angle=PI,about_point=d1.n2p(0))
        d1_dot1 = get_special_dot(color="#f7cee7",radius=radius).move_to(d1.n2p(1))
        d1_dot2 = get_special_dot(color="#f5dbec",radius=radius).move_to(d1.n2p(-1))
        d1_all = Group(d1,d1_dot1,d1_dot2)
        #我们再由这个性质进行拓展，得到了二维平面里点的表达式，我们发现这竟然和向量的内积表达式居然一样
        #这或许说明对于向量的内积而言，其重要性质应当也是————————
        d2 = Axes(x_range=default,y_range=default,axis_config={"include_tip":True}).move_to(np.array([0,10,0]))
        d2.rotate(axis=np.array([1,0,0]),angle=PI/6,about_point=d2.c2p(0,0))
        d2.x_axis.set_color(BLUE_A)
        d2.y_axis.set_color(GREEN_A)
        d2_dot1 = get_special_dot(color=BLUE_A,radius=radius).move_to(d2.c2p(1,1))
        d2_dot2 = get_special_dot(color=GREEN_A,radius=radius).move_to(d2.c2p(-1,2))
        d2_all = Group(d2,d2_dot1,d2_dot2)
        #这里留做一个练习，如何用这个性质将点的乘积延拓到三维空间呢？
        #在下一期视频，我们会聊聊向量到底是什么，他为什么能表示方向？
        d3 = ThreeDAxes(x_range=default,y_range=default,z_range=default,axis_config={"include_tip":True}).move_to(np.array([6,3,0]))
        d3.x_axis.set_color(BLUE_A)
        d3.y_axis.set_color(GREEN_A)
        d3.z_axis.set_color(YELLOW_A)
        d3_dot1 = get_special_dot(color=GOLD_A,radius=radius).move_to(d3.c2p(0.6,0.2,1))
        d3_dot2 = get_special_dot(color=RED_A,radius=radius).move_to(d3.c2p(-1,0,0.5))
        d3_all = Group(d3,d3_dot1,d3_dot2)

        #公式aXa=||a||^2
        eq = Tex(r"a\times a = ||a||^2",font_size=72).set_color_by_gradient("#f7cee7","#f5dbec","#e9cefa").fix_in_frame()
        eq.shift(DOWN)
        signal = Tex(r"\Longrightarrow",font_size=100).set_color(BLUE_A).rotate(-30*DEG)
        signal.move_to(np.array([-1.5,0.5,0]))
        signal.fix_in_frame()
        self.play(FadeIn(d1_all,scale=1.5,shift=DOWN),run_time=2)
        self.wait()
        self.play(AnimationGroup(
                AnimationGroup(d1_all.animate.shift(3*UP+2*LEFT),
                  frame.animate.reorient(0,60,0,ORIGIN,height=8),run_time=3),
                AnimationGroup(Write(signal),
                  FadeIn(eq,shift=UP,scale=0.5),run_time=3,lag_ratio=0.5)
                  ,run_time=6,lag_ratio=0.5)
                  ,rate_func=rush_into)
        d1_all.add_updater(lambda mob:mob.rotate(angle=0.6*DEG,about_point=d1.n2p(0),axis=OUT))
        signal1 = Tex(r"\Longrightarrow",font_size=72).set_color(GREEN_A).rotate(90*DEG)
        signal1.move_to(np.array([0,1,0]))
        signal1.fix_in_frame()
        self.play(AnimationGroup(
            Write(signal1),
            FadeIn(d2_all,scale=0.5,shift=IN),lag_ratio=0.7,run_time=5
        ))
        signal12 = Tex(r"\Longrightarrow",font_size=72).set_color(GOLD_A).rotate(30*DEG)
        signal12.shift(np.array([-2,2,0]))
        signal12.fix_in_frame()
        signal2 = VGroup(Tex(r"\Longrightarrow",font_size=72).set_color(GOLD_A),
                         Tex("?",font_size=72).set_color(RED_A))
        signal2.rotate(-30*DEG)
        signal2.shift(np.array([2,2,0]))
        signal2.fix_in_frame()
        d2_all.add_updater(lambda mob:mob.rotate(angle=0.7*DEG,about_point=d2.c2p(0,0)))
        k=VGroup(signal,signal1)
        srec_eq=SurroundingRectangle(eq).set_color(GREEN_A)
        self.play(AnimationGroup(Write(srec_eq),
                                 Transform(k,signal12),run_time=5,lag_ratio=0.6))
        self.remove(signal,signal1)
        self.add(signal12)
        self.wait(2)
        self.play(AnimationGroup(
            Write(signal2),
            FadeIn(d3_all,scale=0.5,shift=IN+LEFT),lag_ratio=0.5,run_time=5
        ))
        d3_all.add_updater(lambda mob:mob.rotate(angle=0.6*DEG,about_point=d3.c2p(0,0,0),axis=OUT))
        self.wait(7)
        d1_all.clear_updaters()
        d2_all.clear_updaters()
        d3_all.clear_updaters()
        self.wait()
class End(Scene):
    def construct(self) -> None:
        str0="片名：{}".format("聊聊向量——点积")
        str1="出品人：{}".format("擢兰风而霞起")
        str2="使用工具：{}".format("ManimGL")
        str3="参考文献：{}".format("太多啦")
        str4="特别鸣谢：{}".format("所有开源作者")
        str_list=[str0,str1,str2,str3,str4]
        text=Group()
        for i in range(len(str_list)):
            text.add(Text(str_list[i],font="KaiTi",
                          font_size=24).set_color(TEAL_A))
        text.arrange(DOWN,aligned_edge=LEFT,buff=1)
        text[3].set_color(GREEN_A)
        text.shift(7*DOWN+2*RIGHT)
        self.play(text.animate.shift(14*UP),run_time=15)
class addition1(Scene):
    def construct(self) -> None:
        
        frame = self.camera.frame
        epsilon = 1e-5
        n_points = 10
        states = [
            [10, 10, 10 + n * epsilon]
            for n in range(n_points)
        ]
        colors = color_gradient([BLUE_E, BLUE_A], len(states))
        line_dot1 = get_special_dot(color=colors[-1], radius=0.25/3)
        line_dot2 = get_special_dot(color=GREEN_A, radius=0.25/3)
        k=NumberLine(x_range=(-6,6,1),include_tip=True,color="#f7cee7").scale(0.75)
        line = Group(line_dot1,line_dot2,k)
        line_dot1.move_to(k.n2p(-3))
        line_dot2.move_to(k.n2p(3))
        line.shift(3*UP)
        coord_dot1 = Tex(r"x_1").set_color(line_dot1[0].get_color()).next_to(line_dot1,DOWN)
        coord_dot2 = Tex(r"x_2").set_color(line_dot2[0].get_color()).next_to(line_dot2,DOWN)
        self.add(line,coord_dot1,coord_dot2)
        self.wait()
        ###########比如我们可以将x_2写成lambda\timesx_1就简写为lambda x_1
        ###########然后我们的点自然有这样的数乘运算，
        ###########那么我们的乘法从数值上似乎只满足这样的条件就足够了a*a=|a|**2（因为从运算律上还要满足交换、分配）
        dot = Dot().set_color(BLUE_A).move_to(k.n2p(0))
        origin_point = Tex("O").set_color(BLUE_A).next_to(k.n2p(0),DOWN)
        self.play(FadeIn(origin_point,shift=UP,scale=1.2),
                  FadeIn(dot,shift=DOWN,scale=1.2))
        self.wait()
        linedot1 = line_dot1.copy()
        linedot2 = line_dot2.copy()
        self.add(linedot1,linedot2)
        self.play(line_dot1.animate.shift(2*DOWN),coord_dot1.animate.shift(2*DOWN),
                  line_dot2.animate.shift(2*DOWN),coord_dot2.animate.shift(2*DOWN),run_time=2
                  )
        self.wait()
        coord_dot2_coef_x = Tex(r"x_1").set_color(line_dot1[0].get_color()).move_to(coord_dot2)
        dot2_bydot1 = line_dot1.copy().next_to(line_dot2,UP)
        coef_point = Tex(r"\lambda").set_color(line_dot2[0].get_color()).next_to(dot2_bydot1,LEFT)
        coef_num = Tex(r"\lambda").set_color(line_dot2[0].get_color()).move_to(np.array([coef_point.get_center()[0],coord_dot2_coef_x.get_center()[1],0]))
        coord_dot2_coef = VGroup(coord_dot2_coef_x,coef_num)
        dot2_withdot1 = Group(coef_point,dot2_bydot1)
        self.play(Transform(coord_dot2,coord_dot2_coef),run_time=2)
        self.remove(coord_dot2)
        self.add(coord_dot2_coef)
        self.wait()
        self.play(FadeIn(dot2_withdot1,shift=DOWN,scale=1.5),run_time=2)
        self.wait()
        coef_point1 = Tex(r"0.5").set_color(line_dot2[0].get_color()).move_to(coef_point)
        coef_num1 = Tex(r"0.5").set_color(line_dot2[0].get_color()).move_to(coef_num)
        coef_point2 = Tex(r"-0.5").set_color(line_dot2[0].get_color()).next_to(dot2_bydot1,LEFT)
        coef_num2 = Tex(r"-0.5").set_color(line_dot2[0].get_color()).move_to(np.array([coef_point2.get_center()[0],coord_dot2_coef_x.get_center()[1],0]))
        self.play(FadeOut(coef_point,shift=DOWN),FadeOut(coef_num,shift=DOWN),
                  FadeIn(coef_point1,shift=DOWN),FadeIn(coef_num1,shift=DOWN),run_time=1.5)
        self.play(linedot2.animate.move_to(k.n2p(-1.5)),run_time=0.5)
        self.wait(3)
        self.play(FadeOut(coef_point1,shift=DOWN),FadeOut(coef_num1,shift=DOWN),
                  FadeIn(coef_point2,shift=DOWN),FadeIn(coef_num2,shift=DOWN),run_time=1.5)
        self.play(linedot2.animate.move_to(k.n2p(1.5)),run_time=0.5)
        self.wait(3)
        self.play(FadeOut(coef_point2,shift=UP),FadeOut(coef_num2,shift=UP),
                  FadeIn(coef_point,shift=UP),FadeIn(coef_num,shift=UP),run_time=1)
        self.play(linedot2.animate.move_to(k.n2p(3)),run_time=0.5)
        self.wait()
        signal = VGroup(Tex(r"\times").next_to(line_dot1,RIGHT),
                        Tex(r"\times"),
                        Tex(r"\times"),
                        Tex(r"\Leftrightarrow").set_color(BLUE_A),
                        Tex("=").set_color(GREEN_A).rotate(PI/2),
                        )
        self.play(FadeIn(signal[0],shift=DOWN),
                  line_dot2.animate.next_to(signal[0],RIGHT))
        self.wait()
        signal[3].next_to(line_dot2,RIGHT,buff=1)
        temp_dot = line_dot1.copy().next_to(signal[3],RIGHT,buff=1)
        signal[1].next_to(temp_dot,RIGHT) 
        dot_temp = coef_point.copy().next_to(signal[1],RIGHT)       
        self.play(AnimationGroup(
            FadeIn(signal[3],shift=DOWN),
            AnimationGroup(
                FadeIn(temp_dot),
                FadeIn(signal[1]),
                coef_point.animate.next_to(signal[1],RIGHT),
                dot2_bydot1.animate.next_to(dot_temp,RIGHT),
            ),lag_ratio=0.3
        ))
        self.wait()
        signal[4].next_to(signal[1],DOWN,buff=0.4)
        signal[2].next_to(signal[4],DOWN,buff=0.4)
        dot_temp1 = coef_num.copy().next_to(signal[2],RIGHT)
        self.play(
            AnimationGroup(
                FadeIn(signal[4],shift=RIGHT),
                AnimationGroup(
                    FadeIn(signal[2]),
                    coord_dot1.animate.next_to(signal[2],LEFT),
                    coef_num.animate.next_to(signal[2],RIGHT),
                    coord_dot2_coef_x.animate.next_to(dot_temp1,RIGHT),
                ),lag_ratio=0.2
            ))
        self.wait()
        self.play(AnimationGroup(
            AnimationGroup(FlashAround(coef_point),FlashAround(coef_num),run_time=2),
            AnimationGroup(FadeOut(coef_point,shift=DOWN),FadeOut(coef_num,shift=DOWN)),
            AnimationGroup(dot2_bydot1.animate.next_to(signal[1],RIGHT),
                           coord_dot2_coef_x.animate.next_to(signal[2],RIGHT)),lag_ratio=0.8
        ))
        self.wait()
        temp_equality = Tex("=")
        self.play(
            AnimationGroup(
                AnimationGroup(
                AnimationGroup(SelfRotatingWithShift(signal[4],angle=3*PI/2),run_time=1.5),
                AnimationGroup(Group(temp_dot,signal[1],dot2_bydot1).animate.next_to(temp_equality,LEFT),
                VGroup(coord_dot1,signal[2],coord_dot2_coef_x).animate.next_to(temp_equality,RIGHT),
                FadeOut(line_dot1),
                FadeOut(line_dot2),
                FadeOut(signal[0]),
                FadeOut(signal[3]),run_time=1.1),lag_ratio=0.3
                ),
                AnimationGroup(
                    frame.animate.reorient(0,0,0,UP,6)
                ),lag_ratio=0.5
            )
        )
        self.wait()
        temp_rec = Rectangle(width=4.2,height=0.5)
        self.play(FlashAround(temp_rec,color=YELLOW_A))
        self.wait()
        eq_right = Tex(r"|x_1|^2").set_color(BLUE_A).next_to(signal[4],RIGHT)
        temp_vg = VGroup(coord_dot1,signal[2],coord_dot2_coef_x)
        self.play(Transform(temp_vg,eq_right))
        self.remove(temp_vg)
        self.add(eq_right)
        self.wait()
        line1 = Line(start=k.n2p(-3),end=k.n2p(0))
        brace_x_1 = BraceLabel(line1,text="|x_1|",brace_direction=UP).set_color(BLUE_A)
        self.play(FlashAround(linedot1))
        self.wait()
        self.play(AnimationGroup(
            frame.animate.shift(0.2*UP),
            FadeIn(brace_x_1,DOWN),lag_ratio=0.5
        ))
        self.wait()
        number_plane = NumberPlane(x_range=(-4,4,1),background_line_style = dict(
            stroke_color=BLUE_B,
            stroke_width=2,
            stroke_opacity=0.5,
        ),faded_line_ratio=2).scale(0.5).shift(3*LEFT)
        self.play(AnimationGroup(
            AnimationGroup(frame.animate.reorient(0,0,0,ORIGIN,8),run_time=4),
            AnimationGroup(FadeOut(k),FadeOut(brace_x_1),FadeOut(origin_point),FadeOut(dot),
                           Group(temp_dot,signal[1],dot2_bydot1,signal[4],eq_right).animate.shift(3.5*UP)),
            AnimationGroup(FadeIn(number_plane,scale=1.2),
                           linedot1.animate.move_to(number_plane.c2p(-1,2)),
                           linedot2.animate.move_to(number_plane.c2p(2,1)),
                           ),lag_ratio=0.2
        ))
        self.wait()
if __name__ == "__main__":
    os.system("manimgl {} addition1 -c black --uhd -w".format(__file__))