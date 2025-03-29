from manimlib import *
import random
import numpy as np
import os
#surface functions
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
def square_func(u, v):
    return (u, v, 0)
def rectangle_func(v, u):
    u1 = np.sqrt(3)*(2*u-1)
    v1 = 2*v-1
    return (u1, v1, 0)
def half_mobius_func(u,v):
    return mobius_strip_func(u,0.5*v)
def mobius_strip_func(u, v, outer_radius=1.5, inner_radius=0.5):
    theta = TAU * v
    phi = theta / 2
    p = math.cos(theta) * RIGHT + math.sin(theta) * UP
    q = math.cos(phi) * p + math.sin(phi) * OUT
    return outer_radius * p + inner_radius * q * (2 * u - 1)
def mobius_strip_func1(v, u, outer_radius=1.5, inner_radius=0.5):
    theta = TAU * v
    phi = theta / 2
    p = math.cos(theta) * RIGHT + math.sin(theta) * UP
    q = math.cos(phi) * p + math.sin(phi) * OUT
    return outer_radius * p + inner_radius * q * (2 * u - 1)
def get_full_surface(band_func, x_range):
    surface = ParametricSurface(
        band_func, x_range, (0, TAU),
    )
    surface.set_color(BLUE_D)
    surface.set_shadow(0.5)
    surface.add_updater(lambda m: m.sort_faces_back_to_front(DOWN))
    # surface = TexturedSurface(surface, "EarthTextureMap", "NightEarthTextureMap")
    # surface = TexturedSurface(surface, "WaterColor")
    # inv_surface = ParametricSurface(
    #     reversed_band(band_func), x_range[::-1], (0, TAU),
    # )
    m1, m2 = meshes = VGroup(
        SurfaceMesh(surface, normal_nudge=1e-3),
        SurfaceMesh(surface, normal_nudge=-1e-3),
    )
    bound = VGroup(
        ParametricCurve(lambda t: band_func(x_range[0], t), (0,TAU,0.01)),
        ParametricCurve(lambda t: band_func(x_range[1], t), (0, TAU,0.01)),
    )
    bound.set_stroke(RED, 3)
    bound.apply_depth_test()
    meshes.set_stroke(WHITE, 0.5, 0.5)
    return Group(surface, m1, m2, bound)
def trapezoid(u,v,point3 = (0.8,1),point4 = (3.2,1),a=4):
    #传入上底两个顶点
    c,h=point3
    d,h=point4
    return np.array([a*u*(1-v)+c*v+(d-c)*u*v,h*v,0])
def mobius(u, v):
    x = (1 + 0.5 * v * np.cos(u / 2)) * np.cos(u)
    y = (1 + 0.5 * v * np.cos(u / 2)) * np.sin(u)
    z = 0.5 * v * np.sin(u / 2)
    return np.array([x, y, z])
def mobius1(v,u):
    x = (1 + 0.5 * v * np.cos(u / 2)) * np.cos(u)
    y = (1 + 0.5 * v * np.cos(u / 2)) * np.sin(u)
    z = 0.5 * v * np.sin(u / 2)
    return np.array([x, y, z])
def Plane(u, v):
    return np.array([3*u, 3*v, 0])

def cylinder(u,v):
    return np.array([np.cos(u)+2, np.sin(u), v])
def cylinder1(u,v):
    return np.array([np.cos(v), np.sin(v), u])
class prob_exp(InteractiveScene):
    def construct(self):
        frame=self.camera.frame
        axes, plane = self.get_axes_and_plane()
        self.play(Write(plane))
        #定义两个曲面，平面与莫比乌斯面
        surfaces = Group(ParametricSurface(rectangle_func).set_color([BLUE_A,TEAL,BLUE_A]),
                         ParametricSurface(mobius_strip_func).set_color([BLUE_A,TEAL,BLUE_A])
            )
        for surface in surfaces:
            surface.set_shading(0.25, 0.25, 0)
            surface.set_opacity(0.75)
        rectangle , mobius_strip= surfaces
        mobius_strip.shift(2*OUT)
        text=VGroup(Tex("1").set_color(GREEN).next_to(rectangle,RIGHT,buff=0.2),
                    Tex(r"\lambda").set_color(BLUE).next_to(rectangle,DOWN,buff=0.2))
        self.play(FadeIn(rectangle),run_time=2)
        self.play(Write(text[0]),Write(text[1]))
        self.wait()
        square = rectangle.copy()
        self.play(
            frame.animate.reorient(5, 79, 0, (0.4, 0.01, 1.41), 5.07),
            FadeIn(axes),
            square.animate.shift(2*OUT),
            run_time=3
        )
        
        self.play(
            frame.animate.reorient(-21, 84, 0, (0.4, 0.01, 1.41), 5.07),
            Transform(square,mobius_strip),
            run_time=5
        )
        self.remove(square)
        self.add(mobius_strip)
        self.play(frame.animate.scale(2).reorient(-21, 60, 0, (-2.41, -0.13, 1.34), 7.76)
                  ,run_time=3)
        #创建两点，在两个面上运动
        u_tracker = ValueTracker(0.7)
        v_tracker = ValueTracker(0.8)
        #点的颜色
        def mobius_update_dot(mob):
            u = u_tracker.get_value()
            v = v_tracker.get_value()
            pos = mobius_strip_func(u, v)+2.05*OUT
            mob.move_to(pos)
        def rectangle_update_dot(mob):
            u = u_tracker.get_value()
            v = v_tracker.get_value()
            pos = rectangle_func(u, v)+0.02*OUT
            mob.move_to(pos)
        rectangle_point = get_special_dot(radius=0.08)
        rectangle_point.add_updater(rectangle_update_dot)
        rectangle_point.apply_depth_test()
        mobius_point = get_special_dot(radius=0.1,color=BLUE)
        mobius_point.add_updater(mobius_update_dot)
        mobius_point.apply_depth_test()
        #向量，模拟映射
        vector = Arrow(start=rectangle_point.get_center(),end=mobius_point.get_center(),fill_color=BLUE_A)
        
        func = Text("I(x,y)").to_edge(UL).fix_in_frame()
        #绘制轨迹便于描述保距。
        rectangle_trace = TracedPath(rectangle_point.get_center,
                                  stroke_color=YELLOW,
                                  stroke_width=3)
        mobius_trace = TracedPath(mobius_point.get_center,
                                  stroke_color=BLUE,
                                  stroke_width=3)

        self.play(Write(func))
        self.play(FadeIn(rectangle_point))
        self.wait()
        self.play(AnimationGroup(
                    FlashAlongPath(Line(start=rectangle_point.get_center(),end=mobius_point.get_center()),color=BLUE_A),
                    FadeIn(mobius_point)
                    ,lag_ratio=0.2
                    ,run_time=3))
        self.play(Write(vector))
        vector.add_updater(lambda mob:mob.become(Arrow(start=rectangle_point.get_center(),end=mobius_point.get_center(),fill_color=BLUE_A)))
        self.add(mobius_trace,rectangle_trace)
        
        property_I = VGroup(Text("保距性", font="KaiTi").next_to(func,6*DOWN).fix_in_frame(),
                            Text("光滑性", font="KaiTi").next_to(func,12*DOWN).fix_in_frame())
        self.play(Write(property_I[0]))
        self.play(Write(property_I[1]))
        line = Line(start=ORIGIN,end=2*RIGHT, color=YELLOW).next_to(property_I[0],RIGHT).fix_in_frame()
        line.shift(0.2*DOWN)
        line.fix_in_frame()
        line1 = Line(start=ORIGIN,end=2*RIGHT, color=BLUE).next_to(property_I[0],RIGHT).fix_in_frame()
        line1.shift(0.2*UP)
        line1.fix_in_frame()
        self.wait(2)
        #展示保距性，用曲线拟合轨迹
        self.play(property_I[0].animate.scale(1.2))
        self.wait(2)
        self.play(u_tracker.animate.set_value(0.2),
                  v_tracker.animate.set_value(0.7),run_time=3)
        mobius_point.clear_updaters()
        rectangle_point.clear_updaters()
        vector.clear_updaters()
        frame.save_state()
        #展示三维轨迹
        self.play(frame.animate.reorient(0, 25, 10, (-0.41, -0.13, 1.34), 7.76),run_time=3)
        self.wait(2)
        self.play(frame.animate.restore(),run_time=2)
        all = Group(axes,plane,mobius_strip,rectangle,text)
        self.play(
                  FadeOut(mobius_point),
                  FadeOut(rectangle_point),
                  FadeOut(vector))
        curve1 = ParametricCurve(lambda t:rectangle_func(0.2+0.5*t,0.7+0.1*t)+0.02*OUT,t_range=(0,1,0.02)).set_stroke(YELLOW, 3, 1)
        
        curve2 = ParametricCurve(lambda t:mobius_strip_func(0.2+0.5*t,0.7+0.1*t)+2.05*OUT,t_range=(0,1,0.02)).set_stroke(BLUE, 3, 1)
        
        self.remove(rectangle_trace,mobius_trace)
        self.add(curve1,curve2)
        
        self.wait()
        
        self.play(Transform(curve1,line),
                  Transform(curve2,line1),
                  axes.animate.shift(2*RIGHT),
                  plane.animate.shift(2*RIGHT),
                  mobius_strip.animate.shift(2*RIGHT),
                  rectangle.animate.shift(2*RIGHT),
                  text.animate.shift(2*RIGHT),run_time=3)
        
        self.wait(3)
        #展示光滑性
        self.play(property_I[0].animate.scale(1/1.2),
                  property_I[1].animate.scale(1.2),
                  run_time=2)
        self.play(FadeOut(axes),
                  FadeOut(plane),
                  FadeOut(mobius_strip),
                  FadeOut(rectangle),
                  FadeOut(text),run_time=2)
        self.wait()
        self.play(property_I[1].animate.scale(1/1.2),
            FadeIn(axes),
                  FadeIn(plane),
                  FadeIn(mobius_strip),
                  FadeIn(rectangle),
                  FadeIn(text),run_time=2)
        self.wait(2)
        text_temp = Tex(r"\lambda > \sqrt{3}").set_color(BLUE).next_to(rectangle,DOWN,buff=0.2)
        self.play(AnimationGroup(
            frame.animate.reorient(0, 50, 0, text_temp.get_center()+2*UP+LEFT, 7.76)
            ,Transform(text[1],text_temp),lag_ratio=0.3
        ),run_time=4)
        self.wait()
    def get_axes_and_plane(
        self,
        x_range=(-3, 3),
        y_range=(-3, 3),
        z_range=(0, 5),
        depth=4,
    ):
        axes = ThreeDAxes(x_range, y_range, z_range)
        axes.set_depth(depth, stretch=True, about_edge=IN)
        axes.set_stroke(GREY_B, 1)
        plane = NumberPlane(x_range, y_range)
        plane.background_lines.set_stroke(BLUE, 1, 0.75)
        plane.faded_lines.set_stroke(BLUE, 0.5, 0.25)
        plane.axes.match_style(axes)
        plane.set_z_index(-1)
        return axes, plane
class triangle_mobius(Scene):
  def construct(self):
        rectangle = Rectangle(width=2*np.sqrt(3), height=2, fill_opacity=0.5, color=GREY)
        rectangle.set_stroke(width=0)
        # 定义矩形的四周
        DL_temp = rectangle.get_corner(DL)
        UL_temp = rectangle.get_corner(UL)
        UR_temp = rectangle.get_corner(UR)
        DR_temp = rectangle.get_corner(DR)
        UP_point1=UL_temp+2/3*(UR_temp-UL_temp)
        DOWN_point1=DL_temp+1/3*(DR_temp-DL_temp)
        x_line_color = BLUE
        y_line_color = GREEN
        # 展示折叠线
        left_edge = Line(DL_temp, UL_temp)
        left_edge.set_stroke(x_line_color,width = 3)
        right_edge = Line(UR_temp, DR_temp)
        right_edge.set_stroke(x_line_color,width = 3)
        up_edge = Line(UP_point1 , UR_temp)
        up_edge.set_stroke(y_line_color,width = 3)
        down_edge = Line(DOWN_point1 , DL_temp)
        down_edge.set_stroke(y_line_color,width = 3)

        left_tips = ArrowTip(angle=90 * DEG).get_grid(3, 1, buff=0.4)
        left_tips.move_to(left_edge)
        left_tips.set_color(x_line_color)
        right_tips = ArrowTip(angle=-90 * DEG).get_grid(3, 1, buff=0.4)
        right_tips.move_to(right_edge)
        right_tips.set_color(x_line_color)
        up_tips = ArrowTip(angle=0)
        up_tips.move_to(up_edge)
        up_tips.set_color(y_line_color)
        down_tips = ArrowTip(angle=0)
        down_tips.move_to(down_edge)
        down_tips.set_color(y_line_color)
        left_arrows = VGroup(left_edge, left_tips).set_opacity(0.25)
        right_arrows =  VGroup(right_edge, right_tips).set_opacity(0.25)
        up_arrows =  VGroup(up_edge, up_tips).set_opacity(0.25)
        down_arrows =  VGroup(down_edge, down_tips).set_opacity(0.25)
        center = rectangle.get_center()
        # 绘制四个三角形
        triangle1 = Polygon(DL_temp, UL_temp, DOWN_point1,)
        triangle2 = Polygon(UL_temp, UP_point1 , DOWN_point1)
        triangle3 = Polygon(DOWN_point1, UP_point1, DR_temp)
        triangle4 = Polygon(UP_point1, UR_temp, DR_temp)
        
        self.play(Write(rectangle))
        self.wait(2)
        
        self.remove(rectangle)
        for triangle in [triangle1, triangle2, triangle3, triangle4]:
            triangle.match_style(rectangle)
            triangle.set_z_index(-1)
            triangle.set_shading(0.25, 0, 0)
            self.add(triangle)
        self.play(FadeIn(left_arrows)
                  ,FadeIn(right_arrows)
                  ,FadeIn(up_arrows)
                  ,FadeIn(down_arrows))
        self.play(left_arrows.animate.set_opacity(1)
                  ,right_arrows.animate.set_opacity(1)
                  ,up_arrows.animate.set_opacity(1)
                  ,down_arrows.animate.set_opacity(1))
        self.wait(2)
        #对角折叠
        self.play(Rotate(triangle1, PI, about_point=(UL_temp+DOWN_point1)/2, axis=(DOWN_point1-UL_temp)),
                  Rotate(left_arrows, PI, about_point=(UL_temp+DOWN_point1)/2, axis=(DOWN_point1-UL_temp)),
                  Rotate(down_arrows, PI, about_point=(UL_temp+DOWN_point1)/2, axis=(DOWN_point1-UL_temp)),
                  Rotate(triangle4, PI, about_point=(DR_temp+UP_point1)/2, axis=UP_point1-DR_temp),
                  Rotate(right_arrows, PI, about_point=(DR_temp+UP_point1)/2, axis=UP_point1-DR_temp),
                  Rotate(up_arrows, PI, about_point=(DR_temp+UP_point1)/2, axis=UP_point1-DR_temp),
            run_time=2)
        #再折叠
        self.wait(2)
        self.play(Rotate(triangle2, PI, about_point=rectangle.get_center(), axis=np.array([1,np.sqrt(3),0])),
                  Rotate(triangle1, PI, about_point=rectangle.get_center(), axis=np.array([1,np.sqrt(3),0])),
                  Rotate(left_arrows, PI, about_point=rectangle.get_center(), axis=np.array([1,np.sqrt(3),0])),
            run_time=2)
        self.play(Rotate(triangle2, -PI, about_point=rectangle.get_center(), axis=np.array([1,np.sqrt(3),0])),
                  Rotate(triangle1, -PI, about_point=rectangle.get_center(), axis=np.array([1,np.sqrt(3),0])),
                  Rotate(left_arrows, -PI, about_point=rectangle.get_center(), axis=np.array([1,np.sqrt(3),0])),
            run_time=2)
        dot = get_special_dot()
        dot.shift(np.array([0.4,0.2,0]))
        self.play(FadeIn(dot))
        self.wait(2)
        #展示如何从外部进入内部
        self.play(Rotate(triangle1, PI, about_point=rectangle.get_center(), axis=np.array([1,np.sqrt(3),0])),
                  Rotate(left_arrows, PI, about_point=rectangle.get_center(), axis=np.array([1,np.sqrt(3),0])),
                  triangle4.animate.set_color(BLUE),
                  triangle1.animate.set_color(GREEN),
                  run_time=3)
        line_temp1 = Line(DOWN_point1,UL_temp).set_stroke(width=1,color=RED,opacity=0.6)
        line_temp2 = Line(DOWN_point1,DR_temp).set_stroke(width=1,color=RED,opacity=0.6)
        self.play(Write(line_temp1),Write(line_temp2))
        self.wait(2)
        self.play(left_arrows.animate.scale(1.2),right_arrows.animate.scale(1.2),run_time=3,rate_func=there_and_back)
        self.wait(2)
        self.play(dot.animate.move_to(np.array([0.2,-0.4,0])))
        connect_path = CubicBezier(np.array([0.2,-0.4,0]),np.array([0,-1,0])
                              ,np.array([-0.8,-0.6,0]),np.array([-0.6,0,0]),stroke_color=BLUE_A)
        self.play(FlashAlongPath(Line(DOWN_point1,UL_temp)),
                  FlashAlongPath(Line(DOWN_point1,DR_temp)),run_time=2)
        self.wait()
        self.play(AnimationGroup(FlashAlongPath(connect_path),MoveAlongPath(dot,connect_path),lag_ratio=0.3),run_time=3)
        self.wait(2)
        self.play(FadeOut(line_temp1),
                  FadeOut(line_temp2),
                  FadeOut(dot),
                  Rotate(triangle2, -PI, about_point=rectangle.get_center(), axis=np.array([1,np.sqrt(3),0])),
                  triangle4.animate.set_color(GREY),
                  triangle1.animate.set_color(GREY),run_time=3)
        all = Group(triangle1,triangle2,triangle3,triangle4,left_arrows,right_arrows,up_arrows,down_arrows)
        self.play(all.animate.shift(3*LEFT))
        scissor = ImageMobject("scissor.png")
        scissor.rotate_about_origin(195*DEGREES)
        scissor.scale(0.4)
        scissor.move_to(np.array([-2.4,1,0]))
        self.wait(2)
        self.play(FadeIn(scissor))
        result = Group(Square().scale(1.3) , Text("?",font_size=72).set_color(RED)).shift(3*RIGHT)
        scissor_trace = TracedPath(scissor.get_center,stroke_color=YELLOW)
        self.add(scissor_trace)
        tex = TitleText("想象形状").set_color(BLUE_A)
        self.play(Write(tex))
        self.play(scissor.animate.shift(2/np.sqrt(3)*LEFT+2*DOWN)
                  ,up_arrows.animate.set_opacity(0.2)
                  ,down_arrows.animate.set_opacity(0.2)
                  ,run_time=4)
        self.wait(1)
        self.play(TransformFromCopy(all , result)
                  ,FlashAlongPath(Line(triangle3.get_center(),result[0].get_center()).set_color(BLUE_A))
                  ,run_time=2)
        self.wait(2)
class MobiusStripAnimation(Scene):
    def construct(self):
        # 调整摄像机视角以获得三维效果
        frame = self.camera.frame
        self.camera.frame.set_euler_angles(phi=45 * DEGREES)
        
        surface1 = self.LineToSurface(Plane,u = [-1,1] , t_range = [-1,1,0.02])
        self.play(frame.animate.reorient(-30, 61, 0),
                  surface1.animate.set_opacity(0.8).shift(3*LEFT),run_time=2)
        surface2 = self.LineToSurface(cylinder,u = [0,2*PI] , t_range = [-2,2,0.02],color = [RED_A,TEAL,RED_A])
        self.play(frame.animate.reorient(0, 61, 0)
                  ,surface2.animate.shift(2*RIGHT),run_time=3)
        mesh1 =Group()
        mesh2 = Group()
        for i in range(20):
            u = -1 + 2/20*i
            mesh1.add(ParametricCurve(lambda v: Plane(u, v),
            t_range=[-1,1,0.02],color = [BLUE_A,TEAL,BLUE_A],
            stroke_width=2).shift(3*LEFT))
        for i in range(20):
            u = 0 + 2*PI/20*i
            mesh2.add(ParametricCurve(lambda v: cylinder(u, v),
            t_range=[-2,2,0.02],color = [BLUE_A,TEAL,BLUE_A],
            stroke_width=2).shift(2*RIGHT))
        
        title_k = TitleText("名为折痕").set_color(TEAL_A).fix_in_frame()
        self.play(
                )
        self.play(
            AnimationGroup(
                AnimationGroup(surface1.animate.set_shading(0.25, 0.25, 0).set_opacity(0.7),
                               surface2.animate.set_shading(0.25, 0.25, 0).set_opacity(0.7),run_time=2),
                AnimationGroup(FadeIn(mesh1),FadeIn(mesh2),run_time=2)
                ,Write(title_k,run_time=2)
            ,lag_ratio=0.2)
        )
        self.wait(2)
        
        ###################################################
        #演示莫比乌斯面的线动成面。
        title = TitleText("莫比乌斯面").set_color(BLUE_A).fix_in_frame()
        self.play(Transform(title_k,title),          
                  FadeOut(mesh1),
                  FadeOut(surface1),
                  FadeOut(surface2),
                  FadeOut(mesh2),
                  frame.animate.reorient(0, 0, 0),
                  run_time=3)

        self.add(title)
        self.remove(title_k)
        self.play(frame.animate.reorient(0, 45, 0))
        self.wait(2)
        def frame_updater(mob):
            mob.increment_euler_angles(dtheta=PI/100)
            mob.set_phi(45*DEGREES+0.5*np.sin(self.get_time()))
        frame.add_updater(frame_updater)
        surface3 = self.LineToSurface(mobius,u = [0,2*PI] , t_range = [-1,1,0.02],color = [GREEN_A,TEAL,GREEN_A])
        frame.clear_updaters()
        self.play(frame.animate.reorient(20, 65, 0),FadeOut(title),run_time=2)
        self.play(surface3.animate.shift(3*LEFT),run_time=3)
        self.wait(2)
        self.play(surface3.animate.shift(3*RIGHT),run_time=3)
        m1 = Group()
        for i in range(20):
            u = 0 + 2*PI/20*i
            m1.add(ParametricCurve(lambda v: mobius(u, v),
            t_range=[-1,1,0.02],color = [BLUE_A,TEAL,BLUE_A],
            stroke_width=2))
        line_t_1 = m1[0].copy()
        line_t_2 = m1[10].copy()
        self.play(FadeIn(m1),surface3.animate.set_opacity(0.5),run_time=2)
        self.wait()
        self.play(frame.animate.reorient(-40, 65, 0))
        self.play(AnimationGroup(FlashAlongPath(ParametricCurve(lambda v: mobius(0, v),t_range=[-2,2,0.02],color = [YELLOW],stroke_width=4)),
                                line_t_1.animate.set_color([RED_A,TEAL]),lag_ratio=0.2),
                  AnimationGroup(FlashAlongPath(ParametricCurve(lambda v: mobius(PI, v),t_range=[-2,2,0.02],color = [YELLOW],stroke_width=4)),
                                 line_t_2.animate.set_color([RED_A,TEAL]),lag_ratio=0.2),
                                 run_time=3)
        self.wait()
        self.play(FadeOut(m1))
        self.wait()
        self.play(line_t_1.animate.become(ParametricCurve(lambda v: mobius(0, v),
            t_range=[-4,4,0.02],color = [BLUE_A,TEAL,BLUE_A],
            stroke_width=2)),
                  line_t_2.animate.set_stroke(color=RED,width=4),
                  surface3.animate.set_opacity(0.2),run_time=2)
        mobius_strip = Group(surface3,line_t_2,line_t_1)
        self.play(frame.animate.reorient(0,0,0),
                  Rotate(mobius_strip,PI/2,RIGHT),run_time=3)
        self.play(Rotate(mobius_strip,-PI/2,OUT),run_time=2)
        
        self.wait()
        
        target = ParametricCurve(lambda v: mobius(0, v),
            t_range=[-1,1,0.02],color = [RED],
            stroke_width=4).rotate(PI/2,axis=RIGHT)
        target.rotate(-PI/2,axis=OUT)
        self.play(line_t_1.animate.become(target)
                  ,surface3.animate.set_opacity(0.6))
        ################沿着一条T剪开
        title = TitleText("沿T剪开").set_color(color=GREEN_A)
        self.play(mobius_strip.animate.shift(4*LEFT+DOWN)
                  ,run_time=2)
        scissor = ImageMobject("scissor.png")
        scissor.rotate_about_origin(135*DEGREES)
        scissor.scale(0.4)
        scissor.move_to(np.array([-2.5,1,0]))
        scissor_trace = TracedPath(scissor.get_center,stroke_color=YELLOW
                                   ,time_traced=2)
        self.add(scissor_trace)
        self.wait(2)
        self.play(scissor.animate.shift(LEFT)
                  ,Write(title)
                  ,run_time=3)
        self.remove(scissor_trace)
        self.wait()

        mobius_strip.add(mobius_strip[1].copy())
        self.add(mobius_strip[3])
        trapezoid1 = Group(ParametricSurface(trapezoid).set_color([GREEN_A,TEAL]).set_opacity(0.6),
                           ParametricCurve(lambda t:trapezoid(0,t)).set_color(RED),
                           Line(np.array([1.7,0,0]),np.array([2,1,0])).set_color(RED),
                           ParametricCurve(lambda t:trapezoid(1,t)).set_color(RED)).shift(0.5*DOWN)
        title1 = TitleText("再展平").set_color(color=BLUE_A)
        self.play(Transform(title,title1),
                  FadeOut(scissor),run_time=2)
        self.add(title1)
        self.remove(title)
        self.play(
            TransformFromCopy(mobius_strip,trapezoid1),run_time=4)
        self.add(trapezoid1)
        self.play(FadeOut(title1),run_time=2)

    #######################################################
    # #这是一个模拟线动成面的动画函数 t_range为v的范围
    def LineToSurface(self,
                      surface_func,
                      u : list,
                      surface_func1 = None,
                      n = 50,
                      t_range = [-1, 1, 0.02],
                      color : list = [BLUE_A,TEAL],
                      ):
        min_u,max_u = u
        tracker = ValueTracker(min_u)
        history = VGroup()  # 用于保存所有历史线段
        current_line = ParametricCurve(
            lambda v: surface_func(tracker.get_value(), v),
            t_range=t_range,
            color=BLUE,
            stroke_width=2
        ).set_color(color)
        current_line.add_updater(lambda mob: mob.become(ParametricCurve(
            lambda v: surface_func(tracker.get_value(), v),
            t_range=t_range,
            color=BLUE,
            stroke_width=2
        ).set_color(color)))
        if surface_func1 != None:
            surface = ParametricSurface(surface_func1,v_range=[min_u,max_u]
                                    ,u_range=[t_range[0],t_range[1]]).set_color(color=color)
        else:
            surface = ParametricSurface(surface_func,u_range=[min_u,max_u]
                                    ,v_range=[t_range[0],t_range[1]]).set_color(color=color)
        self.add(current_line,history)
        
        last_u = min_u
        delta_u = (max_u-min_u)/n
         # 将u分为100段，每段间隔delta_u
        def update_history(mob):
            nonlocal last_u
            current_u = tracker.get_value()
            if current_u >= last_u + delta_u:
                # 添加当前线段的静态副本到历史组
                new_line = current_line.copy().clear_updaters()
                history.add(new_line)
                last_u = current_u
        history.add_updater(update_history)
        # 播放动画，u从0变化到2π
        self.play(
            tracker.animate.set_value(max_u),
            run_time=6,
            rate_func=linear
        )
        current_line.clear_updaters()
        history.remove_updater(update_history)
        self.wait(1)
        self.play(FadeOut(current_line),
                  FadeOut(history),
                  FadeIn(surface))
        return surface
class Proof(Scene):
    def construct(self):
        surface = ParametricSurface(mobius ,u_range=(0,2*PI),v_range=(-1,1), color = [GREEN_A,TEAL,GREEN_A]).set_opacity(0.6)
        line_t_1 = ParametricCurve(lambda v: mobius(0, v),
            t_range=[-1,1,0.02],color = [RED],
            stroke_width=4)
        line_t_2 = ParametricCurve(lambda v: mobius(PI, v),
            t_range=[-1,1,0.02],color = [RED],
            stroke_width=4)
        mobius_strip = Group(surface,line_t_2,line_t_1).rotate(PI/2,axis=RIGHT,about_point=ORIGIN)
        mobius_strip.rotate(-PI/2,axis=OUT,about_point=ORIGIN)
        mobius_strip.shift(3*LEFT)
        trapezoid1 = Group(ParametricSurface(trapezoid).set_color([GREEN_A,TEAL]).set_opacity(0.6),
                           ParametricCurve(lambda t:trapezoid(0,t)).set_color(RED),
                           Line(np.array([1.7,0,0]),np.array([2,1,0])).set_color(RED),
                           ParametricCurve(lambda t:trapezoid(1,t)).set_color(RED)).shift(0.5*DOWN)
        self.add(mobius_strip,trapezoid1)
        self.wait()
        self.play(mobius_strip[0].animate.set_opacity(0.2).set_color(GREY_A)
                  ,trapezoid1[0].animate.set_opacity(0.2).set_color(GREY_A)
                  ,trapezoid1[1].animate.set_color([BLUE_A,TEAL])
                  ,trapezoid1[2].animate.set_color([YELLOW,TEAL])
                  ,trapezoid1[3].animate.set_color([BLUE_A,TEAL])
                  ,mobius_strip[1].animate.set_color([BLUE_A,TEAL])
                  ,mobius_strip[2].animate.set_color([YELLOW,TEAL])
                  ,run_time=4)
        ####################################
        #梯形的点
        point_A = get_special_dot(color=BLUE_A).move_to(trapezoid1[1].get_end())
        point_B = get_special_dot(color=GREEN_A).move_to(trapezoid1[1].get_start())
        point_B1 = get_special_dot(color=GREEN_A).move_to(trapezoid1[3].get_end())
        point_A1 = get_special_dot(color=BLUE_A).move_to(trapezoid1[3].get_start())
        point_u = get_special_dot(color=YELLOW).move_to(trapezoid1[2].get_start())
        point_v = get_special_dot(color=RED).move_to(trapezoid1[2].get_end())
        
        ####################################
        #莫比乌斯的点
        mobiuspoint_A = get_special_dot(color=BLUE_A).move_to(mobius_strip[1].get_end())
        mobiuspoint_B = get_special_dot(color=GREEN_A).move_to(mobius_strip[1].get_start())
        mobiuspoint_u = get_special_dot(color=YELLOW).move_to(mobius_strip[2].get_start())
        mobiuspoint_v = get_special_dot(color=RED).move_to(mobius_strip[2].get_end())
        
        ####################################
        #莫比乌斯的线
        mobius_curveD1 = ParametricCurve(lambda t:mobius(t,1),t_range=(0,PI,0.1)).set_color(GREEN)
        mobius_curveD2 = ParametricCurve(lambda t:mobius(t,-1),t_range=(PI,2*PI,0.1)).set_color(GREEN)
        mobius_curveH1 = ParametricCurve(lambda t:mobius(t,1),t_range=(PI,2*PI,0.1)).set_color(RED)
        mobius_curveH2 = ParametricCurve(lambda t:mobius(t,-1),t_range=(0,PI,0.1)).set_color(RED)
        ####################################
        #梯形的线
        trapezoid_curveD1 = Line(start=point_u.get_center(),end=point_B.get_center()).set_color(GREEN)
        trapezoid_curveD2 = Line(start=point_A1.get_center(),end=point_u.get_center()).set_color(GREEN)
        trapezoid_curveH1 = Line(start=point_A.get_center(),end=point_v.get_center()).set_color(RED)
        trapezoid_curveH2 = Line(start=point_v.get_center(),end=point_B1.get_center()).set_color(RED)
        ###########################################
        #合起来
        mobius_points = Group(mobiuspoint_A,mobiuspoint_B,mobiuspoint_u,mobiuspoint_v)
        trapezoid_points = Group(point_A,point_A1,point_B,point_B1,point_u,point_v)
        trapezoid_curves = Group(trapezoid_curveD1,trapezoid_curveD2,trapezoid_curveH1,trapezoid_curveH2)
        mobius_curves = Group(mobius_curveD1,mobius_curveD2,mobius_curveH1,mobius_curveH2).rotate(PI/2,axis=RIGHT,about_point=ORIGIN)
        mobius_curves.rotate(-PI/2,axis=OUT,about_point=ORIGIN)
        mobius_curves.shift(3*LEFT)
        self.play(FadeIn(mobius_points),FadeIn(trapezoid_points),run_time=3)
        self.wait()
        self.play(Write(trapezoid_curves[0])
                  ,Write(trapezoid_curves[1])
                  ,Write(trapezoid_curves[2])
                  ,Write(trapezoid_curves[3])
                  ,run_time=2)
        self.wait()
        self.play(TransformFromCopy(trapezoid_curves,mobius_curves),run_time=3)
        self.wait(2)
        frame = self.camera.frame
        frame.save_state()
        ############################################
        ##引入参数t与λ
        line0 = DashedLine(point_B.get_center(),4*UP)
        line1 = DashedLine(point_A.get_center(),np.array([0,point_A.get_center()[1],0])).set_color(BLUE_A)
        
        rectangle_pre = Polygon(point_B.get_center()
                                ,np.array([0,point_A.get_center()[1],0])
                                ,point_B1.get_center()
                                ,np.array([point_B1.get_center()[0],point_B.get_center()[1],0])).shift(2*UP)
        text = Group(Tex("t").next_to(line1,UP)
                     ,Tex("1").next_to(rectangle_pre,LEFT)
                     ,Tex(r"\lambda").next_to(rectangle_pre,DOWN))
        self.play(AnimationGroup(frame.animate.reorient(0,0,0,center=trapezoid1[0].get_center()).scale(0.7)
                  ,AnimationGroup(Write(line0)
                  ,Write(line1)
                  ,Write(text[0]))
                  ,lag_ratio=0.3),run_time=4)
        self.wait(2)
        self.play(AnimationGroup(FadeIn(rectangle_pre)
                                 ,Write(text[1])
                                 ,Write(text[2])),run_time=3)
        color_dict = {"Red" : RED, "Green" : GREEN}
        targeteq = Group(Tex(r"\lambda = Red + t",tex_to_color_map=color_dict).move_to(rectangle_pre),
                         Tex(r"\lambda = Red + t = Green - t",tex_to_color_map=color_dict).move_to(rectangle_pre))
        self.play(AnimationGroup(Transform(text[2],targeteq[0])
                                 ,AnimationGroup(FadeOut(rectangle_pre)
                                                 ,FadeOut(text[0])
                                                 ,FadeOut(text[1])
                                                 ,FadeOut(line0)
                                                 ,FadeOut(line1)),lag_ratio=0.4),run_time=3)
        self.wait(2)
        self.remove(text)
        self.add(targeteq[0])
        self.play(AnimationGroup(
            Transform(targeteq[0],targeteq[1]),
            frame.animate.restore()
                                 ,lag_ratio=0.3),run_time=4)
        self.remove(targeteq[0])
        self.add(targeteq[1])
        ############################################
        ##展示不等式
        ######################
        #演示红色
        self.play(FadeOut(targeteq[1]),run_time=2)
        frame.save_state()
        self.play(AnimationGroup(frame.animate.reorient(0,10,10)
                  ,FlashAlongPath(
                    mobius_curveH1.copy().set_color(BLUE),
                  )
                  ,FlashAlongPath(
                    mobius_curveH2.copy().set_color(BLUE),
                  )
                  ,lag_ratio=0.2),run_time=4)
        self.wait()
        eq = Group(Tex(r"Red \geq T_1 = \sqrt{1+t^2}",tex_to_color_map=color_dict).move_to(3*UP))
        self.play(AnimationGroup(
            AnimationGroup(frame.animate.restore()
                           ,TransformFromCopy(mobius_curveH1,trapezoid_curveH1)
                           ,TransformFromCopy(mobius_curveH2,trapezoid_curveH2)
                           ,TransformFromCopy(mobius_strip[1],trapezoid1[1])
                           ,run_time=5)))
        self.wait()
        self.play(
            AnimationGroup(FlashAlongPath(
                    trapezoid_curveH1.copy().set_color(BLUE),
                  ),
                  FlashAlongPath(
                    trapezoid_curveH2.copy().set_color(BLUE),
                  ),
                  FlashAlongPath(
                    trapezoid1[1].copy().set_color(RED),
                  ),run_time=2,lag_ratio=0.3
                  )
            ,Write(eq[0]),lag_ratio=0.5)
        ######################
        #演示绿色
        self.play(eq.animate.to_edge(UL),run_time=2)
        green_line1 = Line(mobiuspoint_v.get_center(),mobiuspoint_A.get_center()).set_color([BLUE,GREEN])
        green_line2 = Line(mobiuspoint_v.get_center(),mobiuspoint_B.get_center()).set_color([BLUE,GREEN])
        
        self.play(FlashAlongPath(green_line1.copy().set_color(YELLOW))
                  ,FlashAlongPath(green_line2.copy().set_color(YELLOW))
            ,run_time=3
        )
        self.play(FlashAlongPath(green_line1.copy().set_color(YELLOW))
                  ,FlashAlongPath(green_line2.copy().set_color(YELLOW))
            ,run_time=3
        )
        self.wait(2)
        frame.save_state()
        self.play(AnimationGroup(
            AnimationGroup(frame.animate.reorient(center=(mobiuspoint_v.get_center())),run_time=6),
            AnimationGroup(FadeOut(mobius_strip[0]),
            FadeOut(mobius_curves),
            FadeOut(trapezoid1)
            ,FadeOut(trapezoid_curves)
            ,FadeOut(trapezoid_points)
            ,FadeOut(eq[0])
            ,run_time=2)
            ,AnimationGroup(Write(green_line1),Write(green_line2),run_time=2)
            ,lag_ratio=0.2)
        )
        mobius_line1 = Line(mobiuspoint_B.get_center(),np.array([mobiuspoint_B.get_center()[0],
                                                                mobiuspoint_v.get_center()[1],0]))
        mobius_line2 = DashedLine(mobiuspoint_v.get_center(),np.array([mobiuspoint_B.get_center()[0],
                                                                mobiuspoint_v.get_center()[1],0]))

        para = Group(Tex(r"\sqrt{1+t^2}").next_to(mobius_strip[1],UP),Tex("k>1").next_to(mobius_line1,RIGHT))
        self.play(AnimationGroup(AnimationGroup(Write(mobius_line1),Write(mobius_line2),run_time=4),
                                 AnimationGroup(Write(para[0]),Write(para[1]),run_time=2),lag_ratio=0.3
                                 ))
        self.wait(2)
        green_line2.save_state()
        self.play(AnimationGroup(
            frame.animate.reorient(0,30,-10)
            ,Rotate(green_line2,angle=PI,about_edge=mobiuspoint_v.get_center(),axis=RIGHT)
            ,lag_ratio=0.3
        ),run_time=4)
        self.wait(2)
        target_eq = Tex(r">\sqrt{5+t^2}").next_to(mobius_line1,RIGHT,buff=2)
        self.play(AnimationGroup(
            AnimationGroup(
                frame.animate.reorient(0,0,0)
                ,green_line2.animate.restore()
            ,run_time=2),
            AnimationGroup(FadeOut(para[0]),
                           FadeOut(para[1]),
                           FadeOut(mobius_line1),
                           FadeOut(mobius_line2),
                           run_time=2),
            Write(target_eq),lag_ratio=0.5
        ))
        self.wait(2)
        self.play(AnimationGroup(
            frame.animate.restore(),
            FadeIn(mobius_strip[0]),
            FadeIn(mobius_curves)
            ,run_time=3,lag_ratio=0.2
        ))
        self.wait(2)
        eq.add(Tex(r"Green \geq V = \sqrt{5+t^2}",tex_to_color_map=color_dict).move_to(2*UP))
        self.play(AnimationGroup(
            AnimationGroup(
                Transform(target_eq,eq[1])
                ,run_time=3),
            AnimationGroup(
                FadeIn(trapezoid1)
                ,FadeIn(trapezoid_curves)
                ,FadeIn(trapezoid_points)
                ,FadeIn(eq[0]),run_time=2)
                ,lag_ratio=0.5
            ))
        self.remove(target_eq)
        self.add(eq[1])
        self.wait(2)
        self.play(AnimationGroup(
                AnimationGroup(eq[0].animate.move_to(UP)
                               ,eq[1].animate.move_to(DOWN),run_time=3)
                ,AnimationGroup(
                    FadeOut(mobius_strip),
                    FadeOut(trapezoid1),
                    FadeOut(mobius_points),
                    FadeOut(trapezoid_points),
                    FadeOut(mobius_curves),
                    FadeOut(trapezoid_curves),
                    FadeOut(green_line1),
                    FadeOut(green_line2)
                ,run_time=2),
            lag_ratio=0.2
        ))
class Proof1(Scene):
    def construct(self):
        frame = self.camera.frame
        color_dict = {"Red" : RED, "Green" : GREEN}
        eq = Group(
            Tex(r"Red \geq T_1 = \sqrt{1+t^2}",tex_to_color_map=color_dict).move_to(UP),
            Tex(r"Green \geq V = \sqrt{5+t^2}",tex_to_color_map=color_dict).move_to(DOWN),
            Tex(r"Red +t \geq \sqrt{1+t^2}+t",tex_to_color_map=color_dict).move_to(UP),
            Tex(r"Green -t\geq \sqrt{5+t^2}-t",tex_to_color_map=color_dict).move_to(DOWN)
        
        )
        self.add(eq[0],eq[1])
        self.play(Transform(eq[0],eq[2]),
                  Transform(eq[1],eq[3]),
                  run_time=2
                  )
        self.add(eq[2])
        self.remove(eq[0])
        self.add(eq[3])
        self.remove(eq[1])
        signal = Group(Tex(r"\lambda"))
        eq.add(Tex(r"\geqslant").rotate(PI/2).set_color(RED))
        eq.add(Tex(r"\geqslant").rotate(-PI/2).set_color((GREEN)))
        signal[0].shift(1.2*LEFT),
        eq[4].next_to(signal[0],UP)
        eq[5].next_to(signal[0],DOWN)
        self.play(AnimationGroup(frame.animate.shift(LEFT)
                                 ,AnimationGroup(
                                  FadeIn(signal[0]),   
                                  FadeIn(eq[4]),  
                                  FadeIn(eq[5]),  
                                 ),lag_ratio=0.3),run_time= 4
    )
        eqred = Group(eq[2],eq[4])
        eqgreen = Group(eq[3],eq[5])
        lefteq = Group(Tex(r"\sqrt{5+t^2}-t"),Tex(r"\leqslant").set_color(GREEN)).arrange(RIGHT)
        righteq = Group(Tex(r"\sqrt{1+t^2}+t"),Tex(r"\geqslant").set_color(RED)).arrange(LEFT)
        lefteq.next_to(signal[0],LEFT)
        righteq.next_to(signal[0],RIGHT)
        self.play(Transform(eqgreen,lefteq),
                  Transform(eqred,righteq),run_time=3)
        self.wait(2)
        self.add(lefteq,righteq)
        self.remove(eqgreen,eqred)
        axe = Axes((0, 5), (0, 5),height=2.5,width=2.5,).move_to(signal[0].get_center()+2*UP+2*RIGHT)
        axe1 = Axes((0, 5), (0, 5),height=2.5,width=2.5).shift(signal[0].get_center()+2*UP+2*LEFT)
        red_graph = axe.get_graph(
            function=lambda x : np.sqrt(1+x**2)+x,
            x_range = [0,2.5],
            color=RED
        )
        green_graph = axe1.get_graph(
            function=lambda x : np.sqrt(5+x**2)-x,
            x_range = [0,5],
            color=GREEN
        )
        self.add(axe, axe1 , red_graph,green_graph)
        self.play(AnimationGroup(
            AnimationGroup(lefteq.animate.shift(DOWN),
                  righteq.animate.shift(DOWN),
                  signal[0].animate.shift(DOWN),run_time=2),
            AnimationGroup(Write(axe),
                                 Write(axe1),
                                 Write(red_graph),
                                 Write(green_graph),run_time=3)
            ,lag_ratio=0.2
            ))
        self.wait(2)
        ##########
        #将函数图像合在一起
        self.play(axe.animate.shift(2*LEFT),
                  red_graph.animate.shift(2*LEFT),
                  axe1.animate.shift(2*RIGHT),
                  green_graph.animate.shift(2*RIGHT),
                  run_time=3,rate_func=rush_into)
        self.wait(2)
        min_graph = axe1.get_graph(
            function=lambda x : np.sqrt(3),
            x_range = [0,5],
            color=TEAL_A
        )
        min_label = axe.get_graph_label(
            graph=min_graph, label=r"y= \sqrt{3}"
        )
        self.play(AnimationGroup(
            AnimationGroup(
            red_graph.animate.become(axe.get_graph(
            function=lambda x : np.sqrt(1+x**2)+x,
            x_range = [1/np.sqrt(3),2.5],
            color=RED
        )),
            green_graph.animate.become(
            axe1.get_graph(
            function=lambda x : np.sqrt(5+x**2)-x,
            x_range = [0,1/np.sqrt(3)],
            color=GREEN
        )),run_time=2),
            AnimationGroup(
                Write(min_graph),Write(min_label),run_time=2
            ),lag_ratio=0.3)
        )
        self.wait(2)
        target_eq = Tex(r"\lambda \geqslant \sqrt{3}").set_color([BLUE_A,TEAL]).move_to(LEFT+0.5*DOWN)
        self.play(AnimationGroup(
            AnimationGroup(lefteq.animate.shift(DOWN),
                  righteq.animate.shift(DOWN),
                  signal[0].animate.shift(DOWN),run_time=2),
            AnimationGroup(Write(target_eq),run_time=1.5)
            ,lag_ratio=0.3
            ))
class addtion1(Scene):
    def construct(self) -> None:
        frame = self.camera.frame
        self.play(frame.animate.reorient(0, 45, 0))
        
        self.LineToSurface(mobius,u = [0,2*PI] , t_range = [-1,1,0.02],color = [GREEN_A,TEAL,GREEN_A])
        frame.clear_updaters()
        self.wait()
    def LineToSurface(self,
                      surface_func,
                      u : list,
                      surface_func1 = None,
                      n = 10,
                      t_range = [-1, 1, 0.02],
                      color : list = [BLUE_A,TEAL],
                      ):

        min_u,max_u = u
        tracker = ValueTracker(min_u)
        history = VGroup()  # 用于保存所有历史线段
        current_line = ParametricCurve(
            lambda v: surface_func(tracker.get_value(), v),
            t_range=t_range,
            color=BLUE,
            stroke_width=2
        ).shift(RIGHT).set_color(color)
        current_line.add_updater(lambda mob: mob.become(ParametricCurve(
            lambda v: surface_func(tracker.get_value(), v),
            t_range=t_range,
            color=BLUE,
            stroke_width=2
        ).shift(RIGHT).set_color(color)))
        point1 = get_special_dot(color = GREEN)
        point1.add_updater(lambda mob : mob.move_to(current_line.get_start()))
        point2 = get_special_dot(color = BLUE)
        point2.add_updater(lambda mob : mob.move_to(current_line.get_end()))
        trace1 , trace2 = TracedPath(point1.get_center,stroke_color=GREEN),TracedPath(point2.get_center,stroke_color=BLUE)
        self.add(current_line,point1,point2,trace1,trace2,history)
        last_u = min_u
        delta_u = (max_u-min_u)/n
         # 将u分为100段，每段间隔delta_u
        def update_history(mob):
            nonlocal last_u
            current_u = tracker.get_value()
            if current_u >= last_u + delta_u:
                # 添加当前线段的静态副本到历史组
                new_line = current_line.copy().clear_updaters()
                self.add(new_line)
                last_u = current_u
        
        # 播放动画，u从0变化到2π
        history.add_updater(update_history)
        self.play(
            tracker.animate.set_value(max_u),
            run_time=6,
            rate_func=double_smooth
            
        )
        current_line.clear_updaters()
        history.remove_updater(update_history)
if __name__ == "__main__":
    os.system("manimgl {} addtion1 -c black --uhd -w".format(__file__))