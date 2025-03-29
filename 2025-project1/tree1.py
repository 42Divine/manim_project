from manimlib import *
from os import system
import cv2
import numpy as np
def generate_situation_point_set(set1,set_old=set(),func=None):
    if func is None:
        func = set()
    set_new=set()
    for i in set1:
        for j in func:
            set_new.add((i[0]+j[0],i[1]+j[1]))
            set_new.add((i[0]-j[0],i[1]-j[1]))
    return set_new-set_old
def generate_situation_points_set(set1,func=None):
    if func is None:
        func = set()
    
    set_new=set()
    for i in set1:
        
        for j in func:
            set_new.add((i[0]+j[0],i[1]+j[1]))
    return set_new
def trans_image(scene,img_path 
                ,location=ORIGIN
                ,brightness_threshold = 30
                ,delta_x=ORIGIN):
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
        
      # 创建图片对象并获取其大小
    image = ImageMobject(img_path)
    image.scale(1.5) 
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
class flash_line(FlashAround):
    def get_path(self, mobject: Mobject) -> Underline:
        return Line(start=mobject.get_start(), end=mobject.get_end(), buff=self.buff)
class ColorableGrid:
    class CoordinateGroup:
        """内嵌的坐标集类"""
        def __init__(self, grid, coordinates, color=BLUE,add=True):
            self.grid = grid          # 引用外部网格
            self.coordinates = set(coordinates)
            self.color = color
            self.original_coordinates = set(coordinates)
            # 将自己添加到网格的管理中
            if add:
                self.squares=self.grid.add_coordinate_group(self)
                
            else:
                self.squares=VGroup()
                for i in self.coordinates:
                    self.squares.add(self.grid.cells[(i[0],i[1])])
            #self.squares.set_color_by_gradient(self.color)
        def translate(self, vector, animate=True, run_time=1,color=None):
            """平移操作"""
            dr, dc = vector[0],vector[1]
            new_coordinates = {(r + dr, c + dc) for r, c in self.coordinates}
            
            # 检查新位置是否有效
            if not self.is_valid_position(new_coordinates):
                print("警告：平移后将超出网格范围")
                return False
                
            if animate:

                self.grid.animate_translation(self, vector, run_time,color=color)
            else:
                # 清除原位置
                self.grid.clear_cells(self.coordinates)
                # 更新位置
                self.coordinates = new_coordinates
                # 在新位置上色
                self.grid.color_cells(self.coordinates, self.color, animate=False)
            
            return True
        def translates(self,vector_set,run_time=1,color=None):
            self.grid.animate_translations(self,vector_set,run_time=run_time,color=color)
        def is_valid_position(self, coordinates=None):
            """检查位置是否有效"""
            if coordinates is None:
                coordinates = self.coordinates
            return all(0 <= r < self.grid.n and 0 <= c < self.grid.n 
                      for r, c in coordinates)
        
        def get_coordinates(self):
            return list(self.coordinates)
            
        def remove(self):
            """从网格中移除自己"""
            self.grid.remove_coordinate_group(self)
            self.grid.clear_cells(self.coordinates)
        def get_center(self):
            """获取指定格子的中心坐标"""
            return self.squares.get_center()
        def get_center_with_direction(self,direction=RIGHT):
            return self.squares.get_bounding_box_point(direction)
    def __init__(self, scene, size=8, n=16, cell_color=WHITE, center_offset=ORIGIN,add=True,fill_opacity=0.8):
        self.scene = scene
        self.size = size
        self.n = n
        self.cell_size = size / n
        self.cell_color = cell_color
        self.center_offset = center_offset  # 新增：存储网格中心的偏移量
        self.fill_opacity=fill_opacity
        # 存储网格和方块的数据结构
        self.grid = VGroup()
        self.cells = {}  # 存储已经上色的方块
        self.cells_group=VGroup()
        # 新增：存储和管理坐标集
        self.coordinate_groups = []  # 存储所有的坐标集
        self.group_count = 0        # 坐标集计数
        self.add=add
        self._create_grid()
    
    def create_coordinate_group(self, coordinates, color=BLUE,add=True):
        """创建新的坐标集"""
        return self.CoordinateGroup(self, coordinates, color,add)
    
    def add_coordinate_group(self, group):
        """添加坐标集到管理列表"""
        self.coordinate_groups.append(group)
        self.group_count += 1
        # 显示新的坐标集
        squares=self.color_cells(group.coordinates, group.color)
        return squares

    def remove_coordinate_group(self, group):
        """移除坐标集"""
        if group in self.coordinate_groups:
            self.coordinate_groups.remove(group)
            self.group_count -= 1
    
    def get_group_count(self):
        """获取当前坐标集数量"""
        return self.group_count
    def animate_translations(self,coord_group,vector_set,run_time=1,color=None):
        animations_line=[]
        animations=[]
        for i in range(len(vector_set)):
            animations1,animations2=self.animate_translation(coord_group,vector_set[i],run_time=run_time,animate=False,color=color)
            for j in range(len(animations1)):
                animations_line.append(animations1[j])
            for j in range(len(animations2)):
                animations.append(animations2[j])
        self.scene.play(*animations_line,run_time=run_time)
        self.scene.play(*animations,run_time=run_time)
    def animate_translation(self, coord_group, vector, run_time=1,direction=RIGHT,animate=True,color=None):
        """动画展示坐标群的平移"""
        dr, dc = vector[0],vector[1]
        animations = []
        squares = {}
        if color is None:
            for row, col in coord_group.get_coordinates():
                squares[(row,col)]=self.cells[(row,col)].copy()
        else:
            color=color
        # 为每个坐标创建方块
            for row, col in coord_group.get_coordinates():
                square = Square(
                side_length=self.cell_size,
                fill_opacity=self.fill_opacity,
                fill_color=color,
                stroke_width=0
            ).set_color_by_gradient(color)
                square.move_to(self.get_cell_center(row, col))
                squares[(row, col)] = square
            
        # 计算目标位置
        for (row, col), square in squares.items():
            target_center = self.get_cell_center(row + dr, col + dc)
            animations.append(square.animate.move_to(target_center))
            cell_key=(row+dr,col+dc)
            self.cells[cell_key] = square
            self.cells_group.add(square)
        direction=np.array([dc,dr,0])
        start_point=coord_group.get_center()
        end_point=coord_group.get_center()+np.array([dc,dr,0])*self.cell_size
        vector1 = Arrow(
            start_point, 
            end_point, 
            buff=0.1
        )

        if type(coord_group.color) is list:
            vector1.set_color_by_gradient([coord_group.color[0], coord_group.color[1]])
        else:
            vector1.set_color_by_gradient([coord_group.color, YELLOW])
        # 展示路径
        animations_line=[]
        animations_line.append(ShowCreation(vector1))
        animations_line.append(flash_line(vector1))
        # 播放动画
        animations.append(Uncreate(vector1))
        if animate:
            self.scene.play(*animations_line,run_time=run_time)
            self.scene.play(*animations, run_time=run_time)        
        # 清理旧的方块
        #for square in squares.values():
        #    self.scene.remove(square)
        
        # 更新坐标集的位置
        #coord_gro = {(r + dr, c + dc) 
        #                        for r, c in coord_group.coordinates}
        # 在新位置创建方块
        #self.color_cells(coord_gro, coord_group.color, animate=False)
        return animations_line,animations
    def hide_grid_lines(self, background_color=BLACK, animate=True, run_time=1):
        """
        隐藏方格线，将线条颜色设置为背景色。
        
        :param background_color: 背景色，默认为黑色
        :param animate: 是否使用动画
        :param run_time: 动画持续时间
        """
        animations = []
        for line in self.grid:
            if animate:
                animations.append(line.animate.set_stroke(color=background_color))
            else:
                line.set_stroke(color=background_color)
        
        if animate and animations:
            self.scene.play(*animations, run_time=run_time)
    def _create_grid(self):
        """创建基础网格"""
        # 创建横向和纵向的线
        for i in range(self.n + 1):
            pos = -self.size/2 + i * self.cell_size
            
            # 创建横线
            h_line = Line(
                start=np.array([-self.size/2, pos, 0]) + self.center_offset,
                end=np.array([self.size/2, pos, 0]) + self.center_offset,
                stroke_width=(3 if i in [0, self.n] else 1)
            )
            
            # 创建纵线
            v_line = Line(
                start=np.array([pos, -self.size/2, 0]) + self.center_offset,
                end=np.array([pos, self.size/2, 0]) + self.center_offset,
                stroke_width=(3 if i in [0, self.n] else 1)
            )
            
            self.grid.add(h_line, v_line)
        
        # 添加到场景
        if self.add:
            self.scene.add(self.grid)
    
    def get_cell_center(self, row, col):
        """获取指定格子的中心坐标"""
        x = -self.size/2 + (col + 0.5) * self.cell_size
        y = -self.size/2 + (row + 0.5) * self.cell_size
        return np.array([x, y, 0]) + self.center_offset
    
    def color_cell(self, row, col, color=BLUE, animate=True):
        """为指定的格子上色"""
        # 检查坐标是否有效
        if not (0 <= row < self.n and 0 <= col < self.n):
            raise ValueError(f"Invalid cell position: ({row}, {col})")
            
        # 如果该位置已经有方块，先移除
        cell_key = (row, col)
        if cell_key in self.cells:
            self.scene.remove(self.cells[cell_key])
            
        # 创建新的方块
        square = Square(
            side_length=self.cell_size,
            fill_opacity=self.fill_opacity,
            fill_color=color,
            stroke_width=0
        )
        square.move_to(self.get_cell_center(row, col))
        
        # 存储方块引用
        self.cells[cell_key] = square
        self.cells_group.add(square)
        # 添加到场景
        if animate:
            if self.add:
                self.scene.play(FadeIn(square))
        else:
            if self.add:
                self.scene.add(square)
    def clear_outside_area(self, area, animate=True):
        """
        清除指定区域外的所有颜色。
        
        :param area: 指定的区域，格式为 [(row1, col1), (row2, col2), ...]
        :param animate: 是否使用动画
        """
        # 将区域转换为集合以便快速查找
        area_set = set(area)
        
        # 找出需要清除的格子
        cells_to_clear = [cell for cell in self.cells if cell not in area_set]
        
        # 清除这些格子
        self.clear_cells(cells_to_clear, animate=animate)
    def clear_cells(self, lis, animate=True):
        """清除指定格子的颜色"""
        if animate:
        # 创建要移除的方块组
            squares_to_remove = VGroup()
            for row, col in lis:
                cell_key = (row, col)
                if cell_key in self.cells:
                    squares_to_remove.add(self.cells[cell_key])
                    
                    self.cells_group.remove(self.cells[cell_key])
                    del self.cells[cell_key]
        # 如果有方块需要移除，执行动画
            if len(squares_to_remove) > 0:
                self.scene.play(FadeOut(squares_to_remove))
            
        # 从cells字典中移除这些方块
        #    for row, col in lis:
        #        cell_key = (row, col)
        #        if cell_key in self.cells:
        #            del self.cells[cell_key]
        #            self.cells_group.remove(self.cells[cell_key])
        else:
        # 不需要动画时直接移除
            for row, col in lis:
                cell_key = (row, col)
                if cell_key in self.cells:
                    self.scene.remove(self.cells[cell_key])
                    del self.cells[cell_key]
                    self.cells_group.remove(self.cells[cell_key])
    def clear_grid(self,animate=True):
        if animate:
            vg=VGroup()
            vg.add(self.cells_group)
            vg.add(self.grid)
            self.scene.play(FadeOut(vg))
        else:
            self.scene.remove(self.cells_group)
            self.scene.remove(self.grid)
        
    def color_cells(self, positions, color=BLUE, animate=True):
        """为多个格子同时上色"""
        if animate:
            squares = VGroup()
            for row, col in positions:
                square = Square(
                    side_length=self.cell_size,
                    fill_opacity=self.fill_opacity,
                    fill_color=color,
                    stroke_width=0
                )
                square.move_to(self.get_cell_center(row, col))
                self.cells[(row, col)] = square
                squares.add(square)
                self.cells_group.add(square)
            if self.add:
                self.scene.play(FadeIn(squares))
            return squares
        else:
            for row, col in positions:
                self.color_cell(row, col, color, animate=False)
        
    def add_grid(self,condition=None):
        self.add=True
        if self.add:
            if condition is None:
                self.scene.add(self.grid)
                self.scene.add(self.cells_group)
            elif condition==0:
                self.scene.play(Write(self.grid))
                self.scene.play(FadeIn(self.cells_group))
class CoordinateGroup:
    def __init__(self, coordinates, color=BLUE):
        self.coordinates = set(coordinates)
        self.color = color
        self.original_coordinates = set(coordinates)
    
    def translate(self, vector, grid, steps=20):
        """
        动画平移
        vector: (delta_row, delta_col) 平移向量
        grid: ColorableGrid实例，用于更新显示
        steps: 动画的步数
        """
        dr, dc = vector
        # 计算每步的增量
        step_r = dr / steps
        step_c = dc / steps
        
        # 保存起始位置
        start_coords = set(self.coordinates)
        
        # 逐步移动
        for i in range(steps + 1):
            # 计算当前步骤的偏移量
            current_dr = int(step_r * i)
            current_dc = int(step_c * i)
            
            # 更新坐标
            self.coordinates = {(r + current_dr, c + current_dc) 
                              for r, c in start_coords}
            
            # 更新显示
            yield self.coordinates
            
        # 确保最终位置精确
        self.coordinates = {(r + dr, c + dc) for r, c in start_coords}
        yield self.coordinates
    
    def get_coordinates(self):
        return list(self.coordinates)
    
    def is_valid_position(self, grid_size):
        return all(0 <= r < grid_size and 0 <= c < grid_size 
                  for r, c in self.coordinates)
# 示例使用
class tree1(Scene):
    def construct(self):
        grid = ColorableGrid(self,size=50,n=100)
        self.camera.frame.scale(1.2)
        set1=set([(50,50)])
        list1=generate_situation_points_set(set1,func=[(0,0),(1,0),(0,1),(-1,0),(0,-1)])
        set2=generate_situation_point_set(set1,list1,func=[(-1,2),(2,1)])
        list2=generate_situation_points_set(set2,func=[(0,0),(1,0),(0,1),(-1,0),(0,-1)])
        set3=generate_situation_point_set(set2,list2,func=[(-1,2),(2,1)])
        list3=generate_situation_points_set(set3,func=[(0,0),(1,0),(0,1),(-1,0),(0,-1)])
        set4=generate_situation_point_set(set3,list3,func=[(-1,2),(2,1)])
        list4=generate_situation_points_set(set4,func=[(0,0),(1,0),(0,1),(-1,0),(0,-1)])
        set5=generate_situation_point_set(set4,list4,func=[(-1,2),(2,1)])
        list5=generate_situation_points_set(set5,func=[(0,0),(1,0),(0,1),(-1,0),(0,-1)])
        set6=generate_situation_point_set(set5,list5,func=[(-1,2),(2,1)])
        list6=generate_situation_points_set(set6,func=[(0,0),(1,0),(0,1),(-1,0),(0,-1)])
        set7=generate_situation_point_set(set6,list6,func=[(-1,2),(2,1)])
        list7=generate_situation_points_set(set7,func=[(0,0),(1,0),(0,1),(-1,0),(0,-1)])
        set8=generate_situation_point_set(set7,list7,func=[(-1,2),(2,1)])
        set_all=set1|set2|set3|set4|set5|set6|set7|set8
        
        GROUP1=grid.create_coordinate_group(set_all,color=GREEN)
        set_region=set()
        self.wait(1)
        for i in range(16):
            for j in range(16):
                set_region.add((42+i,41+j))
        # 找出需要清除的格子
        grid.clear_outside_area(set_region,animate=True)
        self.wait(2)
        GROUP2=grid.create_coordinate_group(set_region-set_all,color=BLUE_C)
        self.wait(2)
        grid.hide_grid_lines(background_color=BLACK, animate=True, run_time=1)
        self.wait(2)
        self.play(self.camera.frame.animate.scale(1.2).shift(DOWN+4*RIGHT))
        self.wait(2)
        image2=trans_image(self,"images/green_yellow.png",location=7.9*RIGHT,brightness_threshold=10)
        self.wait(1)
class tree1_1(Scene):
    def construct(self):
        grid = ColorableGrid(self,size=1.5,n=3,center_offset=UP*2+LEFT*5)
        grid.create_coordinate_group(set([(1,1),(1,0),(0,1),(1,2),(2,1)]),color=GREEN)
        
            # 创建多个坐标集
        set1=set([(3,3)])
        list1=generate_situation_points_set(set1,func=[(0,0),(1,0),(0,1),(-1,0),(0,-1)])
        set2=generate_situation_point_set(set1,list1,func=[(-1,2),(2,1)])
        list2=generate_situation_points_set(set2,func=[(0,0),(1,0),(0,1),(-1,0),(0,-1)])
        self.wait(2)
        grid_solution=ColorableGrid(self,size=3.5,n=7,center_offset=DOWN+LEFT*5,add=False)
        grid_solution.create_coordinate_group(list1, color=GREEN)
        grid_solution.create_coordinate_group(list2, color=BLUE)
        grid_solution.add_grid(condition=0)
        self.wait(2)
        grid1=ColorableGrid(self,size=1.5,n=3,center_offset=UP*2+LEFT*1.7,add=False)
        grid1.create_coordinate_group([(1,1),(1,0),(0,1),(1,2)],color=GREEN)
        grid1.add_grid(condition=0)
        self.wait(2)
        grid2=ColorableGrid(self,size=2,n=4,center_offset=UP*2+RIGHT*1.7,add=False)
        grid2.create_coordinate_group([(2,2),(3,2),(2,3),(2,1),(1,2),(0,2)],color=GREEN)
        grid2.add_grid()
        self.wait(2)
        grid3=ColorableGrid(self,size=2,n=4,center_offset=UP*2+RIGHT*5,add=False)
        grid3.create_coordinate_group([(2,2),(3,2),(2,3),(2,1),(1,2),(0,2),(0,1)],color=GREEN)
        grid3.add_grid()
        self.wait(2)
        
class tree1_2(Scene):
    def construct(self):
        self.wait(2)
        grid = ColorableGrid(self,size=5,n=10,center_offset=RIGHT*3,add=False)
        grid.add_grid(condition=0)
        i=(8,-8)
        l_shape = grid.create_coordinate_group([i,(i[0]+1,i[1]),(i[0]-1,i[1]),(i[0],i[1]+1),(i[0],i[1]-1)], color=BLUE)
        self.wait(2)
        l_shape.translate(vector=np.array([-3,12,0]),color=GREEN)
        self.wait(2)
        l_shape.translates(vector_set=[np.array([-4,14,0]),np.array([-2,10,0]),np.array([-1,13,0]),np.array([-5,11,0])])
        self.wait(2)
        grid.clear_outside_area(set([(0,0)]),animate=True)
        self.wait(2)
        l_shape1=grid.create_coordinate_group([(8,-8),(6,-8),(7,-7),(7,-9),
                                               (6,-9),(8,-9),(6,-7),(8,-7)],color=GREEN)
        self.wait(2)
        l_shape2=grid.create_coordinate_group([(2,-8),(0,-8),(1,-7),(1,-9),
                                               ],color=BLUE)
        self.wait(2)
        grid.clear_outside_area(set([(0,0)]),animate=True)
        self.wait(2)
        l_shape2=grid.create_coordinate_group([(6,-8),(4,-8),(5,-7),(5,-9),(5,-8),(3,-8)
                                               ],color=BLUE)
        self.wait(2)
class tree1_3(Scene):
    def construct(self):
        grid = ColorableGrid(self,size=5,n=10,center_offset=RIGHT*3,add=False,fill_opacity=0.9)
        grid.add_grid(condition=0)
        text=Text("基本图形-方块",font="KaiTi",color=WHITE)
        text.move_to(UP*3+LEFT*3)
        self.play(Write(text))
        chunk1=grid.create_coordinate_group([(4,-8)],color=GREEN)
        self.wait(2)
        self.play(FadeOut(text),run_time=0.5)
        text=Text("平移",font="KaiTi",color=WHITE)
        text.move_to(UP*3+LEFT*3)
        self.play(Write(text))
        chunk1.translate(vector=np.array([1,12,0]),color=GREEN)
        self.wait(2)
        chunk1.translate(vector=np.array([1,13,0]),color=GREEN)
        chunk1.translate(vector=np.array([1,11,0]),color=GREEN)
        chunk2=grid.create_coordinate_group([(5,4),(5,3),(5,5)],color=GREEN,add=False)
        self.wait(2)
        self.play(FadeOut(text),run_time=0.5)
        text=Text("再平移",font="KaiTi",color=WHITE)
        text.move_to(UP*3+LEFT*3)
        self.play(Write(text))
        chunk2.translate(vector=np.array([1,0,0]),color=GREEN)
        chunk2.translate(vector=np.array([-1,0,0]),color=GREEN)
        self.wait(2)
        self.play(FlashAround(grid.cells[(5,4)]))
        self.wait(2)
        chunk3=grid.create_coordinate_group([(5,4),(5,3),(5,5),(6,3),(6,4),(6,5),(4,3),(4,4),(4,5)],color=GREEN,add=False)
        chunk3.translates(vector_set=[np.array([3,0,0]),np.array([-3,0,0]),np.array([0,3,0]),np.array([0,-3,0])])
        self.wait(1)
        chunk3.translates(vector_set=[np.array([3,3,0]),np.array([-3,3,0]),np.array([-3,-3,0]),np.array([3,-3,0])],color=GREEN)
class tree1_4(Scene):
    def construct(self):
        grid = ColorableGrid(self,size=6.5,n=13,center_offset=RIGHT*3,add=False,fill_opacity=0.9)
        grid.add_grid(condition=0)
        self.wait(2)
        text=Text("待解图形",font="KaiTi").set_color_by_gradient(GREY,BLUE)
        text.move_to(UP*3+LEFT*3)
        self.play(Write(text))
        i=(9,-5)
        l_shape = grid.create_coordinate_group([i,(i[0]+1,i[1]),(i[0]-1,i[1]),(i[0],i[1]-1),(i[0],i[1]+1),(i[0]-2,i[1]),(i[0]-2,i[1]-1)], color=GOLD)
        self.wait(2)
        l_shape.translate(vector=(-3,11),color=[BLUE,BLUE_A,BLUE_B])
        text1=Text("求解向量：",font="KaiTi").set_color_by_gradient(GREY,BLUE)
        text1.move_to(DOWN*0.4+LEFT*3)
        vec_set=VGroup(Tex("vec1:").set_color_by_gradient(GREY,BLUE),Tex("vec2:").set_color_by_gradient(GREY,BLUE)).arrange(DOWN,buff=0.5)
        vec_set.move_to(DOWN*1.6+LEFT*4)
        self.play(Write(text1),Write(vec_set[0]))
        self.wait(2)
        i=(6,6)
        l_shape1=grid.create_coordinate_group([i,(i[0]+1,i[1]),(i[0]-1,i[1]),(i[0],i[1]-1),(i[0],i[1]+1),(i[0]-2,i[1]),(i[0]-2,i[1]-1)],color=[BLUE,BLUE_A,BLUE_B],add=False)
        vec=Tex("(2,1)")
        vec.next_to(vec_set[0],RIGHT,buff=0.5)
        self.play(Write(vec))
        l_shape1.translate(vector=(1,2),color=[GREEN,GREEN_A,GREEN_B])
        self.wait(0.5)
        l_shape1.translate(vector=(-1,-2),color=[GREEN,GREEN_A,GREEN_B])
        self.wait(2)
        k=[i,(i[0]+1,i[1]),(i[0]-1,i[1]),(i[0],i[1]-1),(i[0],i[1]+1),(i[0]-2,i[1]),(i[0]-2,i[1]-1)]
        l_shape2_set=[]
        for i in range(len(k)):
            l_shape2_set.append(k[i])
            l_shape2_set.append((k[i][0]+1,k[i][1]+2))
            l_shape2_set.append((k[i][0]-1,k[i][1]-2))
        l_shape2=grid.create_coordinate_group(l_shape2_set,add=False)
        vec1=Tex("(-1,3)")
        vec1.next_to(vec_set[1],RIGHT,buff=0.5)
        self.play(Write(vec_set[1]),Write(vec1))
        self.wait(1)
        l_shape2.translate(vector=(3,-1),color=[YELLOW,YELLOW_A,YELLOW_B])
        l_shape2.translate(vector=(-3,1),color=[YELLOW,YELLOW_A,YELLOW_B])
        self.wait(2)
        vg=VGroup()
        for i in range(len(l_shape2_set)):
            vg.add(grid.cells[l_shape2_set[i]])
            vg.add(grid.cells[(l_shape2_set[i][0]+3,l_shape2_set[i][1]-1)])
            vg.add(grid.cells[(l_shape2_set[i][0]-3,l_shape2_set[i][1]+1)])
        vg.add(grid.grid)
        vg1=VGroup(text1,vec_set,vec1,vec)
        arc = ArcBetweenPoints(
            vg1.get_center(), 
            3*RIGHT+2.22*UP,
            angle=PI,        # 控制圆弧的弧度（PI表示半圆）
            stroke_width=0   # 隐藏路径线条
        )
        # 添加物体到场景
        # 创建并运行动画
        self.play(FadeOut(vg,RIGHT),MoveAlongPath(vg1, arc),run_time=3)
class tree1_4_addition(Scene):
    def construct(self):
        grid = ColorableGrid(self,size=6.5,n=13,center_offset=3*RIGHT)
        grid.hide_grid_lines()
        
        text=Text("待解图形",font="KaiTi").set_color_by_gradient(GREY,BLUE)
        text.move_to(UP*3+LEFT*3)
        self.play(Write(text))
        i=(9,-5)
        l_shape = grid.create_coordinate_group([i,(i[0]+1,i[1]),(i[0]-1,i[1]),(i[0],i[1]-1),(i[0],i[1]+1),(i[0]-2,i[1]),(i[0]-2,i[1]-1)], color=GOLD)
        
        text1=Text("求解向量：",font="KaiTi").set_color_by_gradient(GREY,BLUE)
        text1.move_to(DOWN*0.4+LEFT*3)
        def generate_vec_set(vec1,vec2,location=DOWN*1.6+LEFT*4):
            vec_set=VGroup(Tex("vec1:").set_color_by_gradient(GREY,BLUE),Tex("vec2:").set_color_by_gradient(GREY,BLUE)).arrange(DOWN,buff=0.5)
            vec_set.move_to(location)
            vec=Tex(str(vec1),font="KaiTi")
            vec.next_to(vec_set[0],RIGHT,buff=0.5)
            vec1=Tex(str(vec2),font="KaiTi")
            vec1.next_to(vec_set[1],RIGHT,buff=0.5)
            return VGroup(vec_set,vec,vec1)
        vec_set=generate_vec_set((2,1),(-1,3))
        self.play(Write(text1))
        self.play(Write(vec_set))
        vg1=VGroup(text1,vec_set)
        arc = ArcBetweenPoints(
            vg1.get_center(), 
            3*RIGHT+2.22*UP,
            angle=PI,        # 控制圆弧的弧度（PI表示半圆）
            stroke_width=0   # 隐藏路径线条
        )
        self.play(MoveAlongPath(vg1, arc),run_time=1)
        self.play(vec_set.animate.next_to(l_shape.squares,3.3*RIGHT,buff=1))
        self.wait(2)
        
        i1=(4,-5)
        l_shape1=grid.create_coordinate_group([i1,(i1[0]+1,i1[1]),(i1[0]-1,i1[1]),(i1[0],i1[1]-1),(i1[0],i1[1]+1)], color=BLUE_A)
        vec_set1=generate_vec_set((2,-1),(1,2),l_shape1.squares.get_center())
        self.play(vec_set1.animate.next_to(l_shape1.squares,3.3*RIGHT,buff=1))
        i2=(0,-5)
        l_shape2=grid.create_coordinate_group([i2,(i2[0]+1,i2[1]),(i2[0],i2[1]-1),(i2[0],i2[1]+1)], color=GREEN_A)
        vec_set2=generate_vec_set((2,1),(-2,1),l_shape2.squares.get_center())
        self.play(vec_set2.animate.next_to(l_shape2.squares,3.3*RIGHT,buff=1))
        self.wait(2)
        self.camera.frame.add_updater(lambda m: m.shift(0.02*DOWN))
        i3=(-4,-5)
        l_shape3=grid.create_coordinate_group([i3,(i3[0]+1,i3[1]),(i3[0]-1,i3[1]),(i3[0]-2,i3[1]),(i3[0],i3[1]-1),(i3[0],i3[1]+1)], color=GREEN_A)
        vec_set3=generate_vec_set("None","None",l_shape3.squares.get_center())
        self.play(vec_set3.animate.next_to(l_shape3.squares,3.3*RIGHT,buff=1))
        i4=(-9,-5)
        l_shape4=grid.create_coordinate_group([i4,(i4[0]+1,i4[1]),(i4[0]-1,i4[1]),(i4[0]-2,i4[1]),(i4[0]-2,i4[1]-1),(i4[0]-2,i4[1]+1),(i4[0],i4[1]-1),(i4[0],i4[1]+1)], color=GREEN_A)
        vec_set4=generate_vec_set("???","???",l_shape4.squares.get_center())
        self.play(vec_set4.animate.next_to(l_shape4.squares,3.3*RIGHT,buff=1))
        text=Text("终于结束了...",font="KaiTi").set_color_by_gradient(GREY,BLUE)
        text.move_to(DOWN*9.5)
        self.wait(0.5)
        self.play(Write(text))
        self.wait(2)
class tree1_5(Scene):
    def construct(self):
        text=Text("密铺分类:",font="KaiTi").set_color_by_gradient(GREY_A,BLUE_A).shift(3.5*UP)
        text_1=Text("周期性",font="KaiTi").set_color_by_gradient(GREY_A,GREEN_A).shift(3*LEFT+2*UP)
        text_2=Text("非周期性",font="KaiTi").set_color_by_gradient(GREY_A,GREEN_A).shift(3*RIGHT+2*UP)
        self.play(Write(text))
        self.wait(2)
        self.play(Write(text_1))
        self.wait(2)
        self.play(Write(text_2))
        self.play(text_1.animate.scale(1.2))
        self.wait(2)
        self.play(text_1.animate.scale(1/1.2))
        self.wait(2)
        self.play(text_2.animate.scale(1.2))
        self.wait(2)
        self.play(text_2.animate.scale(1/1.2))
        self.wait(2)
class tree1_5_addition1(Scene):
    def construct(self):
        grid = ColorableGrid(self,size=50,n=100,fill_opacity=1)
        self.camera.frame.scale(0.5)
        set1=set([(50,50)])
        list1=generate_situation_points_set(set1,func=[(0,0),(1,0),(0,1),(-1,0),(0,-1)])
        set2=generate_situation_point_set(set1,list1,func=[(-1,2),(2,1)])
        list2=generate_situation_points_set(set2,func=[(0,0),(1,0),(0,1),(-1,0),(0,-1)])
        set3=generate_situation_point_set(set2,list2,func=[(-1,2),(2,1)])
        list3=generate_situation_points_set(set3,func=[(0,0),(1,0),(0,1),(-1,0),(0,-1)])
        set4=generate_situation_point_set(set3,list3,func=[(-1,2),(2,1)])
        list4=generate_situation_points_set(set4,func=[(0,0),(1,0),(0,1),(-1,0),(0,-1)])
        set5=generate_situation_point_set(set4,list4,func=[(-1,2),(2,1)])
        list5=generate_situation_points_set(set5,func=[(0,0),(1,0),(0,1),(-1,0),(0,-1)])
        set6=generate_situation_point_set(set5,list5,func=[(-1,2),(2,1)])
        list6=generate_situation_points_set(set6,func=[(0,0),(1,0),(0,1),(-1,0),(0,-1)])
        set7=generate_situation_point_set(set6,list6,func=[(-1,2),(2,1)])
        list7=generate_situation_points_set(set7,func=[(0,0),(1,0),(0,1),(-1,0),(0,-1)])
        set8=generate_situation_point_set(set7,list7,func=[(-1,2),(2,1)])
        set_all=set1|set2|set3|set4|set5|set6|set7|set8
        GROUP1=grid.create_coordinate_group(set_all,color=GREEN)
        self.wait(1)
        GROUP1.translate(vector=np.array([-1,2,0]))
        GROUP1.translate(vector=np.array([2,1,0]))
        self.wait(2)
class tree1_6(Scene):
    def construct(self):
        grid = ColorableGrid(self,size=6.5,n=13,center_offset=RIGHT*3,add=False,fill_opacity=0.9)
        grid.add_grid(condition=0)
        self.wait(2)
        text=Text("待解图形P",font="KaiTi").set_color_by_gradient(GREY,BLUE)
        text.move_to(UP*3+LEFT*3)
        self.play(Write(text))
        i=(9,-5)
        l_shape = grid.create_coordinate_group([i,(i[0]+1,i[1]),(i[0]-1,i[1]),(i[0],i[1]-1),(i[0],i[1]+1),(i[0]-2,i[1]),(i[0]-2,i[1]-1)], color=GOLD)
        self.wait(2)
        l_shape.translate(vector=(-3,11),color=[BLUE,BLUE_A,BLUE_B])
        text1=Text("求解向量：",font="KaiTi").set_color_by_gradient(GREY,BLUE)
        text1.move_to(DOWN*0.4+LEFT*3)
        vec_set=VGroup(Tex("vec1:").set_color_by_gradient(GREY,BLUE),Tex("vec2:").set_color_by_gradient(GREY,BLUE)).arrange(DOWN,buff=0.5)
        vec_set.move_to(DOWN*1.6+LEFT*4)
        self.play(Write(text1),Write(vec_set[0]))
        self.wait(2)
        i=(6,6)
        l_shape1=grid.create_coordinate_group([i,(i[0]+1,i[1]),(i[0]-1,i[1]),(i[0],i[1]-1),(i[0],i[1]+1),(i[0]-2,i[1]),(i[0]-2,i[1]-1)],color=[BLUE,BLUE_A,BLUE_B],add=False)
        vec=Tex("(2,1)")
        vec.next_to(vec_set[0],RIGHT,buff=0.5)
        self.play(Write(vec))
        vec1=Tex("(-1,3)")
        vec1.next_to(vec_set[1],RIGHT,buff=0.5)
        self.play(Write(vec_set[1]),Write(vec1))
        l_shape1.translate(vector=(1,2),color=[GREEN,GREEN_A,GREEN_B])
        self.wait(0.5)
        l_shape1.translate(vector=(-1,-2),color=[GREEN,GREEN_A,GREEN_B])
        
        k=[i,(i[0]+1,i[1]),(i[0]-1,i[1]),(i[0],i[1]-1),(i[0],i[1]+1),(i[0]-2,i[1]),(i[0]-2,i[1]-1)]
        l_shape2_set=[]
        for i in range(len(k)):
            l_shape2_set.append(k[i])
            l_shape2_set.append((k[i][0]+1,k[i][1]+2))
            l_shape2_set.append((k[i][0]-1,k[i][1]-2))
        l_shape2=grid.create_coordinate_group(l_shape2_set,add=False)
        
        self.wait(1)
        l_shape2.translate(vector=(3,-1),color=[YELLOW,YELLOW_A,YELLOW_B])
        l_shape2.translate(vector=(-3,1),color=[YELLOW,YELLOW_A,YELLOW_B])
        self.wait(2)
        vg=VGroup()
        for i in range(len(l_shape2_set)):
            vg.add(grid.cells[l_shape2_set[i]])
            vg.add(grid.cells[(l_shape2_set[i][0]+3,l_shape2_set[i][1]-1)])
            vg.add(grid.cells[(l_shape2_set[i][0]-3,l_shape2_set[i][1]+1)])
        i=(9,-5)
        l_shape3 = grid.create_coordinate_group([i,(i[0]+1,i[1]),(i[0]-1,i[1]),(i[0],i[1]-1),(i[0],i[1]+1),(i[0]-2,i[1])], color=YELLOW_A,add=False)
        self.play(FadeOut(vg,RIGHT),TransformMatchingParts(l_shape.squares,l_shape3.squares))
        vec_new=Tex("None")
        vec_new.move_to(vec)
        vec1_new=Tex("None")
        vec1_new.move_to(vec1)
        self.play(FadeOut(vec),FadeOut(vec1),Write(vec_new),Write(vec1_new))
        self.wait()
class tree1_7(Scene):
    def construct(self):
        grid = ColorableGrid(self,size=6.5,n=13,center_offset=RIGHT*3,add=False,fill_opacity=0.9)
        grid.add_grid(condition=0)
        self.wait(2)
        text=Text("可密铺图形P",font="KaiTi").set_color_by_gradient(GREY,BLUE)
        text.move_to(UP*3+LEFT*3)
        self.play(Write(text))
        i=(9,-5)
        l_shape = grid.create_coordinate_group([i,(i[0]+1,i[1]),(i[0]-1,i[1]),(i[0],i[1]-1),(i[0],i[1]+1),(i[0]-2,i[1])], color=GREEN_A)
        l_shape.translate(vector=(-3,11),color=[BLUE,BLUE_A,BLUE_B])
        text1=Text("求解向量：",font="KaiTi").set_color_by_gradient(GREY,BLUE)
        text1.move_to(DOWN*0.4+LEFT*3)
        vec_set=VGroup(Tex("vec1:").set_color_by_gradient(GREY,BLUE),Tex("vec2:").set_color_by_gradient(GREY,BLUE)).arrange(DOWN,buff=0.5)
        vec_set.move_to(DOWN*1.6+LEFT*4)
        self.play(Write(text1))
        vec=Tex("None")
        vec.next_to(vec_set[0],RIGHT,buff=0.5)
        vec1=Tex("None")
        vec1.next_to(vec_set[1],RIGHT,buff=0.5)
        self.play(Write(vec),Write(vec_set[0]),Write(vec_set[1]),Write(vec1))
        self.wait()
        l_shape.translates(vector_set=[(-4,9),(-2,13)])
        self.wait(2)
        grid.clear_outside_area([i,(i[0]+1,i[1]),(i[0]-1,i[1]),(i[0],i[1]-1),(i[0],i[1]+1),(i[0]-2,i[1])])
        self.wait(2)
class tree1_8(Scene):
    def construct(self):
        grid = ColorableGrid(self,size=50,n=100,add=False)
        self.camera.frame.rotate(np.arctan(-1/2))
        grid.add_grid(condition=0)
        self.wait(0.5)
        i=(50,50)
        l_shape = grid.create_coordinate_group([i,(i[0]+1,i[1]),(i[0],i[1]+1),(i[0],i[1]-1),(i[0]-1,i[1])], color=YELLOW)
        vector_set1=[(-1,2),(-2,4),(-3,6),(-4,8),(-5,10),(-6,12),(-7,14)]
        color_set1=[YELLOW_A,YELLOW_B,GREEN_A,GREEN_B,BLUE_A,BLUE_B,GREY_A,ORANGE,RED_A]
        k=[i,(i[0]+1,i[1]),(i[0],i[1]+1),(i[0],i[1]-1),(i[0]-1,i[1])]
        point_set=[i,(i[0]+1,i[1]),(i[0],i[1]+1),(i[0],i[1]-1),(i[0]-1,i[1])]
        for l in range(len(vector_set1)):
            l_shape.translates(vector_set=[vector_set1[l],(-vector_set1[l][0],-vector_set1[l][1])],color=[color_set1[l],color_set1[l]])
        for p in range(len(vector_set1)):
            for j in range(len(k)):
                point_set.append((k[j][0]+vector_set1[p][0],k[j][1]+vector_set1[p][1]))
                point_set.append((k[j][0]-vector_set1[p][0],k[j][1]-vector_set1[p][1]))
        point_set=set(point_set)
        point_set=list(point_set)
        l_shape1=grid.create_coordinate_group(point_set,add=False)
        l_shape1.translates(vector_set=[(2,1),(-2,-1)])
        l_shape1.translates(vector_set=[(4,2),(-4,-2)])
        l_shape1.translates(vector_set=[(6,3),(-6,-3)])
        l_shape1.translates(vector_set=[(8,4),(-8,-4)])
        self.wait(2)
        vector1=Arrow(ORIGIN,RIGHT+UP*2,color=BLUE,buff=0)
        vector2=Arrow(ORIGIN,DOWN+RIGHT*2,color=BLUE,buff=0)
        self.play(Write(vector1),Write(vector2))
        self.wait(2)
        square=Polygon(ORIGIN,RIGHT+UP*2,3*RIGHT+UP,2*RIGHT+DOWN,fill_opacity=0.5)
        self.play(FadeIn(square))
        self.wait(2)
        self.play(FadeOut(vector1),FadeOut(vector2),FadeOut(square))
        b=[]
        for i in range(len(k)):
            b.append(k[i])
            b.append((k[i][0]+2,k[i][1]+1))
            b.append((k[i][0]+1,k[i][1]+3))
            b.append((k[i][0]-1,k[i][1]+2))
        
        grid.clear_outside_area(b)
        l_shape2=grid.create_coordinate_group(b,add=False)
        self.wait(2)
        l_shape2.translates(vector_set=[(4,2),(-4,-2),(-2,4),(2,-4)],color=[GREEN_A])
        self.wait(2)
class tree1_9(Scene):
    def construct(self):
        grid = ColorableGrid(self,size=6.5,n=13,center_offset=RIGHT*3,add=False,fill_opacity=0.9)
        grid.add_grid(condition=0)
        self.wait(2)
        text=Text("可密铺图形P",font="KaiTi").set_color_by_gradient(GREY,BLUE)
        text.move_to(UP*3+LEFT*4)
        self.play(Write(text))
        i=(9,-7)
        l_shape = grid.create_coordinate_group([i,(i[0]+1,i[1]),(i[0]-1,i[1]),(i[0],i[1]-1),(i[0],i[1]+1)], color=GREEN_A)
        text1=Text("解向量(假设其没有)",font_size=24,font="KaiTi").set_color_by_gradient(GREY,BLUE)
        text1.move_to(DOWN*0.4+LEFT*4)
        vec_set=VGroup(Tex("vec1:").set_color_by_gradient(GREY,BLUE),Tex("vec2:").set_color_by_gradient(GREY,BLUE)).arrange(DOWN,buff=0.5)
        vec_set.move_to(DOWN*1.6+LEFT*5)
        self.play(Write(text1))
        vec=Tex("None")
        vec.next_to(vec_set[0],RIGHT,buff=0.5)
        vec1=Tex("None")
        vec1.next_to(vec_set[1],RIGHT,buff=0.5)
        self.play(Write(vec),Write(vec_set[0]),Write(vec_set[1]),Write(vec1))
        self.wait()
        l_shape.translates(vector_set=[(-2,13),(-3,15),(0,14),(-1,16)])
        self.wait(2)
        grid.hide_grid_lines()
        text2=Text("平移组合图形G",font="KaiTi").set_color_by_gradient(GREY,BLUE)
        text2.move_to(UP*3+RIGHT*4)
        self.play(Write(text2))
        self.wait(2)
        solution=VGroup(Text("解向量",font_size=24,font="KaiTi").set_color_by_gradient(GREY,BLUE),
                        VGroup(Tex("vec1:"),Tex("(2,4)") ).arrange(RIGHT,buff=0.5),
                        VGroup(Tex("vec2:"),Tex("(4,-2)") ).arrange(RIGHT,buff=0.5)
                        ).arrange(DOWN,buff=0.2)
        solution.move_to(DOWN*1.5+RIGHT*3)
        self.play(FadeIn(solution))
class GridDemo1(Scene):
    def construct(self):
        grid = ColorableGrid(self,size=50,n=100,add=False)
        grid.add_grid(condition=0)
        self.wait(0.5)
        i=(50,50)
        l_shape = grid.create_coordinate_group([i,(i[0]+1,i[1]),(i[0],i[1]+1),(i[0],i[1]-1)], color=RED)
        vector_set1=[(1,-2),(2,0),(1,2),(-1,2),(-2,0),(-1,-2)]
        l_shape.translates(vector_set=list(vector_set1),color=GREEN)
        vector_set2=set()
        for i in range(len(vector_set1)):
            for j in range(len(vector_set1)):
                vector_plus=(vector_set1[i][0]+vector_set1[j][0],vector_set1[i][1]+vector_set1[j][1])
                if vector_plus!=(0,0):
                    vector_set2.add(vector_plus)
        vector_set2=vector_set2-set(vector_set1)
        vector_set2=list(vector_set2)
        l_shape.translates(vector_set=vector_set2,color=YELLOW)
        
        vector_set3=set()
        for i in range(len(vector_set2)):
            for j in range(len(vector_set1)):
                vector_plus=(vector_set2[i][0]+vector_set1[j][0],vector_set2[i][1]+vector_set1[j][1])
                if vector_plus!=(0,0):
                    vector_set3.add(vector_plus)
        vector_set3=vector_set3-set(vector_set1)-set(vector_set2)
        vector_set3=list(vector_set3)
        l_shape.translates(vector_set=vector_set3,color=BLUE)
        vector_set4=set()
        for i in range(len(vector_set3)):
            for j in range(len(vector_set1)):
                vector_plus=(vector_set3[i][0]+vector_set1[j][0],vector_set3[i][1]+vector_set1[j][1])
                if vector_plus!=(0,0):
                    vector_set4.add(vector_plus)
        vector_set4=vector_set4-set(vector_set1)-set(vector_set2)-set(vector_set3)
        vector_set4=list(vector_set4)
        l_shape.translates(vector_set=vector_set4,color=ORANGE)
        self.wait(2)
class GridDemo(Scene):
    def construct(self):
        grid = ColorableGrid(self,size=50,n=100,add=False)
        grid.add_grid(condition=0)
        self.wait(0.5)
        i=(50,50)
        l_shape = grid.create_coordinate_group([i,(i[0]+1,i[1]),(i[0]-1,i[1]),(i[0],i[1]+1),(i[0],i[1]-1)], color=RED)
        
        vector_set1=[(-1,2),(2,1),(1,-2),(-2,-1)]
        l_shape.translates(vector_set=list(vector_set1),color=ORANGE)
        vector_set2=set()
        for i in range(len(vector_set1)):
            for j in range(len(vector_set1)):
                vector_plus=(vector_set1[i][0]+vector_set1[j][0],vector_set1[i][1]+vector_set1[j][1])
                if vector_plus!=(0,0):
                    vector_set2.add(vector_plus)
        vector_set2=vector_set2-set(vector_set1)
        vector_set2=list(vector_set2)
        l_shape.translates(vector_set=vector_set2,color=YELLOW)
        
        vector_set3=set()
        for i in range(len(vector_set2)):
            for j in range(len(vector_set1)):
                vector_plus=(vector_set2[i][0]+vector_set1[j][0],vector_set2[i][1]+vector_set1[j][1])
                if vector_plus!=(0,0):
                    vector_set3.add(vector_plus)
        vector_set3=vector_set3-set(vector_set1)-set(vector_set2)
        vector_set3=list(vector_set3)
        l_shape.translates(vector_set=vector_set3,color=GREEN)
        vector_set4=set()
        for i in range(len(vector_set3)):
            for j in range(len(vector_set1)):
                vector_plus=(vector_set3[i][0]+vector_set1[j][0],vector_set3[i][1]+vector_set1[j][1])
                if vector_plus!=(0,0):
                    vector_set4.add(vector_plus)
        vector_set4=vector_set4-set(vector_set1)-set(vector_set2)-set(vector_set3)
        vector_set4=list(vector_set4)
        l_shape.translates(vector_set=vector_set4,color=ORANGE)
        self.wait(2)   
class GridDemo2(Scene):
    def construct(self):
        self.camera.frame.rotate(PI/3)
        grid = ColorableGrid(self,size=50,n=100,add=False)
        
        grid.add_grid(condition=0)
        self.wait(0.5)
        i=(50,50)
        l_shape = grid.create_coordinate_group([i,(i[0]+1,i[1]),(i[0]-1,i[1]),(i[0],i[1]-1),(i[0],i[1]+1),(i[0]-2,i[1]),(i[0]-2,i[1]-1)], color=RED)
        
        vector_set1=[(1,2),(4,1),(3,-1),(-1,-2),(-4,-1),(-3,1)]
        l_shape.translates(vector_set=list(vector_set1),color=ORANGE)
        vector_set2=set()
        for i in range(len(vector_set1)):
            for j in range(len(vector_set1)):
                vector_plus=(vector_set1[i][0]+vector_set1[j][0],vector_set1[i][1]+vector_set1[j][1])
                if vector_plus!=(0,0):
                    vector_set2.add(vector_plus)
        vector_set2=vector_set2-set(vector_set1)
        vector_set2=list(vector_set2)
        l_shape.translates(vector_set=vector_set2,color=YELLOW)
        vector_set3=set()
        for i in range(len(vector_set2)):
            for j in range(len(vector_set1)):
                vector_plus=(vector_set2[i][0]+vector_set1[j][0],vector_set2[i][1]+vector_set1[j][1])
                if vector_plus!=(0,0):
                    vector_set3.add(vector_plus)
        vector_set3=vector_set3-set(vector_set1)-set(vector_set2)
        vector_set3=list(vector_set3)
        l_shape.translates(vector_set=vector_set3,color=GREEN)
        vector_set4=set()
        for i in range(len(vector_set3)):
            for j in range(len(vector_set1)):
                vector_plus=(vector_set3[i][0]+vector_set1[j][0],vector_set3[i][1]+vector_set1[j][1])
                if vector_plus!=(0,0):
                    vector_set4.add(vector_plus)
        vector_set4=vector_set4-set(vector_set1)-set(vector_set2)-set(vector_set3)
        vector_set4=list(vector_set4)
        l_shape.translates(vector_set=vector_set4,color=ORANGE)
        vector_set5=set()
        for i in range(len(vector_set4)):
            for j in range(len(vector_set1)):
                vector_plus=(vector_set4[i][0]+vector_set1[j][0],vector_set4[i][1]+vector_set1[j][1])
                if vector_plus!=(0,0):
                    vector_set5.add(vector_plus)
        vector_set5=vector_set5-set(vector_set1)-set(vector_set2)-set(vector_set3)-set(vector_set4)
        vector_set5=list(vector_set5)
        l_shape.translates(vector_set=vector_set5,color=BLUE_A)
        vector_set6=set()
        for i in range(len(vector_set5)):
            for j in range(len(vector_set1)):
                vector_plus=(vector_set5[i][0]+vector_set1[j][0],vector_set5[i][1]+vector_set1[j][1])
                if vector_plus!=(0,0):
                    vector_set6.add(vector_plus)
        vector_set6=vector_set6-set(vector_set1)-set(vector_set2)-set(vector_set3)-set(vector_set4)-set(vector_set5)
        vector_set6=list(vector_set6)
        l_shape.translates(vector_set=vector_set6,color=GREEN_A)
        vector_set7=set()
        for i in range(len(vector_set6)):
            for j in range(len(vector_set1)):
                vector_plus=(vector_set6[i][0]+vector_set1[j][0],vector_set6[i][1]+vector_set1[j][1])
                if vector_plus!=(0,0):
                    vector_set7.add(vector_plus)
        vector_set7=vector_set7-set(vector_set1)-set(vector_set2)-set(vector_set3)-set(vector_set4)-set(vector_set5)-set(vector_set6)
        vector_set7=list(vector_set7)
        l_shape.translates(vector_set=vector_set7,color=YELLOW_A)
        self.play(self.camera.frame.animate.scale(1.2))
        self.wait(2)
if __name__ == "__main__":
    os.system("manimgl {} tree1_1 -c black --uhd -w".format(__file__))