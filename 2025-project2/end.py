from manimlib import *
from os import system
class ShowIncreasingSubsetsExample(Scene):
    def construct(self):
        str0="片名：{}".format("莫比乌斯的极限")
        str1="出品人：{}".format("擢兰风而霞起")
        str2="使用工具：{}".format("Manim")
        str3="参考文献：{}".format("The Optimal \n          Paper Moebius Band")
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
if __name__ == "__main__":
    system("manimgl {} ShowIncreasingSubsetsExample -c black --uhd -w".format(__file__))