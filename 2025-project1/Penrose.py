from manimlib import *
import cv2
import numpy as np
import os

import cmath
import math

class PenroseTiling(Scene):
    def construct(self):
        # Initialize parameters
        base = 7
        divisions = 5  # Example number of subdivisions
        phi = (5 ** 0.5 + 1) / 2  # Golden ratio
        scale_factor = 2  # Adjust this factor to increase the size of the initial tiling

        # Create initial triangles
        triangles = self.create_initial_triangles(base, scale_factor)

        # Animate subdivisions
        for _ in range(divisions):
            triangles = self.subdivide(triangles, phi)
            self.animate_tiling(triangles)

    def create_initial_triangles(self, base, scale_factor):
        triangles = []
        for i in range(base * 2):
            v2 = cmath.rect(scale_factor, (2 * i - 1) * math.pi / (base * 2))
            v3 = cmath.rect(scale_factor, (2 * i + 1) * math.pi / (base * 2))
            if i % 2 == 0:
                v2, v3 = v3, v2  # Mirror every other triangle
            triangles.append(("thin", 0, v2, v3))
        return triangles

    def subdivide(self, triangles, phi):
        new_triangles = []
        for shape, v1, v2, v3 in triangles:
            if shape == "thin":
                p1 = v1 + (v2 - v1) / phi
                new_triangles += [("thin", v3, p1, v2), ("thicc", p1, v3, v1)]
            else:
                p2 = v2 + (v1 - v2) / phi
                p3 = v2 + (v3 - v2) / phi
                new_triangles += [("thicc", p3, v3, v1), ("thicc", p2, p3, v2), ("thin", p3, p2, v1)]
        return new_triangles

    def animate_tiling(self, triangles):
        group = VGroup()
        for shape, v1, v2, v3 in triangles:
            color = BLUE if shape == "thin" else ORANGE
            points = [np.array([v.real, v.imag, 0]) for v in [v1, v2, v3]]
            polygon = Polygon(*points, color=color, fill_opacity=0.5).set_stroke(color=color,width=0.01)
            group.add(polygon)

        self.play(ShowCreation(group))
        self.wait(1)
        self.remove(group)
if __name__ == "__main__":
    os.system("manimgl {} PenroseTiling -c black --uhd -w".format(__file__))