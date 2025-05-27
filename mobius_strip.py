"""
Mobius Strip Modeling Assignment - Karkhana.io
Author: [Your Name]
Description: This script models a Mobius strip using parametric equations,
generates a 3D mesh, and computes the surface area and edge length numerically.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MobiusStrip:
    def __init__(self, R=1.0, w=0.3, n=200):
        """
        Initialize Mobius strip parameters.

        Args:
            R (float): Radius from the center to the strip center.
            w (float): Width of the strip.
            n (int): Resolution (number of points per axis).
        """
        self.R = R
        self.w = w
        self.n = n

        # Create mesh grid for parameters u and v
        self.u = np.linspace(0, 2 * np.pi, n)
        self.v = np.linspace(-w / 2, w / 2, n)
        self.U, self.V = np.meshgrid(self.u, self.v)

        # Generate the 3D coordinates
        self.X, self.Y, self.Z = self._generate_mesh()

    def _generate_mesh(self):
        """Generate the 3D surface mesh of the Mobius strip."""
        U = self.U
        V = self.V
        X = (self.R + V * np.cos(U / 2)) * np.cos(U)
        Y = (self.R + V * np.cos(U / 2)) * np.sin(U)
        Z = V * np.sin(U / 2)
        return X, Y, Z

    def compute_surface_area(self):
        """
        Approximate the surface area numerically using the cross product of partial derivatives.
        """
        dU = 2 * np.pi / (self.n - 1)
        dV = self.w / (self.n - 1)

        # Partial derivatives
        Xu = np.gradient(self.X, axis=1) / dU
        Xv = np.gradient(self.X, axis=0) / dV
        Yu = np.gradient(self.Y, axis=1) / dU
        Yv = np.gradient(self.Y, axis=0) / dV
        Zu = np.gradient(self.Z, axis=1) / dU
        Zv = np.gradient(self.Z, axis=0) / dV

        # Cross product magnitude of (Xu, Yu, Zu) x (Xv, Yv, Zv)
        Nx = Yu * Zv - Zu * Yv
        Ny = Zu * Xv - Xu * Zv
        Nz = Xu * Yv - Yu * Xv
        area_element = np.sqrt(Nx**2 + Ny**2 + Nz**2)

        return np.sum(area_element) * dU * dV

    def compute_edge_length(self):
        """
        Compute the approximate length of one edge of the Mobius strip.
        """
        u = self.u
        v_edge = self.w / 2
        x = (self.R + v_edge * np.cos(u / 2)) * np.cos(u)
        y = (self.R + v_edge * np.cos(u / 2)) * np.sin(u)
        z = v_edge * np.sin(u / 2)

        dx = np.diff(x)
        dy = np.diff(y)
        dz = np.diff(z)
        return np.sum(np.sqrt(dx**2 + dy**2 + dz**2))

    def plot(self):
        """Display a 3D plot of the Mobius strip."""
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.X, self.Y, self.Z, color='skyblue', edgecolor='k', linewidth=0.3)
        ax.set_title("Mobius Strip")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    try:
        mobius = MobiusStrip(R=1.0, w=0.3, n=200)
        area = mobius.compute_surface_area()
        edge = mobius.compute_edge_length()

        print(f"Surface Area ≈ {area:.4f} units²")
        print(f"Edge Length ≈ {edge:.4f} units")

        # Optional: show plot
        mobius.plot()
    except Exception as e:
        print(f"An error occurred: {e}")
