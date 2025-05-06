import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons


class KapitzaPendulum:
    def __init__(self):
        self.g = 9.81
        self.history_length = 200
        self.epsilon = 1e-4  # запас над sqrt(2)
        self.stability_threshold = np.sqrt(2)
        self.initialize_attributes()
        self.setup_ui()
        self.setup_plots()

    def initialize_attributes(self):
        self.omega = 50.0
        self.l = 0.5
        self.a = 0.1
        self.auto_amplitude = True
        self.t_eval = np.array([])
        self.phi = np.array([])
        self.theta = np.array([])
        self.x = np.array([])
        self.y = np.array([])
        self.z = np.array([])
        self.z_pivot = np.array([])
        self.history = [[], [], []]

    def setup_ui(self):
        self.fig = plt.figure(figsize=(18, 12))
        self.ax_main = self.fig.add_subplot(231, projection='3d')
        self.ax_pot = self.fig.add_subplot(232)
        self.ax_phase = self.fig.add_subplot(233)
        self.ax_avg = self.fig.add_subplot(234)
        self.ax_fast = self.fig.add_subplot(235)
        self.ax_params = self.fig.add_subplot(236)

        plt.subplots_adjust(bottom=0.25, hspace=0.5, wspace=0.4)

        ax_omega = plt.axes([0.2, 0.18, 0.65, 0.03])
        ax_length = plt.axes([0.2, 0.14, 0.65, 0.03])
        ax_amp = plt.axes([0.2, 0.10, 0.65, 0.03])

        self.s_omega = Slider(ax_omega, 'Частота ω (рад/с)', 10, 200, valinit=self.omega)
        self.s_length = Slider(ax_length, 'Длина l (м)', 0.1, 2, valinit=self.l)
        self.s_amp = Slider(ax_amp, 'Амплитуда a (м)', 0.01, 0.5, valinit=self.a)

        ax_radio = plt.axes([0.2, 0.02, 0.2, 0.06])
        self.radio = RadioButtons(ax_radio, ['Авто амплитуда', 'Ручная амплитуда'], active=0)
        self.radio.on_clicked(self.update_amplitude_mode)

        ax_button = plt.axes([0.5, 0.02, 0.2, 0.06])
        self.button = Button(ax_button, 'Запустить симуляцию')
        self.button.on_clicked(self.run_simulation)

    def setup_plots(self):
        for ax in [self.ax_pot, self.ax_phase, self.ax_avg, self.ax_fast, self.ax_params]:
            ax.grid(True)
        self.ax_main.set_xlabel('X')
        self.ax_main.set_ylabel('Y')
        self.ax_main.set_zlabel('Z')
        self.ax_main.grid(True)

        self.line, = self.ax_main.plot([], [], [], 'k-', lw=2)
        self.bob, = self.ax_main.plot([], [], [], 'ro', ms=8)
        self.pivot, = self.ax_main.plot([], [], [], 'bo', ms=6)
        self.traj, = self.ax_main.plot([], [], [], 'g-', alpha=0.3, lw=1)

    def update_amplitude_mode(self, label):
        self.auto_amplitude = (label == 'Авто амплитуда')
        self.s_amp.set_active(not self.auto_amplitude)
        self.run_simulation(None)

    def calculate_amplitude(self, omega, l):
        omega0 = np.sqrt(self.g / l)
        threshold = self.stability_threshold
        # Строгое выполнение условия (a/l)*(omega/omega0) > sqrt(2)
        a_min = l * threshold * (omega0 / omega)
        a_min *= (1 + self.epsilon)  # Небольшой запас сверху
        return a_min

    def lagrangian_equations(self, t, y):
        phi, theta, phi_dot, theta_dot = y
        phi_ddot = (self.g/self.l) * np.sin(phi) - (self.a*self.omega**2/self.l) * np.sin(phi) * np.cos(self.omega * t)
        theta_ddot = 0
        return [phi_dot, theta_dot, phi_ddot, theta_ddot]

    def run_simulation(self, event):
        try:
            l_new = self.s_length.val
            omega_new = self.s_omega.val
            omega0_new = np.sqrt(self.g / l_new)

            if self.auto_amplitude:
                a_new = self.calculate_amplitude(omega_new, l_new)
            else:
                a_new = self.s_amp.val

            self.l = l_new
            self.omega = omega_new
            self.a = a_new

            if self.auto_amplitude:
                self.s_amp.set_val(a_new)

            raw_stability_value = (self.a / self.l) * (self.omega / omega0_new)
            self.raw_stability_value = raw_stability_value
            self.is_stable = raw_stability_value > (self.stability_threshold + self.epsilon)

            initial_angle = np.pi - 0.1 if self.is_stable else np.pi + 0.1
            t_max = 10.0
            dt = 0.005
            self.t_eval = np.arange(0, t_max, dt)

            sol = solve_ivp(
                self.lagrangian_equations,
                [0, t_max],
                [initial_angle, 0.0, 0.0, 0.0],
                t_eval=self.t_eval,
                method='DOP853',
                rtol=1e-8
            )

            self.process_results(sol)
            self.update_plots()
            self.create_animation()
            plt.draw()

        except Exception as e:
            print(f"Ошибка: {str(e)}")

    def process_results(self, sol):
        self.phi = sol.y[0]
        self.theta = sol.y[1]

        self.x = self.l * np.sin(self.phi) * np.cos(self.theta)
        self.y = self.l * np.sin(self.phi) * np.sin(self.theta)
        self.z = -self.l * np.cos(self.phi) + self.a * np.cos(self.omega * self.t_eval)
        self.z_pivot = self.a * np.cos(self.omega * self.t_eval)

        window_size = int(2 * np.pi / (self.omega * (self.t_eval[1] - self.t_eval[0])))
        if window_size % 2 == 0:
            window_size += 1

        self.X = np.convolve(self.phi, np.ones(window_size) / window_size, mode='same')
        self.xi = self.phi - self.X

        self.final_state = "УСТОЙЧИВЫЙ" if self.is_stable else "НЕУСТОЙЧИВЫЙ"
        self.state_color = "green" if self.is_stable else "red"

    def update_plots(self):
        for ax in [self.ax_pot, self.ax_phase, self.ax_avg, self.ax_fast, self.ax_params]:
            ax.clear()
            ax.grid(True)

        phi_range = np.linspace(0, 2*np.pi, 100)
        omega0 = np.sqrt(self.g / self.l)
        A = 0.25 * (self.a/self.l)**2 * (self.omega/omega0)**2
        U_eff = -self.g * self.l * np.cos(phi_range) + self.g * self.l * A * np.sin(phi_range)**2
        self.ax_pot.plot(phi_range, U_eff, 'b-')
        self.ax_pot.set_title(r"Эффективный потенциал $U_{eff}(\phi)$")
        self.ax_pot.set_xlabel(r'Угол $\phi$ (рад)')
        self.ax_pot.set_ylabel('Потенциальная энергия')

        self.ax_avg.plot(self.t_eval, self.X, 'b-')
        self.ax_avg.set_title("Усредненное движение X(t)")
        self.ax_avg.set_xlabel('Время (с)')
        self.ax_avg.set_ylabel('Угол (рад)')

        self.ax_fast.plot(self.t_eval, self.xi, 'r-')
        self.ax_fast.set_title("Быстрая составляющая ξ(t)")
        self.ax_fast.set_xlabel('Время (с)')
        self.ax_fast.set_ylabel('Угол (рад)')

        dX_dt = np.gradient(self.X, self.t_eval)
        self.ax_phase.plot(self.X, dX_dt, 'b-')
        self.ax_phase.set_title("Фазовый портрет X(t)")
        self.ax_phase.set_xlabel('Угол (рад)')
        self.ax_phase.set_ylabel('Угловая скорость (рад/с)')

        self.ax_params.axis('off')
        params_text = (
            f"Параметры системы:\n"
            f"ω = {self.omega:.1f} рад/с\n"
            f"ω₀ = {omega0:.2f} рад/с\n"
            f"l = {self.l:.2f} м\n"
            f"a = {self.a:.4f} м\n"
            f"(a/l)·(ω/ω₀) = {self.raw_stability_value:.4f}\n"
            f"Условие устойчивости (>√2 ≈ 1.414): {'Да' if self.is_stable else 'Нет'}\n"
            f"Состояние: {self.final_state}"
        )
        self.ax_params.text(0.1, 0.5, params_text, fontsize=10)
        self.update_main_plot()

    def update_main_plot(self):
        self.ax_main.clear()
        max_range = 1.5 * (self.l + self.a)
        self.ax_main.set_xlim([-max_range, max_range])
        self.ax_main.set_ylim([-max_range, max_range])
        self.ax_main.set_zlim([-2*self.l, max_range])
        self.ax_main.set_box_aspect([1, 1, 1])
        self.ax_main.set_xlabel('X')
        self.ax_main.set_ylabel('Y')
        self.ax_main.set_zlabel('Z')

        xx, yy = np.meshgrid(np.linspace(-max_range, max_range, 2),
                             np.linspace(-max_range, max_range, 2))
        self.ax_main.plot_surface(xx, yy, np.zeros_like(xx), color='gray', alpha=0.3)

        title = (
            f"3D Маятник Капицы\n"
            f"ω={self.omega:.1f} рад/с, l={self.l:.2f} м, a={self.a:.4f} м\n"
            f"Состояние: {self.final_state}"
        )
        self.ax_main.set_title(title, color=self.state_color, pad=20)

        self.line, = self.ax_main.plot([], [], [], 'k-', lw=2)
        self.bob, = self.ax_main.plot([], [], [], 'ro', ms=8)
        self.pivot, = self.ax_main.plot([], [], [], 'bo', ms=6)
        self.traj, = self.ax_main.plot([], [], [], 'g-', alpha=0.3, lw=1)
        self.history = [[], [], []]

    def create_animation(self):
        if len(self.x) == 0:
            return

        def animate(i):
            self.line.set_data([0, self.x[i]], [0, self.y[i]])
            self.line.set_3d_properties([self.z_pivot[i], self.z[i]])
            self.bob.set_data([self.x[i]], [self.y[i]])
            self.bob.set_3d_properties([self.z[i]])
            self.pivot.set_data([0], [0])
            self.pivot.set_3d_properties([self.z_pivot[i]])

            self.history[0].append(self.x[i])
            self.history[1].append(self.y[i])
            self.history[2].append(self.z[i])

            if len(self.history[0]) > self.history_length:
                for h in self.history:
                    h.pop(0)

            self.traj.set_data(self.history[0], self.history[1])
            self.traj.set_3d_properties(self.history[2])

            return self.line, self.bob, self.pivot, self.traj

        self.ani = FuncAnimation(
            self.fig,
            animate,
            frames=min(len(self.t_eval), len(self.x)),
            interval=20,
            blit=True,
            repeat=False
        )


if __name__ == '__main__':
    pendulum = KapitzaPendulum()
    plt.show()