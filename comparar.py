import sys
import pyvista as pv
import vtk
from pyvistaqt import QtInteractor
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog


class ControlWindow(QWidget):
    def __init__(self, update_position_callback, get_displacement_callback):
        super().__init__()

        self.update_position_callback = update_position_callback
        self.get_displacement_callback = get_displacement_callback
        self.initUI()

        # Timer para movimento contínuo
        self.timer = QTimer()
        self.timer.timeout.connect(self.move_step)

        self.current_delta = None  # Guarda o deslocamento atual

    def initUI(self):
        self.setWindowTitle("Controle de Posição do VTU")
        self.setGeometry(100, 100, 250, 350)

        layout = QVBoxLayout()

        # Rótulo para exibir deslocamento total
        self.displacement_label = QLabel("Deslocamento: (0, 0, 0)", self)
        layout.addWidget(self.displacement_label)

        # Criando os botões para mover a malha
        buttons = [
            ("X +1", (1, 0, 0)),
            ("X -1", (-1, 0, 0)),
            ("Y +1", (0, 1, 0)),
            ("Y -1", (0, -1, 0)),
            ("Z +1", (0, 0, 1)),
            ("Z -1", (0, 0, -1)),
        ]

        for text, delta in buttons:
            btn = QPushButton(text, self)
            btn.pressed.connect(lambda d=delta: self.start_moving(d))
            btn.released.connect(self.stop_moving)
            layout.addWidget(btn)

        self.setLayout(layout)

    def start_moving(self, delta):
        """Inicia o movimento repetitivo ao segurar o botão"""
        self.current_delta = delta
        self.timer.start(100)  # Move a cada 100ms

    def move_step(self):
        """Executa um passo do movimento"""
        if self.current_delta:
            self.update_position_callback(*self.current_delta)
            self.update_displacement_label()

    def stop_moving(self):
        """Para o movimento quando o botão é solto"""
        self.timer.stop()
        self.current_delta = None

    def update_displacement_label(self):
        """Atualiza a exibição do deslocamento total"""
        displacement = self.get_displacement_callback()
        self.displacement_label.setText(f"Deslocamento: {displacement}")


class VTUViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.vtp_mesh1 = None
        self.vtp_mesh2 = None
        self.vtu_mesh = None

        self.total_displacement = [0, 0, 0]  # Guarda o deslocamento acumulado

        self.setWindowTitle("Visualização 3D")
        self.setGeometry(200, 200, 800, 600)

        # Criando um layout para conter o PyVista
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Criando o widget PyVista
        self.plotter = QtInteractor(self)
        self.layout.addWidget(self.plotter)

    def update_position(self, delta_x=0, delta_y=0, delta_z=0):
        """Move a malha VTU e acumula o deslocamento total"""
        if self.vtu_mesh is not None:
            self.vtu_mesh.points += [delta_x, delta_y, delta_z]
            self.plotter.update_coordinates(self.vtu_mesh.points, render=True)

            # Acumula deslocamento
            self.total_displacement[0] += delta_x
            self.total_displacement[1] += delta_y
            self.total_displacement[2] += delta_z

    def get_total_displacement(self):
        """Retorna o deslocamento acumulado"""
        return tuple(self.total_displacement)

    def load_and_plot(self):
        """Carrega arquivos VTU/VTP e exibe a visualização"""
        # Diálogos de seleção de arquivo
        vtp_file1, _ = QFileDialog.getOpenFileName(None, "Selecione o primeiro arquivo VTP", "", "VTP files (*.vtp)")
        if not vtp_file1:
            print("Nenhum arquivo VTP selecionado. Saindo...")
            return

        vtp_file2, _ = QFileDialog.getOpenFileName(None, "Selecione o segundo arquivo VTP", "", "VTP files (*.vtp)")
        if not vtp_file2:
            print("Nenhum segundo arquivo VTP selecionado. Saindo...")
            return

        vtu_file, _ = QFileDialog.getOpenFileName(None, "Selecione um arquivo VTU", "", "VTU files (*.vtu)")
        if not vtu_file:
            print("Nenhum arquivo VTU selecionado. Saindo...")
            return

        # Carregando as malhas
        self.vtp_mesh1 = pv.read(vtp_file1)
        self.vtp_mesh2 = pv.read(vtp_file2)
        self.vtu_mesh = pv.read(vtu_file)

        # Adicionando as malhas ao PyVista
        self.plotter.add_mesh(self.vtp_mesh1, color='red', opacity=0.7, label="VTP Mesh 1")
        self.plotter.add_mesh(self.vtp_mesh2, color='blue', opacity=0.7, label="VTP Mesh 2")
        self.plotter.add_mesh(self.vtu_mesh, color='green', opacity=0.7, label="VTU Mesh")
        self.plotter.add_legend()
        self.plotter.show()

        # Criando a interface de controle com PyQt5
        self.control_window = ControlWindow(self.update_position, self.get_total_displacement)
        self.control_window.show()

        # Exibir a interface PyQt5
        self.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)  # Criamos QApplication ANTES de qualquer QWidget
    viewer = VTUViewer()
    viewer.load_and_plot()
    sys.exit(app.exec_())  # Mantém a interface PyQt5 rodando
