import pyvista as pv
import numpy as np
import vtk
import os
import sys
import random
from collections import deque  # Importar deque para usar como fila
from PyQt5.QtWidgets import QApplication, QFileDialog, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QCheckBox
from PyQt5.QtCore import QTimer

app = QApplication([])

# Diálogo para selecionar a pasta contendo os arquivos .vtp
folder_path = QFileDialog.getExistingDirectory(None, "Selecionar Pasta com Arquivos VTP")
if not folder_path:
    raise Exception("Nenhuma pasta selecionada.")

# Listar todos os arquivos .vtp na pasta
vtp_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.vtp')]
if not vtp_files:
    raise Exception("Nenhum arquivo .vtp encontrado na pasta selecionada.")

# Vetor de 40 cores fixas em formato RGB (valores entre 0 e 1)
color_palette = [
    [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0], [0.5, 0.5, 0.5], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0],
    [0.75, 0.75, 0.75], [0.75, 0.25, 0.5], [0.25, 0.75, 0.5], [0.75, 0.5, 0.25],
    [0.75, 0.25, 0.25], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5],
    [0.75, 0.75, 0.0], [0.0, 0.75, 0.75], [0.75, 0.0, 0.75], [1.0, 0.5, 0.0],
    [0.0, 1.0, 0.5], [0.5, 0.0, 1.0], [1.0, 0.25, 0.25], [0.25, 1.0, 0.25],
    [1.0, 0.75, 0.25], [0.25, 1.0, 0.75], [0.5, 0.75, 0.25], [0.75, 0.5, 0.5],
    [0.5, 0.75, 0.5], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]
]


# Carregar todas as malhas
all_meshes = []
for file_path in vtp_files:
    mesh = pv.read(file_path)
    if 'Normals' not in mesh.point_data:  # Calcular normais se não estiverem presentes
        mesh.compute_normals(inplace=True)
    all_meshes.append(mesh)

# Criar o plotter
plotter = pv.Plotter()

# Criar uma lista para armazenar as referências dos atores das malhas individuais
mesh_actors = []

# Adicionar cada malha ao plotter com cores diferentes, usando a paleta
for i, mesh in enumerate(all_meshes):
    color = color_palette[i % len(color_palette)]  # Escolher a cor correspondente na paleta (circular)
    actor = plotter.add_mesh(mesh, color=color, opacity=0.7, label=f"Malha {i+1}: {os.path.basename(vtp_files[i])}")
    mesh_actors.append(actor)  # Armazenar referência ao ator


# Adicionar legenda para identificar as malhas
plotter.add_legend()

# Combinar todas as malhas em uma única mesh_torso usando MultiBlock
multi_block = pv.MultiBlock(all_meshes)
mesh_torso = multi_block.combine()
plotter.add_mesh(mesh_torso, opacity=0)  # Torna invisível, mas ainda pode ser usada para cálculos
# Adicionar legenda para identificar as malhas carregadas
plotter.add_legend()
# Criar a esfera de preview (inicialmente no centro)
sphere_radius = 6
preview_sphere = pv.Sphere(radius=sphere_radius)
preview_actor = plotter.add_mesh(preview_sphere, color='blue', opacity=0.9)
initial_position = np.mean(mesh_torso.points, axis=0)
preview_actor.SetPosition(initial_position)

# Criar a linha conectando as duas esferas
line_actor = None

# Variável para controlar a atualização do preview
is_preview_active = True

# Armazenar a posição atual da esfera de preview
current_preview_position = np.array(initial_position)

# Usar uma lista para armazenar os eletrodos adicionados (posição, ator)
electrodes = []

# Variável para controlar o estado de bloqueio
is_space_pressed = False

# Função para atualizar a linha entre as esferas
def update_line(start, end):
    global line_actor
    if line_actor:
        plotter.remove_actor(line_actor)
    line = pv.Line(start, end)
    line_actor = plotter.add_mesh(line, color='yellow')

# Função para adicionar eletrodo
def add_electrode():
    global preview_actor
    if len(electrodes) < 20:  # Limitar a 20 eletrodos
        position = preview_actor.GetPosition()  # Posição da esfera azul
        electrode_sphere = plotter.add_mesh(pv.Sphere(radius=3, center=position), color='green', opacity=1.0)  # Eletrodo como esfera verde
        electrodes.append((position, electrode_sphere))  # Armazenar a posição e o ator do eletrodo
        print(f"Eletrodo #{len(electrodes)} adicionado na posição: X={position[0]:.2f}, Y={position[1]:.2f}, Z={position[2]:.2f}")  # Exibir coordenadas do eletrodo
        print(f"Número de eletrodos na fila: {len(electrodes)}")  # Mostrar a quantidade de eletrodos na fila

# Função para apagar o último eletrodo
def remove_last_electrode():
    global electrodes
    if electrodes:
        position, electrode_sphere = electrodes.pop()  # Remove o último eletrodo e obtém a posição e o ator
        plotter.remove_actor(electrode_sphere)  # Remove a última esfera do plotter
        print("Último eletrodo removido.")  # Confirma a remoção do eletrodo
        print(f"Número de eletrodos restantes na fila: {len(electrodes)}")  # Mostrar a quantidade de eletrodos restantes
    else:
        print("Nenhum eletrodo para remover.")

def find_lowest_y_point(mouse_position, tol=None):
    closest_point_id = mesh_torso.find_closest_point(mouse_position)
    closest_point = mesh_torso.points[closest_point_id]
    x, z = closest_point[0], closest_point[2]
    if tol is None:
        bounds = mesh_torso.bounds  # [xmin, xmax, ymin, ymax, zmin, zmax]
        tol = (bounds[1] - bounds[0]) * 0.001  # 1% da extensão em X, por exemplo
    
    # Usar norma no plano XZ para filtrar candidatos
    mask = np.linalg.norm(mesh_torso.points[:, [0,2]] - np.array([x, z]), axis=1) < tol
    candidates = mesh_torso.points[mask]
    print(f"Encontrados {len(candidates)} pontos candidatos com tol={tol}")
    
    if candidates.size > 0:
        lowest_y_point = candidates[np.argmin(candidates[:, 1])]
        return lowest_y_point
    else:
        return closest_point



# Função para atualizar a posição da esfera ao clicar com o botão esquerdo
def on_left_click(iren, event):
    global is_preview_active, current_preview_position
    if is_preview_active and not is_space_pressed:  # Verificar se não está no modo de bloqueio
        mouse_pos = plotter.pick_mouse_position()

        if mouse_pos is not None:
            new_position = find_lowest_y_point(mouse_pos)
            preview_actor.SetPosition(new_position)
            current_preview_position = new_position  # Atualizar a posição atual

# Função para mover a esfera ao longo das extremidades da malha com as teclas numéricas
def move_preview(iren, event):
    global current_preview_position, preview_actor, is_space_pressed
    if is_space_pressed:  # Ignorar se a barra de espaço estiver pressionada
        return

    delta = 1.0  # Definir o quanto se mover por vez
    key = iren.GetKeySym()  # Capturar a tecla pressionada

    # Copiar a posição atual para realizar as atualizações
    new_position = np.copy(current_preview_position)

    # Atualizar a posição de acordo com as teclas pressionadas (setas e teclas numéricas)
    if key == 'Up':  # Arrow Up - Mover no eixo Z positivo (frente)
        new_position[2] += delta
    elif key == 'Down':  # Arrow Down - Mover no eixo Z negativo (trás)
        new_position[2] -= delta
    elif key == 'Left':  # Arrow Left - Mover no eixo X negativo (esquerda)
        new_position[0] -= delta
    elif key == 'Right':  # Arrow Right - Mover no eixo X positivo (direita)
        new_position[0] += delta
    else:
        return  # Ignorar outras teclas

    # Manter a posição Y atual e encontrar o ponto com o menor Y na malha
    new_position[1] = find_lowest_y_point(new_position)[1]  # Atualiza o Y baseado na malha

    # Atualizar a posição da esfera de preview
    preview_actor.SetPosition(new_position)
    current_preview_position = new_position  # Atualizar a posição atual
    plotter.render()  # Atualizar a visualização

# Função para capturar eventos de tecla
def capture_key_events(iren, event):
    global is_space_pressed, key
    key = iren.GetKeySym()  # Capturar a tecla pressionada
    print(f"Tecla pressionada: {key}")  # Debug

    if key == 'Return':  # Verificar se a tecla pressionada foi a tecla Enter
        add_electrode()
    elif key == 'Backspace':  # Verificar se a tecla pressionada foi a tecla Backspace
        remove_last_electrode()  # Chamar a função para remover o último eletrodo
    elif key == 'space':  # Verificar se a barra de espaço foi pressionada
        is_space_pressed = not is_space_pressed  # Alternar o estado de bloqueio
        print("Modo de controle de câmera " + ("ativado." if is_space_pressed else "desativado."))
    elif key == 's':  # Verificar se a tecla 's' foi pressionada para salvar
        save_files()  # Chamar a função para salvar os arquivos

def save_files():
    global electrodes

    # Diálogo para salvar o arquivo TXT
    txt_file_path, _ = QFileDialog.getSaveFileName(None, "Salvar coordenadas dos eletrodos", "", "TXT files (*.txt);;All files (*)")
    if txt_file_path:
        with open(txt_file_path, 'w') as f:
            for idx, (position, _) in enumerate(electrodes):
                f.write(f"Eletrodo #{idx + 1}: X={position[0]:.2f}, Y={position[1]:.2f}, Z={position[2]:.2f}\n")
        print(f"Arquivo '{txt_file_path}' salvo com sucesso!")
        
def move_preview2(dx=0, dz=0):
    """Função para mover a esfera usando dx e dz como deslocamento."""
    global current_preview_position, preview_actor

    delta = 1.0  # Definir o quanto se mover por vez

    # Copiar a posição atual para realizar as atualizações
    new_position = np.copy(current_preview_position)

    # Atualizar a posição de acordo com os deslocamentos fornecidos
    new_position[0] += dx * delta  # Deslocamento no eixo X
    new_position[2] += dz * delta  # Deslocamento no eixo Z

    # Manter a posição Y atual e encontrar o ponto com o menor Y na malha
    new_position[1] = find_lowest_y_point(new_position)[1]  # Atualiza o Y baseado na malha

    # Atualizar a posição da esfera de preview
    preview_actor.SetPosition(new_position)
    current_preview_position = new_position  # Atualizar a posição atual
    plotter.render()  # Atualizar a visualização


# Interface gráfica adicional
class ControlWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Controle dos Eletrodos")
        self.setGeometry(100, 100, 300, 400)

        layout = QVBoxLayout()
        move_layout = QVBoxLayout()
        h_layout1 = QHBoxLayout()
        h_layout2 = QHBoxLayout()

        # Botões de movimento
        btn_up = QPushButton("↑")
        btn_down = QPushButton("↓")
        btn_left = QPushButton("←")
        btn_right = QPushButton("→")

        # Criar temporizadores para cada botão
        timer_up = QTimer()
        timer_down = QTimer()
        timer_left = QTimer()
        timer_right = QTimer()

        # Conectar os temporizadores à função move_preview2
        timer_up.timeout.connect(lambda: move_preview2(dz=1.0))
        timer_down.timeout.connect(lambda: move_preview2(dz=-1.0))
        timer_left.timeout.connect(lambda: move_preview2(dx=-1.0))
        timer_right.timeout.connect(lambda: move_preview2(dx=1.0))

        # Configurar os botões para iniciar e parar os temporizadores
        btn_up.pressed.connect(lambda: timer_up.start(100))     # Chama a função a cada 100ms enquanto pressionado
        btn_up.released.connect(timer_up.stop)                 # Para o timer quando o botão é solto

        btn_down.pressed.connect(lambda: timer_down.start(100))
        btn_down.released.connect(timer_down.stop)

        btn_left.pressed.connect(lambda: timer_left.start(100))
        btn_left.released.connect(timer_left.stop)

        btn_right.pressed.connect(lambda: timer_right.start(100))
        btn_right.released.connect(timer_right.stop)

        h_layout1.addWidget(btn_left)
        h_layout1.addWidget(btn_up)
        h_layout1.addWidget(btn_right)
        h_layout2.addWidget(btn_down)

        move_layout.addLayout(h_layout1)
        move_layout.addLayout(h_layout2)

        # Botões principais
        button_add = QPushButton("Adicionar Eletrodo")
        button_remove = QPushButton("Remover Último Eletrodo")
        button_save = QPushButton("Salvar")
        button_close = QPushButton("Fechar")

        button_add.clicked.connect(add_electrode)
        button_remove.clicked.connect(remove_last_electrode)
        button_save.clicked.connect(save_files)
        button_close.clicked.connect(lambda: sys.exit(0))

        layout.addLayout(move_layout)
        layout.addWidget(button_add)
        layout.addWidget(button_remove)
        layout.addWidget(button_save)
        layout.addWidget(button_close)

        # Criar checkboxes para alternar a visibilidade das malhas
        self.checkboxes = []
        for i, filename in enumerate(vtp_files):
            checkbox = QCheckBox(f"{os.path.basename(filename)}")
            checkbox.setChecked(True)  # Começam ticados (opacidade 0.7)
            checkbox.stateChanged.connect(lambda state, idx=i: self.toggle_mesh_visibility(idx, state))
            self.checkboxes.append(checkbox)
            layout.addWidget(checkbox)

        self.setLayout(layout)

    def toggle_mesh_visibility(self, index, state):
        """Alterna a opacidade da mesh com base no checkbox"""
        if state == 2:  # Marcado (opacidade normal)
            mesh_actors[index].GetProperty().SetOpacity(0.7)
        else:  # Desmarcado (invisível)
            mesh_actors[index].GetProperty().SetOpacity(0.0)
        plotter.render()  # Atualizar visualização
        
# Criar e mostrar a segunda janela
control_window = ControlWindow()
control_window.show()
        
    
# Conectar o clique do botão esquerdo à função on_left_click
plotter.iren.add_observer("LeftButtonPressEvent", on_left_click)

# Conectar as teclas do teclado para mover a esfera de preview e capturar eventos de tecla
plotter.iren.add_observer("KeyPressEvent", move_preview)
plotter.iren.add_observer("KeyPressEvent", capture_key_events)

# Mostrar o plotter com a funcionalidade de clique
plotter.show()
app.exec_()