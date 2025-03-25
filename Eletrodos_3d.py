import pyvista as pv
import numpy as np
import vtk
import os
import sys
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QCheckBox, QComboBox
)
from PyQt5.QtCore import QTimer

app = QApplication([])

# -----------------------------
#  SELEÇÃO DE PASTA E LEITURA
# -----------------------------
folder_path = QFileDialog.getExistingDirectory(None, "Selecionar Pasta com Arquivos VTP")
if not folder_path:
    raise Exception("Nenhuma pasta selecionada.")

vtp_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.vtp')]
if not vtp_files:
    raise Exception("Nenhum arquivo .vtp encontrado na pasta selecionada.")

color_palette = [
    [1.0, 0.0, 0.0],   # Vermelho
    [0.0, 1.0, 0.0],   # Verde
    [1.0, 1.0, 0.0],   # Amarelo
    [0.0, 1.0, 1.0],   # Ciano
    [0.5, 0.5, 0.5],   # Cinza médio
    [0.5, 0.0, 0.0],   # Marrom escuro avermelhado
    [0.0, 0.5, 0.0],   # Verde escuro
    [0.75, 0.75, 0.0], # Amarelo Ouro
    [0.75, 0.25, 0.0], # Laranja escuro
    [0.25, 0.75, 0.5], # Verde água suave
    [0.75, 0.5, 0.25], # Marrom claro
    [0.5, 0.5, 0.0],   # Mostarda
    [0.75, 0.75, 0.75],# Cinza claro
    [1.0, 0.5, 0.0],   # Laranja
    [0.0, 1.0, 0.5],   # Verde limão suave
    [1.0, 0.25, 0.25], # Vermelho suave
    [0.25, 1.0, 0.25], # Verde vibrante
    [1.0, 0.75, 0.25], # Laranja claro
    [0.25, 1.0, 0.75], # Verde-água
    [0.5, 0.75, 0.25], # Verde amarelado
    [0.75, 0.5, 0.5],  # Rosa queimado
    [0.5, 0.75, 0.5],  # Verde acinzentado
    [0.25, 0.25, 0.25],# Cinza escuro
    [0.75, 0.6, 0.0],  # Laranja queimado
    [0.4, 0.7, 0.3],   # Verde musgo
    [0.0, 0.6, 0.6],   # Verde azulado médio
    [0.8, 0.4, 0.0],   # Terracota
    [0.6, 0.6, 0.0]    # Mostarda escura
]

all_meshes = []
for file_path in vtp_files:
    mesh = pv.read(file_path)
    if 'Normals' not in mesh.point_data:
        mesh.compute_normals(inplace=True)
    all_meshes.append(mesh)

# -----------------------------
#  PLOTTER E MALHAS
# -----------------------------
plotter = pv.Plotter()

# Posição inicial da câmera
INITIAL_CAM_POS = (33.92955911088626, -868.6406385699812, 72.59394144121086)
INITIAL_CAM_FOCAL = (15.194366455078125, -14.439411163330078, -25.041778564453125)

plotter.camera.position = INITIAL_CAM_POS
plotter.camera.focal_point = INITIAL_CAM_FOCAL
plotter.camera.azimuth = 0.0
plotter.camera.elevation = 0.0

mesh_actors = []
for i, mesh in enumerate(all_meshes):
    color = color_palette[i % len(color_palette)]
    actor = plotter.add_mesh(
        mesh, color=color, opacity=0.7, 
        label=f"Malha {i+1}: {os.path.basename(vtp_files[i])}"
    )
    mesh_actors.append(actor)

plotter.add_legend()

multi_block = pv.MultiBlock(all_meshes)
mesh_torso = multi_block.combine()
plotter.add_mesh(mesh_torso, opacity=0)

plotter.add_legend()

# -----------------------------
#  ESFERA DE PREVIEW
# -----------------------------
sphere_radius = 6
preview_sphere = pv.Sphere(radius=sphere_radius)
preview_actor = plotter.add_mesh(preview_sphere, color='blue', opacity=1)
initial_position = np.mean(mesh_torso.points, axis=0)
preview_actor.SetPosition(initial_position)

line_actor = None
is_preview_active = True
current_preview_position = np.array(initial_position)

# =========================================
#  ESTRUTURAS GLOBAIS P/ ELETRODOS
# =========================================
electrodes = []       # [(label, (x, y, z)), ...] na ordem de criação
electrode_actors = {} # label -> (sphere_actor, text_actor)

is_space_pressed = False

# --------------------------------------
# HELPER: cria ou substitui 1 eletrodo
# --------------------------------------
def create_or_replace_electrode(label, position):
    """
    Remove marca antiga (caso exista) e cria novo 'label' em 'position'.
    Adiciona/atualiza a lista electrodes e o dicionário electrode_actors.

    REGRAS ESPECIAIS:
    - Se label == 'COL', sempre insere no INÍCIO de electrodes (primeiro da fila).
    - Caso contrário, insere ao final.
    """
    global electrodes, electrode_actors

    # Se o label já existe, remove do plotter e da lista 'electrodes'
    if label in electrode_actors:
        old_sphere, old_text = electrode_actors[label]
        plotter.remove_actor(old_sphere)
        plotter.remove_actor(old_text)
        # Remove qualquer entrada antiga do label em "electrodes"
        electrodes = [(lbl, pos) for (lbl, pos) in electrodes if lbl != label]

    # Cria nova esfera
    sphere_actor = plotter.add_mesh(
        pv.Sphere(radius=3, center=position),
        color='green', opacity=1.0
    )
    # Cria o texto (rótulo) com fonte menor
    text_actor = plotter.add_point_labels(
        [position],
        [label],
        font_size=8,   # Texto menor
        point_size=5,  # Ponto de referência menor
        always_visible=True
    )

    electrode_actors[label] = (sphere_actor, text_actor)

    # Se for 'COL', insere no INÍCIO da fila
    if label == 'COL':
        electrodes.insert(0, (label, position))
    else:
        electrodes.append((label, position))

    print(f"\nEletrodo ({label}) adicionado em: "
          f"X={position[0]:.2f}, Y={position[1]:.2f}, Z={position[2]:.2f}")

# -----------------------------
#  FUNÇÕES PRINCIPAIS
# -----------------------------
def add_electrode():
    """
    Quando "Adicionar Eletrodo" é clicado:
      - Se label == "COL", apenas cria/atualiza esse ponto e insere no início da fila.
      - Se label == "LA":
         1) Marca LA na posição do preview
         2) Se 'COL' existir, faz marcação automática de RA, LL, RL:
            RA: reflexo de LA em rel. à COL
            LL: LA_x-50, LA_y, LA_z-600
            RL: reflexo de LL em rel. à COL
      - Caso contrário, marca só esse label (V1, V2, etc.).
    """
    global preview_actor
    label = control_window.combo_label.currentText()
    pos = preview_actor.GetPosition()

    if label == "COL":
        # Apenas criar/atualizar a coluna no preview
        create_or_replace_electrode("COL", pos)

    elif label == "LA":
        # 1) Cria LA
        create_or_replace_electrode("LA", pos)

        # 2) Se COL estiver marcado, cria RA, LL, RL
        if "COL" in electrode_actors:
            # Pegamos a posição da coluna
            _, col_pos = next(((lbl, p) for lbl, p in electrodes if lbl == "COL"), (None, None))
            if col_pos is not None:
                x_col = col_pos[0]

                # RA => reflexo de LA em rel. à COL
                # RA_x = 2*x_col - LA_x
                ra_x = 2*x_col - pos[0]
                ra_pos = (ra_x, pos[1], pos[2])
                create_or_replace_electrode("RA", ra_pos)

                # LL => (LA_x-50, LA_y, LA_z-600) [igual antes]
                ll_pos = (pos[0] - 50, pos[1], pos[2] - 600)
                create_or_replace_electrode("LL", ll_pos)

                # RL => reflexo de LL em rel. à COL
                # RL_x = 2*x_col - ll_x
                rl_x = 2*x_col - ll_pos[0]
                rl_pos = (rl_x, ll_pos[1], ll_pos[2])
                create_or_replace_electrode("RL", rl_pos)

        # Se COL não existe, não faz RA, LL, RL
        # (ou seja, só LA)

    else:
        # V1, V2, V3, V4, V5, V6, RA, LL, RL etc. => Marca só esse label
        create_or_replace_electrode(label, pos)

    plotter.render()

def remove_last_electrode():
    """
    Remove o último eletrodo adicionado,
    apaga do plotter e do dicionário de atores.
    """
    global electrodes, electrode_actors
    if not electrodes:
        print("Nenhum eletrodo para remover.")
        return

    # "Pop" retira o último item (label, posição)
    last_label, last_pos = electrodes.pop()

    if last_label in electrode_actors:
        sphere_actor, text_actor = electrode_actors[last_label]
        plotter.remove_actor(sphere_actor)
        plotter.remove_actor(text_actor)
        del electrode_actors[last_label]

    print(f"Último eletrodo removido: {last_label}.")
    print(f"Número de eletrodos restantes: {len(electrodes)}")

def find_lowest_y_point(mouse_position, tol=None):
    closest_point_id = mesh_torso.find_closest_point(mouse_position)
    closest_point = mesh_torso.points[closest_point_id]
    x, z = closest_point[0], closest_point[2]
    if tol is None:
        bounds = mesh_torso.bounds
        tol = (bounds[1] - bounds[0]) * 0.01
    mask = np.linalg.norm(mesh_torso.points[:, [0,2]] - np.array([x, z]), axis=1) < tol
    candidates = mesh_torso.points[mask]
    print(f"Encontrados {len(candidates)} pontos candidatos com tol={tol}")

    if candidates.size > 0:
        lowest_y_point = candidates[np.argmin(candidates[:, 1])]
        return lowest_y_point
    else:
        return closest_point

def find_lowest_y_for_xz(candidate_point, tol=1.0):
    x_candidate, z_candidate = candidate_point[0], candidate_point[2]
    mask = (np.abs(mesh_torso.points[:, 0] - x_candidate) < tol) & \
           (np.abs(mesh_torso.points[:, 2] - z_candidate) < tol)
    candidates = mesh_torso.points[mask]
    if candidates.size > 0:
        lowest_y_point = candidates[np.argmin(candidates[:, 1])]
        return lowest_y_point
    else:
        return candidate_point

def on_left_click(iren, event):
    global is_preview_active, current_preview_position
    if is_preview_active and not is_space_pressed:
        mouse_pos = plotter.pick_mouse_position()
        if mouse_pos is not None:
            candidate_point = find_lowest_y_point(mouse_pos)
            lowest_y_point = find_lowest_y_for_xz(candidate_point, tol=1.0)
            if not np.isclose(candidate_point[1], lowest_y_point[1], atol=1e-3):
                print("Atualizando posição para o ponto com menor Y na vizinhança.")
                candidate_point = lowest_y_point
            preview_actor.SetPosition(candidate_point)
            current_preview_position = candidate_point
            plotter.render()

def move_preview(iren, event):
    global current_preview_position, preview_actor, is_space_pressed
    if is_space_pressed:
        return

    delta = 1.0
    key = iren.GetKeySym()

    new_position = np.copy(current_preview_position)

    if key == 'Up':
        new_position[2] += delta
    elif key == 'Down':
        new_position[2] -= delta
    elif key == 'Left':
        new_position[0] -= delta
    elif key == 'Right':
        new_position[0] += delta
    else:
        return

    new_position[1] = find_lowest_y_point(new_position)[1]
    preview_actor.SetPosition(new_position)
    current_preview_position = new_position
    plotter.render()

def capture_key_events(iren, event):
    global is_space_pressed, key
    key = iren.GetKeySym()
    print(f"Tecla pressionada: {key}")

    if key == 'Return':
        add_electrode()
    elif key == 'Backspace':
        remove_last_electrode()
    elif key == 'space':
        is_space_pressed = not is_space_pressed
        print("Modo de controle de câmera " + ("ativado." if is_space_pressed else "desativado."))
    elif key == 's':
        save_files()

def save_files():
    global electrodes
    txt_file_path, _ = QFileDialog.getSaveFileName(
        None, "Salvar coordenadas dos eletrodos", 
        "", "TXT files (*.txt);;All files (*)"
    )
    if txt_file_path:
        with open(txt_file_path, 'w') as f:
            for idx, (label, position) in enumerate(electrodes, start=1):
                f.write(f"Eletrodo #{idx} ({label}): "
                        f"X={position[0]:.2f}, Y={position[1]:.2f}, Z={position[2]:.2f}\n")
        print(f"Arquivo '{txt_file_path}' salvo com sucesso!")

def move_preview2(dx=0, dz=0):
    global current_preview_position, preview_actor
    delta = 1.0
    new_position = np.copy(current_preview_position)
    new_position[0] += dx * delta
    new_position[2] += dz * delta
    new_position[1] = find_lowest_y_point(new_position)[1]
    preview_actor.SetPosition(new_position)
    current_preview_position = new_position
    plotter.render()

# ---------------------------------------------------
#   JANELA 1: CONTROLE DE ELETRODOS
# ---------------------------------------------------
class ControlWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Controle dos Eletrodos")
        self.setGeometry(1100, 100, 300, 550)

        layout = QVBoxLayout()

        # ComboBox com rótulos
        self.combo_label = QComboBox()
        # Inclui "COL" (coluna) como opção
        # Lembre-se de escolher "COL" primeiro, se quiser automatizar LA -> RA/LL/RL
        electrode_labels = ["COL","V1","V2","V3","V4","V5","V6","LA","RA","LL","RL"]
        self.combo_label.addItems(electrode_labels)
        layout.addWidget(self.combo_label)

        move_layout = QVBoxLayout()
        h_layout1 = QHBoxLayout()
        h_layout2 = QHBoxLayout()

        btn_up = QPushButton("↑")
        btn_down = QPushButton("↓")
        btn_left = QPushButton("←")
        btn_right = QPushButton("→")

        timer_up = QTimer()
        timer_down = QTimer()
        timer_left = QTimer()
        timer_right = QTimer()

        timer_up.timeout.connect(lambda: move_preview2(dz=1.0))
        timer_down.timeout.connect(lambda: move_preview2(dz=-1.0))
        timer_left.timeout.connect(lambda: move_preview2(dx=-1.0))
        timer_right.timeout.connect(lambda: move_preview2(dx=1.0))

        btn_up.pressed.connect(lambda: timer_up.start(100))
        btn_up.released.connect(timer_up.stop)
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

        self.checkboxes = []
        for i, filename in enumerate(vtp_files):
            checkbox = QCheckBox(f"{os.path.basename(filename)}")
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(lambda state, idx=i: self.toggle_mesh_visibility(idx, state))
            self.checkboxes.append(checkbox)
            layout.addWidget(checkbox)

        self.setLayout(layout)

    def toggle_mesh_visibility(self, index, state):
        if state == 2:  # Marcado
            mesh_actors[index].GetProperty().SetOpacity(0.7)
        else:
            mesh_actors[index].GetProperty().SetOpacity(0.0)
        plotter.render()

# ---------------------------------------------------
#  JANELA 2: CONTROLE DE CÂMERA (com reset)
# ---------------------------------------------------
class CameraControlWindow(QWidget):
    def __init__(self, plotter):
        super().__init__()
        self.plotter = plotter
        self.setWindowTitle("Controle de Câmera")
        self.setGeometry(1450, 100, 300, 450)

        main_layout = QVBoxLayout()

        def move_camera(dx=0, dy=0, dz=0):
            cam_pos = list(self.plotter.camera.position)
            cam_pos[0] += dx
            cam_pos[1] += dy
            cam_pos[2] += dz
            self.plotter.camera.position = tuple(cam_pos)
            self.plotter.render()
            self.print_camera_state()

        def rotate_camera(direction):
            if direction == 'left':
                self.plotter.camera.azimuth += 5
            elif direction == 'right':
                self.plotter.camera.azimuth -= 5
            elif direction == 'up':
                self.plotter.camera.elevation += 5
            elif direction == 'down':
                self.plotter.camera.elevation -= 5

            self.plotter.render()
            self.print_camera_state()

        def reset_camera():
            self.plotter.camera.position = INITIAL_CAM_POS
            self.plotter.camera.focal_point = INITIAL_CAM_FOCAL
            self.plotter.camera.azimuth = 0.0
            self.plotter.camera.elevation = 0.0
            self.plotter.render()
            self.print_camera_state()

        # Translação
        btn_cam_x_pos = QPushButton("Camera X+")
        timer_cam_x_pos = QTimer()
        timer_cam_x_pos.timeout.connect(lambda: move_camera(dx=10))
        def on_press_cam_x_pos():
            move_camera(dx=10)
            timer_cam_x_pos.start(100)
        btn_cam_x_pos.pressed.connect(on_press_cam_x_pos)
        btn_cam_x_pos.released.connect(timer_cam_x_pos.stop)

        btn_cam_x_neg = QPushButton("Camera X-")
        timer_cam_x_neg = QTimer()
        timer_cam_x_neg.timeout.connect(lambda: move_camera(dx=-10))
        def on_press_cam_x_neg():
            move_camera(dx=-10)
            timer_cam_x_neg.start(100)
        btn_cam_x_neg.pressed.connect(on_press_cam_x_neg)
        btn_cam_x_neg.released.connect(timer_cam_x_neg.stop)

        btn_cam_y_pos = QPushButton("Camera Y+")
        timer_cam_y_pos = QTimer()
        timer_cam_y_pos.timeout.connect(lambda: move_camera(dy=10))
        def on_press_cam_y_pos():
            move_camera(dy=10)
            timer_cam_y_pos.start(100)
        btn_cam_y_pos.pressed.connect(on_press_cam_y_pos)
        btn_cam_y_pos.released.connect(timer_cam_y_pos.stop)

        btn_cam_y_neg = QPushButton("Camera Y-")
        timer_cam_y_neg = QTimer()
        timer_cam_y_neg.timeout.connect(lambda: move_camera(dy=-10))
        def on_press_cam_y_neg():
            move_camera(dy=-10)
            timer_cam_y_neg.start(100)
        btn_cam_y_neg.pressed.connect(on_press_cam_y_neg)
        btn_cam_y_neg.released.connect(timer_cam_y_neg.stop)

        btn_cam_z_pos = QPushButton("Camera Z+")
        timer_cam_z_pos = QTimer()
        timer_cam_z_pos.timeout.connect(lambda: move_camera(dz=10))
        def on_press_cam_z_pos():
            move_camera(dz=10)
            timer_cam_z_pos.start(100)
        btn_cam_z_pos.pressed.connect(on_press_cam_z_pos)
        btn_cam_z_pos.released.connect(timer_cam_z_pos.stop)

        btn_cam_z_neg = QPushButton("Camera Z-")
        timer_cam_z_neg = QTimer()
        timer_cam_z_neg.timeout.connect(lambda: move_camera(dz=-10))
        def on_press_cam_z_neg():
            move_camera(dz=-10)
            timer_cam_z_neg.start(100)
        btn_cam_z_neg.pressed.connect(on_press_cam_z_neg)
        btn_cam_z_neg.released.connect(timer_cam_z_neg.stop)

        main_layout.addWidget(btn_cam_x_pos)
        main_layout.addWidget(btn_cam_x_neg)
        main_layout.addWidget(btn_cam_y_pos)
        main_layout.addWidget(btn_cam_y_neg)
        main_layout.addWidget(btn_cam_z_pos)
        main_layout.addWidget(btn_cam_z_neg)

        # Rotação
        btn_cam_left = QPushButton("Ângulo Esq.")
        timer_cam_left = QTimer()
        timer_cam_left.timeout.connect(lambda: rotate_camera('left'))
        def on_press_cam_left():
            rotate_camera('left')
            timer_cam_left.start(100)
        btn_cam_left.pressed.connect(on_press_cam_left)
        btn_cam_left.released.connect(timer_cam_left.stop)

        btn_cam_right = QPushButton("Ângulo Dir.")
        timer_cam_right = QTimer()
        timer_cam_right.timeout.connect(lambda: rotate_camera('right'))
        def on_press_cam_right():
            rotate_camera('right')
            timer_cam_right.start(100)
        btn_cam_right.pressed.connect(on_press_cam_right)
        btn_cam_right.released.connect(timer_cam_right.stop)

        btn_cam_up = QPushButton("Ângulo Cima")
        timer_cam_up = QTimer()
        timer_cam_up.timeout.connect(lambda: rotate_camera('up'))
        def on_press_cam_up():
            rotate_camera('up')
            timer_cam_up.start(100)
        btn_cam_up.pressed.connect(on_press_cam_up)
        btn_cam_up.released.connect(timer_cam_up.stop)

        btn_cam_down = QPushButton("Ângulo Baixo")
        timer_cam_down = QTimer()
        timer_cam_down.timeout.connect(lambda: rotate_camera('down'))
        def on_press_cam_down():
            rotate_camera('down')
            timer_cam_down.start(100)
        btn_cam_down.pressed.connect(on_press_cam_down)
        btn_cam_down.released.connect(timer_cam_down.stop)

        main_layout.addWidget(btn_cam_left)
        main_layout.addWidget(btn_cam_right)
        main_layout.addWidget(btn_cam_up)
        main_layout.addWidget(btn_cam_down)

        # Botão de Reset
        btn_reset = QPushButton("Reset Camera")
        btn_reset.clicked.connect(reset_camera)
        main_layout.addWidget(btn_reset)

        self.setLayout(main_layout)

    def print_camera_state(self):
        cam = self.plotter.camera
        pos = cam.position
        focal = cam.focal_point
        az = cam.azimuth
        el = cam.elevation
        print("----- Câmera -----")
        print(f"  Position = {pos}")
        print(f"  Focal point = {focal}")
        print(f"  Azimuth = {az:.2f}°, Elevation = {el:.2f}°")
        print("------------------")

# ----------------------------------------
#  CRIANDO JANELAS E INICIANDO APLICAÇÃO
# ----------------------------------------
control_window = ControlWindow()
control_window.show()

camera_window = CameraControlWindow(plotter)
camera_window.show()

plotter.iren.add_observer("LeftButtonPressEvent", on_left_click)
plotter.iren.add_observer("KeyPressEvent", move_preview)
plotter.iren.add_observer("KeyPressEvent", capture_key_events)

plotter.show()
app.exec_()
