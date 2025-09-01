import pyvista as pv
import numpy as np
import vtk
import os
import sys
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QCheckBox, QComboBox, QScrollArea, QSizePolicy, QLabel
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
    [1.0, 0.0, 0.0],   [0.0, 1.0, 0.0],   [1.0, 1.0, 0.0],
    [0.0, 1.0, 1.0],   [0.5, 0.5, 0.5],   [0.5, 0.0, 0.0],
    [0.0, 0.5, 0.0],   [0.75, 0.75, 0.0], [0.75, 0.25, 0.0],
    [0.25, 0.75, 0.5], [0.75, 0.5, 0.25], [0.5, 0.5, 0.0],
    [0.75, 0.75, 0.75],[1.0, 0.5, 0.0],   [0.0, 1.0, 0.5],
    [1.0, 0.25, 0.25], [0.25, 1.0, 0.25], [1.0, 0.75, 0.25],
    [0.25, 1.0, 0.75], [0.5, 0.75, 0.25], [0.75, 0.5, 0.5],
    [0.5, 0.75, 0.5],  [0.25, 0.25, 0.25],[0.75, 0.6, 0.0],
    [0.4, 0.7, 0.3],   [0.0, 0.6, 0.6],   [0.8, 0.4, 0.0],
    [0.6, 0.6, 0.0]
]

# Separar malhas de torso (normais) e "Linha" (visuais)
torso_meshes = []
torso_filenames = []
linha_raw_points = []   # lista de dicts com info para gerar plano depois

for file_path in vtp_files:
    name = os.path.basename(file_path)
    mesh = pv.read(file_path)

    if 'linha' in name.lower():
        if mesh.n_points >= 3:
            pts = np.asarray(mesh.points)
            linha_raw_points.append({'name': name, 'points': pts})
        else:
            print(f"[AVISO] '{name}' tem menos de 3 pontos - não é possível ajustar um plano.")
    else:
        if 'Normals' not in mesh.point_data:
            mesh.compute_normals(inplace=True)
        torso_meshes.append(mesh)
        torso_filenames.append(name)

if not torso_meshes:
    raise Exception("Nenhuma malha (sem 'Linha' no nome) foi encontrada.")

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

file_actor_map = {}

# Renderizar malhas
torso_actors = []
for i, mesh in enumerate(torso_meshes):
    color = color_palette[i % len(color_palette)]
    actor = plotter.add_mesh(mesh, color=color, opacity=0.7, label=f"Torso {i+1}: {torso_filenames[i]}")
    torso_actors.append(actor)
    file_actor_map[torso_filenames[i]] = actor

plotter.add_legend()

# --- (1) FILTRAR APENAS MALHAS "TORSO" PARA POSICIONAMENTO DE ELETRODOS ---
torso_indices_filtrados = [i for i, n in enumerate(torso_filenames) if 'torso' in n.lower()]
if torso_indices_filtrados:
    torso_meshes_filtrados = [torso_meshes[i] for i in torso_indices_filtrados]
else:
    # fallback: se nenhuma contém "torso", usa todas as malhas existentes
    print("[AVISO] Nenhum arquivo com 'torso' no nome; usando todas as malhas para o posicionamento.")
    torso_meshes_filtrados = torso_meshes

multi_block = pv.MultiBlock(torso_meshes_filtrados)
mesh_torso = multi_block.combine()
plotter.add_mesh(mesh_torso, opacity=0)  # invisível, apenas para projeções

# -----------------------------
#  PLANOS "LINHA" (quadrados, eixos 90°)
# -----------------------------
def best_fit_plane(points: np.ndarray):
    c = points.mean(axis=0)
    A = points - c
    _, _, vt = np.linalg.svd(A, full_matrices=False)
    normal = vt[-1, :]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    return c, normal

def snap_normal_to_axis(normal: np.ndarray) -> np.ndarray:
    # escolhe eixo global mais próximo (±X, ±Y, ±Z)
    idx = np.argmax(np.abs(normal))
    snapped = np.zeros(3)
    snapped[idx] = np.sign(normal[idx]) if normal[idx] != 0 else 1.0
    return snapped

bounds = mesh_torso.bounds
size_max = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]) * 1.2
if size_max <= 0:
    size_max = 100.0

linha_plane_meshes = []
linha_plane_actors = []
for i, item in enumerate(linha_raw_points):
    center, normal = best_fit_plane(item['points'])
    snapped_dir = snap_normal_to_axis(normal)  # <- (3) alinhar ao eixo mais próximo
    # i_size = j_size para quadrado
    plane = pv.Plane(center=center, direction=snapped_dir, i_size=size_max, j_size=size_max, i_resolution=1, j_resolution=1)
    linha_plane_meshes.append(plane)
    color = color_palette[(len(torso_meshes) + i) % len(color_palette)]
    actor = plotter.add_mesh(plane, color=color, opacity=0.25, label=f"Plano: {item['name']}")
    linha_plane_actors.append(actor)
    file_actor_map[item['name']] = actor

plotter.add_legend(size=(0, 0))

# -----------------------------
#  ESFERA DE PREVIEW / ESTADO
# -----------------------------
sphere_radius = 6
preview_sphere = pv.Sphere(radius=sphere_radius)
preview_actor = plotter.add_mesh(preview_sphere, color='blue', opacity=1)
initial_position = np.mean(mesh_torso.points, axis=0)
preview_actor.SetPosition(initial_position)

line_actor = None
is_preview_active = True
current_preview_position = np.array(initial_position)

# (2) Toggle: fixar no menor Y (frente) ou permitir laterais
snap_to_lowest_y = True  # estado global; controlado por checkbox

# =========================================
#  ESTRUTURAS GLOBAIS P/ ELETRODOS
# =========================================
electrodes = []       # [(label, (x, y, z)), ...]
electrode_actors = {} # label -> (sphere_actor, text_actor)
is_space_pressed = False

# --------------------------------------
# HELPER: cria/substitui 1 eletrodo
# --------------------------------------
def create_or_replace_electrode(label, position):
    global electrodes, electrode_actors

    if label in electrode_actors:
        old_sphere, old_text = electrode_actors[label]
        plotter.remove_actor(old_sphere)
        plotter.remove_actor(old_text)
        electrodes = [(lbl, pos) for (lbl, pos) in electrodes if lbl != label]

    sphere_actor = plotter.add_mesh(pv.Sphere(radius=3, center=position), color='green', opacity=1.0)
    text_actor = plotter.add_point_labels([position], [label], font_size=8, point_size=5, always_visible=True)

    electrode_actors[label] = (sphere_actor, text_actor)

    if label == 'COL':
        electrodes.insert(0, (label, position))
    else:
        electrodes.append((label, position))

    print(f"\nEletrodo ({label}) adicionado em: X={position[0]:.2f}, Y={position[1]:.2f}, Z={position[2]:.2f}")

# -----------------------------
#  FUNÇÕES DE PROJEÇÃO/PICK
# -----------------------------
def find_lowest_y_point(mouse_or_xyz, tol=None):
    """
    Usa SOMENTE mesh_torso (filtrado) e força Y mínimo para o mesmo (x,z).
    mouse_or_xyz pode ser coord 3D (np.array/list) ou posição de mouse retornada pelo pick.
    """
    if isinstance(mouse_or_xyz, (list, tuple, np.ndarray)) and len(mouse_or_xyz) == 3:
        ref = np.array(mouse_or_xyz, dtype=float)
    else:
        ref = plotter.pick_mouse_position()
        if ref is None:
            # fallback: posição atual
            return np.array(current_preview_position)

    closest_point_id = mesh_torso.find_closest_point(ref)
    closest_point = mesh_torso.points[closest_point_id]
    x, z = closest_point[0], closest_point[2]
    if tol is None:
        bounds = mesh_torso.bounds
        tol = (bounds[1] - bounds[0]) * 0.01
    mask = np.linalg.norm(mesh_torso.points[:, [0, 2]] - np.array([x, z]), axis=1) < tol
    candidates = mesh_torso.points[mask]
    if candidates.size > 0:
        lowest_y_point = candidates[np.argmin(candidates[:, 1])]
        return lowest_y_point
    else:
        return closest_point

def closest_point_on_surface(pos3d):
    """Retorna o ponto da superfície (mesh_torso) mais próximo a pos3d (3D)."""
    pid = mesh_torso.find_closest_point(pos3d)
    return mesh_torso.points[pid]

def project_to_surface(pos3d, snap_front=True):
    """
    Se snap_front=True: força menor Y no mesmo (x,z) -> "frente".
    Caso contrário: pega o ponto mais próximo na superfície (permite laterais).
    """
    if snap_front:
        return find_lowest_y_point(pos3d)
    else:
        return closest_point_on_surface(pos3d)

# -----------------------------
#  HANDLERS PRINCIPAIS
# -----------------------------
def add_electrode():
    global preview_actor
    label = control_window.combo_label.currentText()
    pos = preview_actor.GetPosition()

    if label == "COL":
        create_or_replace_electrode("COL", pos)

    elif label == "LA":
        create_or_replace_electrode("LA", pos)
        if "COL" in electrode_actors:
            _, col_pos = next(((lbl, p) for lbl, p in electrodes if lbl == "COL"), (None, None))
            if col_pos is not None:
                x_col = col_pos[0]
                ra_x = 2*x_col - pos[0]
                ra_pos = (ra_x, pos[1], pos[2])
                create_or_replace_electrode("RA", ra_pos)

                ll_pos = (pos[0] - 50, pos[1], pos[2] - 600)
                create_or_replace_electrode("LL", ll_pos)

                rl_x = 2*x_col - ll_pos[0]
                rl_pos = (rl_x, ll_pos[1], ll_pos[2])
                create_or_replace_electrode("RL", rl_pos)
    else:
        create_or_replace_electrode(label, pos)

    plotter.render()

def remove_last_electrode():
    global electrodes, electrode_actors
    if not electrodes:
        print("Nenhum eletrodo para remover.")
        return
    last_label, last_pos = electrodes.pop()
    if last_label in electrode_actors:
        sphere_actor, text_actor = electrode_actors[last_label]
        plotter.remove_actor(sphere_actor)
        plotter.remove_actor(text_actor)
        del electrode_actors[last_label]
    print(f"Último eletrodo removido: {last_label}. Restantes: {len(electrodes)}")

def on_left_click(iren, event):
    global is_preview_active, current_preview_position
    if is_preview_active and not is_space_pressed:
        mouse_pos = plotter.pick_mouse_position()
        if mouse_pos is not None:
            if snap_to_lowest_y:
                target = find_lowest_y_point(mouse_pos)
            else:
                target = closest_point_on_surface(mouse_pos)
            preview_actor.SetPosition(target)
            current_preview_position = np.array(target)
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
    # (2) respeitar modo de snap
    snapped = project_to_surface(new_position, snap_front=snap_to_lowest_y)
    preview_actor.SetPosition(snapped)
    current_preview_position = np.array(snapped)
    plotter.render()

def capture_key_events(iren, event):
    global is_space_pressed
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
        None, "Salvar coordenadas dos eletrodos", "", "TXT files (*.txt);;All files (*)"
    )
    if txt_file_path:
        with open(txt_file_path, 'w') as f:
            for idx, (label, position) in enumerate(electrodes, start=1):
                f.write(f"Eletrodo #{idx} ({label}): X={position[0]:.2f}, Y={position[1]:.2f}, Z={position[2]:.2f}\n")
        print(f"Arquivo '{txt_file_path}' salvo com sucesso!")

def import_files():
    global electrodes, electrode_actors
    txt_file_path, _ = QFileDialog.getOpenFileName(
        None, "Importar coordenadas dos eletrodos", "", "TXT files (*.txt);;All files (*)"
    )
    if not txt_file_path:
        return
    for sphere_act, text_act in electrode_actors.values():
        plotter.remove_actor(sphere_act)
        plotter.remove_actor(text_act)
    electrodes.clear()
    electrode_actors.clear()
    with open(txt_file_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                label = line.split("(")[1].split(")")[0]
                coord_str = line.split(":")[1]
                x = float(coord_str.split("X=")[1].split(",")[0])
                y = float(coord_str.split("Y=")[1].split(",")[0])
                z = float(coord_str.split("Z=")[1])
                create_or_replace_electrode(label, (x, y, z))
            except Exception as e:
                print(f"Erro ao ler linha:\n  {line.strip()}\n→ {e}")
    plotter.render()
    print(f"Arquivo '{txt_file_path}' importado com sucesso!")

def move_preview2(dx=0, dz=0):
    global current_preview_position, preview_actor
    delta = 1.0
    new_position = np.copy(current_preview_position)
    new_position[0] += dx * delta
    new_position[2] += dz * delta
    snapped = project_to_surface(new_position, snap_front=snap_to_lowest_y)
    preview_actor.SetPosition(snapped)
    current_preview_position = np.array(snapped)
    plotter.render()

# ---------------------------------------------------
#   JANELA 1: CONTROLE DE ELETRODOS
# ---------------------------------------------------
class ControlWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Controle dos Eletrodos")
        self.setGeometry(1100, 100, 300, 580)

        main_layout = QVBoxLayout(self)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_area.setWidget(scroll_widget)
        layout = QVBoxLayout(scroll_widget)

        self.combo_label = QComboBox()
        electrode_labels = ["COL","V1","V2","V3","V4","V5","V6","LA","RA","LL","RL"]
        self.combo_label.addItems(electrode_labels)
        layout.addWidget(self.combo_label)

        # (2) Checkbox: Fixar na frente (menor Y)
        self.chk_snap = QCheckBox("Fixar na frente (menor Y)")
        self.chk_snap.setChecked(True)
        layout.addWidget(self.chk_snap)
        self.chk_snap.stateChanged.connect(self.on_toggle_snap)

        # Botões de movimentação com "press & hold"
        move_layout = QVBoxLayout()
        h_layout1 = QHBoxLayout()
        h_layout2 = QHBoxLayout()

        btn_up = QPushButton("↑")
        btn_down = QPushButton("↓")
        btn_left = QPushButton("←")
        btn_right = QPushButton("→")

        timer_up = QTimer(); timer_down = QTimer(); timer_left = QTimer(); timer_right = QTimer()
        timer_up.timeout.connect(lambda: move_preview2(dz=1.0))
        timer_down.timeout.connect(lambda: move_preview2(dz=-1.0))
        timer_left.timeout.connect(lambda: move_preview2(dx=-1.0))
        timer_right.timeout.connect(lambda: move_preview2(dx=1.0))

        btn_up.pressed.connect(lambda: (move_preview2(dz=1.0), timer_up.start(100)))
        btn_up.released.connect(timer_up.stop)
        btn_down.pressed.connect(lambda: (move_preview2(dz=-1.0), timer_down.start(100)))
        btn_down.released.connect(timer_down.stop)
        btn_left.pressed.connect(lambda: (move_preview2(dx=-1.0), timer_left.start(100)))
        btn_left.released.connect(timer_left.stop)
        btn_right.pressed.connect(lambda: (move_preview2(dx=1.0), timer_right.start(100)))
        btn_right.released.connect(timer_right.stop)

        h_layout1.addWidget(btn_left); h_layout1.addWidget(btn_up); h_layout1.addWidget(btn_right)
        h_layout2.addWidget(btn_down)
        move_layout.addLayout(h_layout1); move_layout.addLayout(h_layout2)
        layout.addLayout(move_layout)

        # Botões principais
        button_add = QPushButton("Adicionar Eletrodo")
        button_remove = QPushButton("Remover Último Eletrodo")
        button_save = QPushButton("Salvar")
        button_import = QPushButton("Importar")
        button_close = QPushButton("Fechar")

        button_add.clicked.connect(add_electrode)
        button_remove.clicked.connect(remove_last_electrode)
        button_save.clicked.connect(save_files)
        button_import.clicked.connect(import_files)
        button_close.clicked.connect(lambda: sys.exit(0))

        layout.addWidget(button_add)
        layout.addWidget(button_remove)
        layout.addWidget(button_save)
        layout.addWidget(button_import)
        layout.addWidget(button_close)

        # Checkboxes dos arquivos (visibilidade)
        self.checkboxes = []
        for i, filename in enumerate(vtp_files):
            checkbox = QCheckBox(f"{os.path.basename(filename)}")
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(lambda state, idx=i: self.toggle_mesh_visibility(idx, state))
            self.checkboxes.append(checkbox)
            layout.addWidget(checkbox)

        main_layout.addWidget(scroll_area)
        self.setFixedSize(300, 580)

    def on_toggle_snap(self, state):
        global snap_to_lowest_y
        snap_to_lowest_y = (state == 2)  # 2 = Checked
        mode = "frente (menor Y)" if snap_to_lowest_y else "livre (laterais permitidas)"
        print(f"[Modo de projeção] Agora: {mode}")

    def toggle_mesh_visibility(self, index, state):
        fname = os.path.basename(vtp_files[index])
        actor = file_actor_map.get(fname)
        if actor is None:
            print(f"[AVISO] Sem actor para '{fname}'.")
            return
        actor.GetProperty().SetOpacity(0.7 if state == 2 else 0.0)
        plotter.render()

# ---------------------------------------------------
#  JANELA 2: CONTROLE DE CÂMERA
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
            cam_pos[0] += dx; cam_pos[1] += dy; cam_pos[2] += dz
            self.plotter.camera.position = tuple(cam_pos)
            self.plotter.render(); self.print_camera_state()

        def rotate_camera(direction):
            if direction == 'left':   self.plotter.camera.azimuth += 5
            elif direction == 'right': self.plotter.camera.azimuth -= 5
            elif direction == 'up':    self.plotter.camera.elevation += 5
            elif direction == 'down':  self.plotter.camera.elevation -= 5
            self.plotter.render(); self.print_camera_state()

        def reset_camera():
            self.plotter.camera.position = INITIAL_CAM_POS
            self.plotter.camera.focal_point = INITIAL_CAM_FOCAL
            self.plotter.camera.azimuth = 0.0
            self.plotter.camera.elevation = 0.0
            self.plotter.render(); self.print_camera_state()

        # Translação
        btn_cam_x_pos = QPushButton("Camera X+"); timer_cam_x_pos = QTimer()
        timer_cam_x_pos.timeout.connect(lambda: move_camera(dx=10))
        btn_cam_x_pos.pressed.connect(lambda: (move_camera(dx=10), timer_cam_x_pos.start(100)))
        btn_cam_x_pos.released.connect(timer_cam_x_pos.stop)

        btn_cam_x_neg = QPushButton("Camera X-"); timer_cam_x_neg = QTimer()
        timer_cam_x_neg.timeout.connect(lambda: move_camera(dx=-10))
        btn_cam_x_neg.pressed.connect(lambda: (move_camera(dx=-10), timer_cam_x_neg.start(100)))
        btn_cam_x_neg.released.connect(timer_cam_x_neg.stop)

        btn_cam_y_pos = QPushButton("Camera Y+"); timer_cam_y_pos = QTimer()
        timer_cam_y_pos.timeout.connect(lambda: move_camera(dy=10))
        btn_cam_y_pos.pressed.connect(lambda: (move_camera(dy=10), timer_cam_y_pos.start(100)))
        btn_cam_y_pos.released.connect(timer_cam_y_pos.stop)

        btn_cam_y_neg = QPushButton("Camera Y-"); timer_cam_y_neg = QTimer()
        timer_cam_y_neg.timeout.connect(lambda: move_camera(dy=-10))
        btn_cam_y_neg.pressed.connect(lambda: (move_camera(dy=-10), timer_cam_y_neg.start(100)))
        btn_cam_y_neg.released.connect(timer_cam_y_neg.stop)

        btn_cam_z_pos = QPushButton("Camera Z+"); timer_cam_z_pos = QTimer()
        timer_cam_z_pos.timeout.connect(lambda: move_camera(dz=10))
        btn_cam_z_pos.pressed.connect(lambda: (move_camera(dz=10), timer_cam_z_pos.start(100)))
        btn_cam_z_pos.released.connect(timer_cam_z_pos.stop)

        btn_cam_z_neg = QPushButton("Camera Z-"); timer_cam_z_neg = QTimer()
        timer_cam_z_neg.timeout.connect(lambda: move_camera(dz=-10))
        btn_cam_z_neg.pressed.connect(lambda: (move_camera(dz=-10), timer_cam_z_neg.start(100)))
        btn_cam_z_neg.released.connect(timer_cam_z_neg.stop)

        main_layout.addWidget(btn_cam_x_pos); main_layout.addWidget(btn_cam_x_neg)
        main_layout.addWidget(btn_cam_y_pos); main_layout.addWidget(btn_cam_y_neg)
        main_layout.addWidget(btn_cam_z_pos); main_layout.addWidget(btn_cam_z_neg)

        # Rotação
        btn_cam_left = QPushButton("Ângulo Esq."); timer_cam_left = QTimer()
        timer_cam_left.timeout.connect(lambda: rotate_camera('left'))
        btn_cam_left.pressed.connect(lambda: (rotate_camera('left'), timer_cam_left.start(100)))
        btn_cam_left.released.connect(timer_cam_left.stop)

        btn_cam_right = QPushButton("Ângulo Dir."); timer_cam_right = QTimer()
        timer_cam_right.timeout.connect(lambda: rotate_camera('right'))
        btn_cam_right.pressed.connect(lambda: (rotate_camera('right'), timer_cam_right.start(100)))
        btn_cam_right.released.connect(timer_cam_right.stop)

        btn_cam_up = QPushButton("Ângulo Cima"); timer_cam_up = QTimer()
        timer_cam_up.timeout.connect(lambda: rotate_camera('up'))
        btn_cam_up.pressed.connect(lambda: (rotate_camera('up'), timer_cam_up.start(100)))
        btn_cam_up.released.connect(timer_cam_up.stop)

        btn_cam_down = QPushButton("Ângulo Baixo"); timer_cam_down = QTimer()
        timer_cam_down.timeout.connect(lambda: rotate_camera('down'))
        btn_cam_down.pressed.connect(lambda: (rotate_camera('down'), timer_cam_down.start(100)))
        btn_cam_down.released.connect(timer_cam_down.stop)

        main_layout.addWidget(btn_cam_left); main_layout.addWidget(btn_cam_right)
        main_layout.addWidget(btn_cam_up);   main_layout.addWidget(btn_cam_down)

        btn_reset = QPushButton("Reset Camera")
        btn_reset.clicked.connect(reset_camera)
        main_layout.addWidget(btn_reset)

        self.setLayout(main_layout)

    def print_camera_state(self):
        cam = self.plotter.camera
        pos = cam.position; focal = cam.focal_point
        az = cam.azimuth; el = cam.elevation
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
