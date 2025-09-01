import pyvista as pv
import numpy as np
import vtk
import os
import sys
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QCheckBox, QComboBox, QScrollArea, QSizePolicy
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

# Separar malhas para renderização e para projeção dos eletrodos
render_meshes = []          # tudo que não for "Linha" (continua sendo exibido e com checkbox)
render_filenames = []
torso_proj_meshes = []      # APENAS arquivos com "torso" no nome (para snap dos eletrodos)
torso_proj_filenames = []
linha_raw_points = []       # lista de dicts com info para gerar plano depois
linha_filenames = []


for file_path in vtp_files:
    name = os.path.basename(file_path)
    mesh = pv.read(file_path)

    # "Linha" -> só para gerar planos auxiliares
    if 'linha' in name.lower():
        if mesh.n_points >= 3:
            pts = np.asarray(mesh.points)
            linha_raw_points.append({'name': name, 'points': pts})
            linha_filenames.append(name)
        else:
            print(f"[AVISO] '{name}' tem menos de 3 pontos - não é possível ajustar um plano.")
        continue

    # Render: todo VTP que não é "Linha" continua visível/controlável
    if 'Normals' not in mesh.point_data:
        mesh.compute_normals(inplace=True)
    render_meshes.append(mesh)
    render_filenames.append(name)

    # Projeção: apenas nomes contendo "torso"
    if 'torso' in name.lower():
        torso_proj_meshes.append(mesh)
        torso_proj_filenames.append(name)

if not torso_proj_meshes:
    raise Exception("Nenhuma malha com 'torso' no nome foi encontrada para posicionar os eletrodos.")


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

# Renderizar malhas (todas que não são 'Linha')
file_actor_map = {}
torso_actors = []
for i, mesh in enumerate(render_meshes):
    color = color_palette[i % len(color_palette)]
    actor = plotter.add_mesh(mesh, color=color, opacity=0.7,
                             label=f"Mesh {i+1}: {render_filenames[i]}")
    torso_actors.append(actor)
    file_actor_map[render_filenames[i]] = actor

plotter.add_legend()

# MultiBlock para PROJEÇÃO dos eletrodos (apenas 'torso*')
mb_proj = pv.MultiBlock(torso_proj_meshes)
mesh_torso_proj = mb_proj.combine()
plotter.add_mesh(mesh_torso_proj, opacity=0)  # invisível; usado só para o snap


# Criar planos auxiliares a partir dos VTPs "Linha"
def best_fit_plane(points: np.ndarray):
    """
    Retorna (centro, normal) do melhor plano por PCA.
    """
    c = points.mean(axis=0)
    A = points - c
    # Autovetores da covariância
    _, _, vt = np.linalg.svd(A, full_matrices=False)
    normal = vt[-1, :]  # menor autovalor → normal do plano
    # normal normalizada
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    return c, normal

bounds = mesh_torso_proj.bounds  # (xmin,xmax,ymin,ymax,zmin,zmax)
size_max = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]) * 1.2
if size_max <= 0:
    size_max = 100.0  # fallback

linha_plane_meshes = []
linha_plane_actors = []
linha_planes_info = []
for i, item in enumerate(linha_raw_points):
    center, normal = best_fit_plane(item['points'])
    plane = pv.Plane(center=center, direction=normal, i_size=size_max, j_size=size_max, i_resolution=1, j_resolution=1)

    # ✅ guarde os dados para o slice depois
    linha_planes_info.append({'name': item['name'], 'center': center, 'normal': normal})

    linha_plane_meshes.append(plane)
    color = color_palette[(len(torso_proj_meshes) + i) % len(color_palette)]
    actor = plotter.add_mesh(plane, color=color, opacity=0.25, label=f"Plano: {item['name']}")
    linha_plane_actors.append(actor)
    file_actor_map[item['name']] = actor
# ---- INTERSEÇÕES TORSO x PLANOS (realce em vermelho) ----
intersection_actors = []
for info in linha_planes_info:
    # Faz o slice do torso combinado por cada plano (definido por origem/normal)
    slc = mesh_torso_proj.slice(origin=info['center'], normal=info['normal'])

    if slc.n_points > 1:
        # Tubo grosso pra destacar a linha (ajuste o raio conforme escala do teu modelo)
        try:
            tube = slc.tube(radius=max(size_max * 0.005, 2.0), n_sides=24)
            actor = plotter.add_mesh(
                tube, color=(1.0, 0.0, 0.0), opacity=1.0,
                label=f"Interseção: {info['name']}", lighting=False
            )
        except Exception:
            # fallback se tube não estiver disponível por algum motivo
            actor = plotter.add_mesh(
                slc, color=(1.0, 0.0, 0.0), opacity=1.0,
                line_width=6, render_lines_as_tubes=True,
                label=f"Interseção: {info['name']}", lighting=False
            )
        intersection_actors.append(actor)
    else:
        print(f"[AVISO] Sem interseção detectada para '{info['name']}' (slice vazio).")



plotter.add_legend(size=(0, 0))  # reposicionar legenda se necessário


# -----------------------------s
#  ESFERA DE PREVIEW
# -----------------------------
sphere_radius = 6
preview_sphere = pv.Sphere(radius=sphere_radius)
preview_actor = plotter.add_mesh(preview_sphere, color='blue', opacity=1)
initial_position = np.mean(mesh_torso_proj.points, axis=0)
preview_actor.SetPosition(initial_position)

line_actor = None
is_preview_active = True
current_preview_position = np.array(initial_position)

# =========================================
#  ESTRUTURAS GLOBAIS P/ ELETRODOS
# =========================================
electrodes = []
electrode_actors = {}
is_space_pressed = False

# Novo: modo de snap
front_snap_enabled = True  # True = fixa no Y mínimo (frente); False = gruda na superfície (laterais possíveis)

def snap_point_to_torso(target_pos):
    """
    Retorna um ponto na superfície do torso conforme o modo:
      - front_snap_enabled=True: usa o menor Y do vizinho (frente)
      - front_snap_enabled=False: usa o ponto mais próximo na superfície (permite laterais)
    """
    target_pos = np.asarray(target_pos, dtype=float)

    if front_snap_enabled:
        # comportamento atual (frente): menor Y para o XZ selecionado
        candidate_point = find_lowest_y_point(target_pos)
        snapped = find_lowest_y_for_xz(candidate_point, tol=1.0)
        return snapped
    else:
        # lateral/superfície: ponto mais próximo na superfície
        pid = mesh_torso_proj.find_closest_point(target_pos)
        return mesh_torso_proj.points[pid]

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
    closest_point_id = mesh_torso_proj.find_closest_point(mouse_position)
    closest_point = mesh_torso_proj.points[closest_point_id]
    x, z = closest_point[0], closest_point[2]
    if tol is None:
        bounds = mesh_torso_proj.bounds
        tol = (bounds[1] - bounds[0]) * 0.01
    mask = np.linalg.norm(mesh_torso_proj.points[:, [0,2]] - np.array([x, z]), axis=1) < tol
    candidates = mesh_torso_proj.points[mask]
    print(f"Encontrados {len(candidates)} pontos candidatos com tol={tol}")

    if candidates.size > 0:
        lowest_y_point = candidates[np.argmin(candidates[:, 1])]
        return lowest_y_point
    else:
        return closest_point

def find_lowest_y_for_xz(candidate_point, tol=1.0):
    x_candidate, z_candidate = candidate_point[0], candidate_point[2]
    mask = (np.abs(mesh_torso_proj.points[:, 0] - x_candidate) < tol) & \
           (np.abs(mesh_torso_proj.points[:, 2] - z_candidate) < tol)
    candidates = mesh_torso_proj.points[mask]
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
            snapped = snap_point_to_torso(mouse_pos)
            preview_actor.SetPosition(snapped)
            current_preview_position = snapped
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

    # Snap sempre à superfície/“frente” conforme o modo
    snapped = snap_point_to_torso(new_position)
    preview_actor.SetPosition(snapped)
    current_preview_position = snapped
    plotter.render()

def move_preview2(dx=0, dz=0):
    global current_preview_position, preview_actor
    delta = 1.0
    new_position = np.copy(current_preview_position)
    new_position[0] += dx * delta
    new_position[2] += dz * delta
    snapped = snap_point_to_torso(new_position)
    preview_actor.SetPosition(snapped)
    current_preview_position = snapped
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
        
# -------------------------------------------------
#  IMPORTAR COORDENADAS SALVAS
# -------------------------------------------------
def import_files():
    """
    Abre um .txt salvo pelo programa, apaga quaisquer
    eletrodos já existentes e recria todos os pontos
    (esferas + rótulos) a partir do arquivo.
    """
    global electrodes, electrode_actors

    txt_file_path, _ = QFileDialog.getOpenFileName(
        None,
        "Importar coordenadas dos eletrodos",
        "",
        "TXT files (*.txt);;All files (*)"
    )
    if not txt_file_path:
        return  # usuário cancelou

    # 1) Remove tudo que já existe
    for sphere_act, text_act in electrode_actors.values():
        plotter.remove_actor(sphere_act)
        plotter.remove_actor(text_act)
    electrodes.clear()
    electrode_actors.clear()

    # 2) Lê cada linha do arquivo e recria o ponto
    with open(txt_file_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                # Ex.: "Eletrodo #3 (V2): X=10.00, Y=20.00, Z=30.00"
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
        self.setGeometry(1100, 100, 300, 550)  # Define posição e tamanho inicial fixo

        # Layout principal da janela
        main_layout = QVBoxLayout(self)

        # Scroll Area para conter os widgets
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  # Permite redimensionamento automático

        # Widget central dentro do scroll
        scroll_widget = QWidget()
        scroll_area.setWidget(scroll_widget)

        # Layout dentro do widget do scroll
        layout = QVBoxLayout(scroll_widget)

        # ComboBox com rótulos
        self.combo_label = QComboBox()
        electrode_labels = ["COL","V1","V2","V3","V4","V5","V6","LA","RA","LL","RL"]
        self.combo_label.addItems(electrode_labels)
        layout.addWidget(self.combo_label)

        # -------------------------
        # Botões de movimentação
        # -------------------------
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
        layout.addLayout(move_layout)

        # -------------------------
        # Botões principais
        # -------------------------
        button_add = QPushButton("Adicionar Eletrodo")
        button_remove = QPushButton("Remover Último Eletrodo")
        button_save = QPushButton("Salvar")
        button_close = QPushButton("Fechar")
        button_import = QPushButton("Importar")

        button_add.clicked.connect(add_electrode)
        button_remove.clicked.connect(remove_last_electrode)
        button_save.clicked.connect(save_files)
        button_close.clicked.connect(lambda: sys.exit(0))
        button_import.clicked.connect(import_files)

        layout.addWidget(button_add)
        layout.addWidget(button_remove)
        layout.addWidget(button_save)
        layout.addWidget(button_import)
        layout.addWidget(button_close)

        # -------------------------------------------------
        # Toggle: Fixar na frente (Y mínimo) ON/OFF
        # -------------------------------------------------
        self.btn_toggle_front = QPushButton()

        def refresh_toggle_text():
            self.btn_toggle_front.setText(
                "Fixar na frente (Y mínimo): ON" if front_snap_enabled
                else "Fixar na frente (Y mínimo): OFF"
            )

        refresh_toggle_text()

        def on_toggle_front():
            # usa globals já existentes no script
            global front_snap_enabled, current_preview_position
            front_snap_enabled = not front_snap_enabled
            refresh_toggle_text()
            # Re-snap imediato do preview para aderir ao novo modo
            snapped = snap_point_to_torso(preview_actor.GetPosition())
            preview_actor.SetPosition(snapped)
            current_preview_position = snapped
            plotter.render()

        self.btn_toggle_front.clicked.connect(on_toggle_front)
        layout.addWidget(self.btn_toggle_front)

        # -------------------------
        # Checkboxes dos arquivos
        # -------------------------
        self.checkboxes = []
        for i, filename in enumerate(vtp_files):
            checkbox = QCheckBox(f"{os.path.basename(filename)}")
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(lambda state, idx=i: self.toggle_mesh_visibility(idx, state))
            self.checkboxes.append(checkbox)
            layout.addWidget(checkbox)

        # Adiciona o scroll area ao layout principal da janela
        main_layout.addWidget(scroll_area)

        # Impede redimensionamento além do tamanho inicial
        self.setFixedSize(300, 550)

    def toggle_mesh_visibility(self, index, state):
        fname = os.path.basename(vtp_files[index])
        actor = file_actor_map.get(fname)
        if actor is None:
            # Isso é normal para arquivos "Linha" (sem actor) ou itens ignorados.
            print(f"[AVISO] Sem actor associado para '{fname}' (possivelmente 'Linha' ou não renderizado).")
            return

        visible = (state == 2)

        # Agora só controla os planos/malhas, não mexe nas interseções
        try:
            actor.SetVisibility(visible)
            actor.SetPickable(visible)
        except Exception:
            actor.GetProperty().SetOpacity(0.7 if visible else 0.0)

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


