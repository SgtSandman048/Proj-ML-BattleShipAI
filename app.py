import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import tensorflow as tf
from tensorflow import keras

st.set_page_config(page_title="Battleship AI", layout="wide")

st.markdown("""
    <style>
    div[data-testid="stButton"] button { width:100%; height:3em; padding:0; margin:0; }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
with st.sidebar:
    st.markdown("## 🚢 Battleship AI")
    st.markdown("---")
    page = st.radio(
        "เลือกหน้า",
        ["🎮 เล่นเกม (ML)", "🤖 เล่นเกม (DQN)", "📊 เกี่ยวกับโมเดล ML", "🧠 เกี่ยวกับโมเดล Neural Network"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("ML: Hunt+Target + Bayesian + Q-Learning")
    st.caption("NN: Deep Q-Network (Dueling DQN)")

# ============================================================
# SHARED CONSTANTS
# ============================================================
FLEET = {
    "Carrier (5 ช่อง)": 5,
    "Battleship (4 ช่อง)": 4,
    "Cruiser (3 ช่อง)": 3,
    "Submarine (3 ช่อง)": 3,
    "Destroyer (2 ช่อง)": 2
}
SHIPS_NEEDED   = sum(FLEET.values())
SHIP_LENGTHS   = list(FLEET.values())
ROWS_LABEL     = "ABCDEFGHIJ"

# ============================================================
# SHARED: Load base probability map
# ============================================================
@st.cache_data
def load_base_prob():
    base_prob = np.full(100, 1.0 / 100)
    try:
        df = pd.read_csv("battleship_game_squares.csv")
        player = df[df["ai_ships"] == 0]
        stats  = player.groupby("square")["games"].sum().reset_index()
        total  = stats["games"].sum()
        for _, row in stats.iterrows():
            idx = int(row["square"]) - 1
            if 0 <= idx < 100:
                base_prob[idx] = row["games"] / total
    except Exception:
        pass
    return base_prob

BASE_PROB = load_base_prob()

# ============================================================
# ML AI CLASS — Hunt + Target + Bayesian
# ============================================================
class HuntTargetAI:
    MISS_NEIGHBOR = 0.55

    def __init__(self, base_prob_array):
        self.base_prob = base_prob_array.copy()
        self.reset()

    def reset(self):
        self.guessed     = set()
        self.active_hits = []
        self.mode        = "HUNT"
        self.live_prob   = self.base_prob.copy()

    def _neighbors(self, idx):
        r, c = idx // 10, idx % 10
        nb = []
        if r > 0: nb.append((r - 1) * 10 + c)
        if r < 9: nb.append((r + 1) * 10 + c)
        if c > 0: nb.append(r * 10 + (c - 1))
        if c < 9: nb.append(r * 10 + (c + 1))
        return nb

    def _aligned_targets(self):
        if len(self.active_hits) < 2:
            return []
        rows = [h // 10 for h in self.active_hits]
        cols = [h % 10  for h in self.active_hits]
        targets = []
        if len(set(rows)) == 1:
            r = rows[0]; mc, xc = min(cols), max(cols)
            if mc > 0: targets.append(r * 10 + mc - 1)
            if xc < 9: targets.append(r * 10 + xc + 1)
        elif len(set(cols)) == 1:
            c = cols[0]; mr, xr = min(rows), max(rows)
            if mr > 0: targets.append((mr - 1) * 10 + c)
            if xr < 9: targets.append((xr + 1) * 10 + c)
        return targets

    def _bayesian_miss_update(self, idx):
        for nb in self._neighbors(idx):
            if nb not in self.guessed:
                self.live_prob[nb] *= self.MISS_NEIGHBOR
        remaining = [i for i in range(100) if i not in self.guessed]
        total = sum(self.live_prob[i] for i in remaining)
        if total > 0:
            factor = 1.0 / total
            for i in remaining:
                self.live_prob[i] *= factor

    def choose_target(self):
        if self.mode == "TARGET" and self.active_hits:
            for sq in self._aligned_targets():
                if sq not in self.guessed:
                    return sq
            for h in self.active_hits:
                for nb in self._neighbors(h):
                    if nb not in self.guessed:
                        return nb
            self.mode = "HUNT"
            self.active_hits = []
        prob = self.live_prob.copy()
        for sq in self.guessed:
            prob[sq] = -1
        return int(np.argmax(prob))

    def register_result(self, idx, is_hit, ship_sunk=False):
        self.guessed.add(idx)
        if is_hit:
            self.active_hits.append(idx)
            self.mode = "TARGET"
            if ship_sunk:
                self.active_hits = []
                self.mode = "HUNT"
        else:
            self.live_prob[idx] = 0
            self._bayesian_miss_update(idx)

# ============================================================
# DQN AI CLASS
# ============================================================
@keras.utils.register_keras_serializable(package="battleship")
class AdvantageMean(keras.Layer):
    def call(self, adv):
        return tf.reduce_mean(adv, axis=1, keepdims=True)
    def get_config(self):
        return super().get_config()


@st.cache_resource
def load_dqn_model():
    try:
        model   = keras.models.load_model("battleship_dqn_model.keras")
        bp_norm = np.load("base_prob_norm.npy")
        return model, bp_norm
    except Exception as e:
        return None, None


class DQNBattleshipAI:
    def __init__(self, model, base_prob_norm):
        self.model          = model
        self.base_prob_norm = base_prob_norm
        self._state         = np.empty(300, dtype=np.float32)
        self._state[100:200] = base_prob_norm
        self.reset()

    def reset(self):
        self._state[:100] = 0.0
        self._state[200:] = 1.0

    def register_result(self, idx, is_hit, ship_sunk=False):
        self._state[idx]       = 1.0 if is_hit else -1.0
        self._state[200 + idx] = 0.0

    def choose_target(self):
        q      = self.model(self._state[np.newaxis], training=False).numpy()[0]
        avail  = self._state[200:].astype(bool)
        q[~avail] = -np.inf
        return int(np.argmax(q))

# ============================================================
# SHARED: Place ships helper
# ============================================================
def random_place_ships(board_array, coords_dict):
    board_array.fill(0)
    coords_dict.clear()
    for ship_name, length in FLEET.items():
        placed = False
        while not placed:
            r, c = random.randint(0, 9), random.randint(0, 9)
            ori  = random.choice(["H", "V"])
            if ori == "H" and c + length <= 10:
                cells = [(r, c + j) for j in range(length)]
            elif ori == "V" and r + length <= 10:
                cells = [(r + j, c) for j in range(length)]
            else:
                continue
            if all(board_array[rr, cc] == 0 for rr, cc in cells):
                for rr, cc in cells:
                    board_array[rr, cc] = 1
                coords_dict[ship_name] = cells
                placed = True

def random_board_flat():
    board, ship_cells = np.zeros(100, dtype=int), []
    for si, length in enumerate(SHIP_LENGTHS):
        placed = False
        while not placed:
            horiz = random.random() < 0.5
            if horiz:
                r = random.randint(0, 9); c = random.randint(0, 9 - length + 1)
                cells = [r * 10 + c + j for j in range(length)]
            else:
                r = random.randint(0, 9 - length + 1); c = random.randint(0, 9)
                cells = [(r + j) * 10 + c for j in range(length)]
            if all(board[cell] == 0 for cell in cells):
                for cell in cells:
                    board[cell] = si + 1
                ship_cells.append(cells)
                placed = True
    return board, ship_cells

# ============================================================
# PAGE 1: เล่นเกม (ML — Hunt+Target+Bayesian)
# ============================================================
if page == "🎮 เล่นเกม (ML)":
    st.title("🚢 Battleship: Human vs ML AI")
    st.caption("AI ใช้ Hunt+Target + Bayesian Probability Update")

    def ml_init_game():
        b = np.zeros((10, 10), dtype=int)
        c = {}
        random_place_ships(b, c)
        st.session_state.ml_player_board  = b
        st.session_state.ml_player_ships  = c
        ai_b = np.zeros((10, 10), dtype=int)
        ai_c = {}
        random_place_ships(ai_b, ai_c)
        st.session_state.ml_ai_board      = ai_b
        st.session_state.ml_ai_ships      = ai_c
        st.session_state.ml_player_hits   = set()
        st.session_state.ml_player_misses = set()
        st.session_state.ml_ai_hits       = set()
        st.session_state.ml_ai_misses     = set()
        st.session_state.ml_player_sunk       = set()   # ship names
        st.session_state.ml_player_sunk_cells = set()   # (r,c) ของเรือที่จมแล้ว
        st.session_state.ml_ai_sunk           = set()   # ship names
        st.session_state.ml_ai_sunk_cells     = set()   # (r,c) ของเรือที่จมแล้ว
        st.session_state.ml_game_over     = False
        st.session_state.ml_winner        = None
        st.session_state.ml_ai            = HuntTargetAI(BASE_PROB)
        st.session_state.ml_ai_log        = []
        st.session_state.ml_moves         = 0

    if "ml_game_over" not in st.session_state:
        ml_init_game()

    col_h, col_btn = st.columns([5, 1])
    with col_btn:
        if st.button("🔄 เกมใหม่", use_container_width=True):
            ml_init_game()
            st.rerun()

    if st.session_state.ml_game_over:
        if st.session_state.ml_winner == "player":
            st.success(f"🎉 คุณชนะ! ใช้ {st.session_state.ml_moves} การยิง")
        else:
            st.error(f"🤖 ML AI ชนะ! ใช้ {len(st.session_state.ml_ai_hits)} hits")

    def ml_shoot(r, c):
        sq = r * 10 + c
        if st.session_state.ml_game_over:
            return
        if (r, c) in st.session_state.ml_player_hits | st.session_state.ml_player_misses:
            return
        st.session_state.ml_moves += 1
        if st.session_state.ml_ai_board[r, c] == 1:
            st.session_state.ml_player_hits.add((r, c))
            for name, cells in st.session_state.ml_ai_ships.items():
                if all(cell in st.session_state.ml_player_hits for cell in cells) and name not in st.session_state.ml_player_sunk:
                    st.session_state.ml_player_sunk.add(name)
                    st.session_state.ml_player_sunk_cells.update(cells)  # mark destroyed
        else:
            st.session_state.ml_player_misses.add((r, c))

        if len(st.session_state.ml_player_hits) >= SHIPS_NEEDED:
            st.session_state.ml_game_over = True
            st.session_state.ml_winner    = "player"
            return

        # AI turn
        ai    = st.session_state.ml_ai
        ai_sq = ai.choose_target()
        ar, ac = ai_sq // 10, ai_sq % 10
        ai_is_hit  = st.session_state.ml_player_board[ar, ac] == 1
        ship_sunk  = False
        if ai_is_hit:
            st.session_state.ml_ai_hits.add((ar, ac))
            for name, cells in st.session_state.ml_player_ships.items():
                if all(cell in st.session_state.ml_ai_hits for cell in cells) and name not in st.session_state.ml_ai_sunk:
                    st.session_state.ml_ai_sunk.add(name)
                    st.session_state.ml_ai_sunk_cells.update(cells)  # mark destroyed
                    ship_sunk = True
        else:
            st.session_state.ml_ai_misses.add((ar, ac))
        ai.register_result(ai_sq, ai_is_hit, ship_sunk)
        result = "💥 HIT" if ai_is_hit else "💧 MISS"
        st.session_state.ml_ai_log.insert(0, f"Turn {st.session_state.ml_moves}: ML AI → {ROWS_LABEL[ar]}{ac} → {result}")

        if len(st.session_state.ml_ai_hits) >= SHIPS_NEEDED:
            st.session_state.ml_game_over = True
            st.session_state.ml_winner    = "ai"

    left, right = st.columns(2)
    with left:
        st.subheader("🎯 ยิงใส่กระดาน AI")
        hdr = st.columns(11)
        hdr[0].write("")
        for j in range(10): hdr[j+1].markdown(f"<center><b>{j}</b></center>", unsafe_allow_html=True)
        for r in range(10):
            cols = st.columns(11)
            cols[0].markdown(f"**{ROWS_LABEL[r]}**")
            for c in range(10):
                if (r, c) in st.session_state.ml_player_sunk_cells:
                    cols[c+1].button("💀", key=f"ml_a_{r}_{c}", disabled=True)
                elif (r, c) in st.session_state.ml_player_hits:
                    cols[c+1].button("💥", key=f"ml_a_{r}_{c}", disabled=True)
                elif (r, c) in st.session_state.ml_player_misses:
                    cols[c+1].button("💧", key=f"ml_a_{r}_{c}", disabled=True)
                else:
                    if cols[c+1].button("🌊", key=f"ml_a_{r}_{c}", disabled=st.session_state.ml_game_over):
                        ml_shoot(r, c)
                        st.rerun()
        st.caption(f"Hits: {len(st.session_state.ml_player_hits)}/{SHIPS_NEEDED} | Moves: {st.session_state.ml_moves}")

    with right:
        st.subheader("🛡️ กระดานของคุณ")
        hdr2 = st.columns(11)
        hdr2[0].write("")
        for j in range(10): hdr2[j+1].markdown(f"<center><b>{j}</b></center>", unsafe_allow_html=True)
        for r in range(10):
            cols = st.columns(11)
            cols[0].markdown(f"**{ROWS_LABEL[r]}**")
            for c in range(10):
                has_ship = st.session_state.ml_player_board[r, c] == 1
                if (r, c) in st.session_state.ml_ai_sunk_cells:
                    cols[c+1].button("💀", key=f"ml_p_{r}_{c}", disabled=True)
                elif (r, c) in st.session_state.ml_ai_hits:
                    cols[c+1].button("💥", key=f"ml_p_{r}_{c}", disabled=True)
                elif (r, c) in st.session_state.ml_ai_misses:
                    cols[c+1].button("💧", key=f"ml_p_{r}_{c}", disabled=True)
                else:
                    cols[c+1].button("🚢" if has_ship else "🌊", key=f"ml_p_{r}_{c}", disabled=True)
        st.caption(f"AI Hits: {len(st.session_state.ml_ai_hits)}/{SHIPS_NEEDED} | Mode: {st.session_state.ml_ai.mode}")
        if st.session_state.ml_ai_log:
            with st.expander("📋 ML AI Shot History"):
                for entry in st.session_state.ml_ai_log[:8]:
                    st.text(entry)

# ============================================================
# PAGE 2: เล่นเกม (DQN Neural Network)
# ============================================================
elif page == "🤖 เล่นเกม (DQN)":
    st.title("🧠 Battleship: Human vs DQN Neural Network")
    st.caption("AI ใช้ Dueling DQN + Double DQN เรียนรู้จาก 3,000 เกม Self-play")

    dqn_model, bp_norm = load_dqn_model()
    if dqn_model is None:
        st.error("❌ ไม่พบ `battleship_dqn_model.keras` — กรุณา train และ save model ก่อน")
        st.info("รัน notebook จนถึง Cell 23 (Save Model) แล้วลอง refresh หน้านี้")
        st.stop()

    def dqn_init_game():
        board_flat, ship_cells = random_board_flat()
        st.session_state.dqn_player_board = board_flat
        st.session_state.dqn_player_ships = ship_cells
        ai_flat, ai_cells = random_board_flat()
        st.session_state.dqn_ai_board     = ai_flat
        st.session_state.dqn_ai_ships     = ai_cells
        st.session_state.dqn_p_guesses       = set()
        st.session_state.dqn_p_hits          = 0
        st.session_state.dqn_p_moves         = 0
        st.session_state.dqn_p_sunk_cells    = set()   # sq indices ของเรือที่ผู้เล่นจม
        st.session_state.dqn_ai_guesses      = set()
        st.session_state.dqn_ai_hits_n       = 0
        st.session_state.dqn_ai_sunk_cells   = set()   # sq indices ของเรือที่ AI จม
        st.session_state.dqn_ai_moves     = 0
        st.session_state.dqn_game_over    = False
        st.session_state.dqn_winner       = None
        st.session_state.dqn_last_ai      = "-"
        st.session_state.dqn_last_res     = ""
        st.session_state.dqn_log          = []
        st.session_state.dqn_ai           = DQNBattleshipAI(dqn_model, bp_norm)

    if "dqn_game_over" not in st.session_state:
        dqn_init_game()

    col_h, col_btn = st.columns([5, 1])
    with col_btn:
        if st.button("🔄 เกมใหม่", key="dqn_new", use_container_width=True):
            dqn_init_game()
            st.rerun()

    if st.session_state.dqn_game_over:
        if st.session_state.dqn_winner == "player":
            st.success(f"🎉 คุณชนะ! ใช้ {st.session_state.dqn_p_moves} การยิง")
        else:
            st.error(f"🤖 DQN AI ชนะ! ใช้ {st.session_state.dqn_ai_moves} การยิง")

    def dqn_shoot(sq):
        if st.session_state.dqn_game_over or sq in st.session_state.dqn_p_guesses:
            return
        st.session_state.dqn_p_guesses.add(sq)
        st.session_state.dqn_p_moves += 1
        is_hit = st.session_state.dqn_ai_board[sq] > 0
        if is_hit:
            st.session_state.dqn_p_hits += 1
            ship_idx = st.session_state.dqn_ai_board[sq] - 1
            ship_cells_list = st.session_state.dqn_ai_ships[ship_idx]
            if all(c in st.session_state.dqn_p_guesses for c in ship_cells_list):
                st.session_state.dqn_p_sunk_cells.update(ship_cells_list)
        if st.session_state.dqn_p_hits >= SHIPS_NEEDED:
            st.session_state.dqn_game_over = True
            st.session_state.dqn_winner    = "player"
            return
        ai    = st.session_state.dqn_ai
        ai_sq = ai.choose_target()
        st.session_state.dqn_ai_moves += 1
        ai_hit = st.session_state.dqn_player_board[ai_sq] > 0
        if ai_hit:
            st.session_state.dqn_ai_hits_n += 1
            ship_idx = st.session_state.dqn_player_board[ai_sq] - 1
            ship_cells_list = st.session_state.dqn_player_ships[ship_idx]
            if all(c in st.session_state.dqn_ai_guesses | {ai_sq} for c in ship_cells_list):
                st.session_state.dqn_ai_sunk_cells.update(ship_cells_list)
        ai.register_result(ai_sq, ai_hit)
        st.session_state.dqn_ai_guesses.add(ai_sq)
        ar, ac = ai_sq // 10, ai_sq % 10
        res = "💥 HIT" if ai_hit else "💧 MISS"
        st.session_state.dqn_last_ai  = f"{ROWS_LABEL[ar]}{ac}"
        st.session_state.dqn_last_res = res
        st.session_state.dqn_log.insert(0, f"Turn {st.session_state.dqn_ai_moves}: DQN → {ROWS_LABEL[ar]}{ac} → {res}")
        if st.session_state.dqn_ai_hits_n >= SHIPS_NEEDED:
            st.session_state.dqn_game_over = True
            st.session_state.dqn_winner    = "ai"

    left, right = st.columns(2)
    with left:
        st.subheader("🎯 ยิงใส่กระดาน DQN AI")
        hdr = st.columns(11)
        hdr[0].write("")
        for j in range(10): hdr[j+1].markdown(f"<center><b>{j}</b></center>", unsafe_allow_html=True)
        for r in range(10):
            cols = st.columns(11)
            cols[0].markdown(f"**{ROWS_LABEL[r]}**")
            for c in range(10):
                sq = r * 10 + c
                if sq in st.session_state.dqn_p_sunk_cells:
                    cols[c+1].button("💀", key=f"dqn_a_{sq}", disabled=True)
                elif sq in st.session_state.dqn_p_guesses:
                    label = "💥" if st.session_state.dqn_ai_board[sq] > 0 else "💧"
                    cols[c+1].button(label, key=f"dqn_a_{sq}", disabled=True)
                else:
                    if cols[c+1].button("🌊", key=f"dqn_a_{sq}", disabled=st.session_state.dqn_game_over):
                        dqn_shoot(sq)
                        st.rerun()
        st.caption(f"Hits: {st.session_state.dqn_p_hits}/{SHIPS_NEEDED} | Moves: {st.session_state.dqn_p_moves}")

    with right:
        st.subheader("🛡️ กระดานของคุณ")
        st.caption(f"DQN ยิงล่าสุด: **{st.session_state.dqn_last_ai}** {st.session_state.dqn_last_res}")
        hdr2 = st.columns(11)
        hdr2[0].write("")
        for j in range(10): hdr2[j+1].markdown(f"<center><b>{j}</b></center>", unsafe_allow_html=True)
        for r in range(10):
            cols = st.columns(11)
            cols[0].markdown(f"**{ROWS_LABEL[r]}**")
            for c in range(10):
                sq = r * 10 + c
                has_ship = st.session_state.dqn_player_board[sq] > 0
                if sq in st.session_state.dqn_ai_sunk_cells:
                    cols[c+1].button("💀", key=f"dqn_p_{sq}", disabled=True)
                elif sq in st.session_state.dqn_ai_guesses:
                    cols[c+1].button("💥" if has_ship else "💧", key=f"dqn_p_{sq}", disabled=True)
                else:
                    cols[c+1].button("🚢" if has_ship else "🌊", key=f"dqn_p_{sq}", disabled=True)
        st.caption(f"AI Hits: {st.session_state.dqn_ai_hits_n}/{SHIPS_NEEDED} | AI Moves: {st.session_state.dqn_ai_moves}")
        if st.session_state.dqn_log:
            with st.expander("📋 DQN AI Shot History"):
                for entry in st.session_state.dqn_log[:8]:
                    st.text(entry)

# ============================================================
# PAGE 3: เกี่ยวกับโมเดล ML
# ============================================================
elif page == "📊 เกี่ยวกับโมเดล ML":
    st.title("📊 Battleship AI — Machine Learning Model")
    st.markdown("อธิบายแนวทางการพัฒนา AI ตั้งแต่ต้นจนจบ รวมถึงทฤษฎีและแหล่งอ้างอิง")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 การเตรียมข้อมูล",
        "🧠 ทฤษฎีอัลกอริทึม",
        "🔧 ขั้นตอนพัฒนาโมเดล",
        "📚 แหล่งอ้างอิง"
    ])

    with tab1:
        st.header("📁 การเตรียมข้อมูล (Data Preparation)")
        st.subheader("แหล่งที่มาของข้อมูล")
        st.markdown("""
        ข้อมูลที่ใช้ในโปรเจกต์นี้มาจาก **GitHub — Battleship Data**
        ซึ่งบันทึกผลจากเกม Battleship หลายหมื่นเกมที่มีผู้เล่นเป็นมนุษย์และ AI ในโหมดต่างๆ
        """)
        col_a, col_b, col_c = st.columns(3)
        with col_a: st.metric("battleship_games.csv", "59,710 เกม", "ข้อมูลรายเกม")
        with col_b: st.metric("battleship_game_squares.csv", "2,400 แถว", "สถิติรายช่อง")
        with col_c: st.metric("battleship_game_moves.csv", "1,008 แถว", "สถิติจำนวน moves")
        st.divider()
        st.subheader("โครงสร้างไฟล์ CSV")
        with st.expander("📄 battleship_games.csv — ข้อมูลรายเกม", expanded=True):
            st.markdown("""
            | คอลัมน์ | ประเภท | คำอธิบาย |
            |---------|--------|----------|
            | `id` | int | รหัสเกม |
            | `timestampUTC` | int | เวลา Unix ที่เล่น |
            | `ai_win` | int (0/1) | AI ชนะหรือไม่ |
            | `moves` | int | จำนวน moves ทั้งหมดในเกม |
            | `autoplay` | int (0/1) | เล่นอัตโนมัติหรือไม่ |
            | `ai_mode_id` | int (1/2/3) | โหมด AI ที่ใช้ |
            """)
        with st.expander("📄 battleship_game_squares.csv — สถิติความถี่รายช่อง"):
            st.markdown("""
            | คอลัมน์ | ประเภท | คำอธิบาย |
            |---------|--------|----------|
            | `square` | int (1–100) | หมายเลขช่องบนกระดาน 10×10 |
            | `ai_ships` | int (0/1) | 0=เรือผู้เล่น, 1=เรือ AI |
            | `games` | int | จำนวนเกมที่มีเรืออยู่ในช่องนั้น |

            ช่องที่ `games` สูง → ผู้เล่นมักวางเรือที่นั่น → AI ควรเล็งยิงก่อน
            """)
        with st.expander("📄 battleship_game_moves.csv — สถิติจำนวน moves"):
            st.markdown("""
            | คอลัมน์ | ประเภท | คำอธิบาย |
            |---------|--------|----------|
            | `moves` | int | จำนวน moves ทั้งหมดในเกม |
            | `games` | int | จำนวนเกมที่ใช้ moves เท่านั้น |
            | `ai_win` | int | กรองตามผลเกม |

            ใช้วิเคราะห์ว่า AI โหมดไหนชนะด้วย moves น้อยที่สุด
            """)
        st.divider()
        st.subheader("ขั้นตอนการเตรียมข้อมูล (Preprocessing Pipeline)")
        for title, desc in [
            ("1️⃣ โหลดข้อมูล", "อ่าน CSV ทั้ง 3 ไฟล์ด้วย pandas และตรวจสอบ shape, dtype, ค่า null"),
            ("2️⃣ กรองข้อมูล", "เลือกเฉพาะแถวที่ `ai_ships == 0` เพื่อดูพฤติกรรมการวางเรือของผู้เล่นมนุษย์"),
            ("3️⃣ Aggregate", "รวม `games` ตาม `square` ด้วย groupby เพื่อหาความถี่ของแต่ละช่อง"),
            ("4️⃣ Normalize", "แปลงเป็น probability (%) โดยหารด้วยผลรวมทั้งหมด"),
            ("5️⃣ สร้าง Board Array", "แปลง probability เป็น numpy array รูปร่าง (100,) → reshape เป็น (10,10)"),
            ("6️⃣ Validate", "ตรวจว่า probability รวมได้ ~100% และไม่มีค่า NaN หรือ Inf"),
        ]:
            with st.container(border=True):
                st.markdown(f"**{title}** — {desc}")
        st.divider()
        st.subheader("สรุปข้อมูล EDA ที่สำคัญ")
        st.markdown("""
        - ผู้เล่นนิยมวางเรือ **บริเวณขอบกระดาน** มากกว่ากลาง (edge avg ≈ **1.095%** vs center avg ≈ **0.946%**)
        - AI โหมด 3 มี win rate สูงกว่าโหมด 1 และ 2
        - เกมที่ AI ชนะใช้ moves เฉลี่ย **น้อยกว่า** เกมที่แพ้อย่างมีนัยสำคัญ
        - ช่องที่มี probability สูงสุด 10 อันดับแรก **ล้วนอยู่บริเวณขอบ** ทั้งสิ้น
        """)

    with tab2:
        st.header("🧠 ทฤษฎีของอัลกอริทึม ML ที่พัฒนา")
        st.markdown("พัฒนา AI ทั้งหมด **3 วิธี** โดยแต่ละวิธีต่อยอดจากวิธีก่อนหน้า")

        with st.expander("⭐⭐⭐  วิธีที่ 1 — Hunt + Target Mode", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("""
                #### แนวคิด
                แบ่ง AI ออกเป็น 2 โหมดที่สลับกันตามสถานการณ์

                **HUNT Mode** — เลือกช่องที่มี probability สูงสุดจาก base map ยิงไปเรื่อยๆ จนกว่าจะโดนเรือ

                **TARGET Mode** — เมื่อ hit เรือ สลับมายิงช่องรอบๆ ทันที ถ้าโดน 2 ช่องแล้ว ไล่ตามแนวนั้น (H หรือ V) เมื่อจมเรือแล้ว กลับ HUNT mode
                """)
            with c2:
                st.markdown("""
                #### Logic
                ```
                if mode == TARGET:
                    if 2+ hits aligned:
                        shoot end of line
                    else:
                        shoot neighbors of hits
                    if no valid target:
                        mode = HUNT
                else:
                    shoot argmax(base_prob)
                ```
                | ตัวชี้วัด | ค่า |
                |----------|-----|
                | Avg moves | ~38 |
                | ลดลงจากเดิม | ~24% |
                | Complexity | O(1) ต่อ turn |
                """)

        with st.expander("⭐⭐⭐⭐  วิธีที่ 2 — Bayesian Probability Update"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("""
                #### แนวคิด
                อัปเดต probability map ทุกครั้งที่ยิงพลาด เพื่อให้ AI ปรับตัวตามพฤติกรรมของผู้เล่นคนนั้น

                **ปัญหาของวิธีที่ 1** — ใช้ static base_prob ตลอดเกม ถ้าผู้เล่นวางเรือกลางกระดาน AI ก็ยังยิงขอบ

                **Bayesian Update** — ทุกครั้งที่ยิงพลาดที่ตำแหน่ง i จะลด `live_prob` ของช่องรอบข้างลง 45% แล้ว Normalize ใหม่
                """)
            with c2:
                st.markdown("""
                #### สูตร Bayesian Update
                ```
                Miss at idx i:
                  for nb in neighbors(i):
                    live_prob[nb] *= 0.55
                  normalize(live_prob)

                posterior ∝ prior × likelihood
                ```
                | ตัวชี้วัด | ค่า |
                |----------|-----|
                | Avg moves | ~33 |
                | ลดลงจากวิธี 1 | ~13% |
                | ปรับตัวตามผู้เล่น | ✅ |
                """)

        with st.expander("⭐⭐⭐⭐⭐  วิธีที่ 3 — Q-Learning (Reinforcement Learning)"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("""
                #### แนวคิด
                ใช้ Reinforcement Learning ให้ AI เรียนรู้จากการเล่น 20,000 เกมจำลอง (self-play)

                **องค์ประกอบหลัก**
                - **State**: ชุดช่องที่ยิงไปแล้ว
                - **Action**: ช่องที่เลือกยิง (0–99)
                - **Reward**: +10 hit, +50 จม, -1 พลาด
                - **Q-Table**: เก็บ expected reward ของแต่ละช่อง

                **Epsilon-Greedy Policy** — เริ่มต้น ε=1.0 (สุ่มทั้งหมด) Decay ε ทุก episode → exploit มากขึ้น
                """)
            with c2:
                st.markdown("""
                #### Q-Learning Formula (Bellman)
                ```
                Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]

                α = 0.1  (learning rate)
                γ = 0.9  (discount factor)
                ε: 1.0 → 0.05 (decay=0.9995)
                ```
                | ตัวชี้วัด | ค่า |
                |----------|-----|
                | Avg moves | ~30 |
                | Training episodes | 20,000 |
                | เรียนรู้จากประสบการณ์ | ✅ |
                """)

    with tab3:
        st.header("🔧 ขั้นตอนการพัฒนาโมเดล ML")
        st.markdown("Pipeline การพัฒนาตั้งแต่ข้อมูลดิบจนถึง Streamlit App")
        for i, (title, detail) in enumerate([
            ("โหลดและ EDA ข้อมูล", "อ่าน CSV 3 ไฟล์ → วิเคราะห์ distribution, win rate, moves stats → สร้าง Heatmap"),
            ("สร้าง Base Probability Map", "กรอง `ai_ships==0` → groupby square → normalize → numpy array (100,)"),
            ("พัฒนา Hunt+Target AI", "Implement HuntTargetAI class → test ด้วย 500 เกม simulate → วัด avg moves"),
            ("เพิ่ม Bayesian Update", "เพิ่ม `_bayesian_miss_update()` → ลด probability รอบช่องที่ miss → normalize"),
            ("Train Q-Learning", "self-play 20,000 episodes → ε-greedy → update Q-table ทุก step → decay ε"),
            ("เปรียบเทียบและเลือก", "Boxplot comparison → Hunt+Target ดีที่สุดใน trade-off ความซับซ้อน/ประสิทธิภาพ"),
            ("Deploy ใน Streamlit", "Implement App ด้วย st.session_state → sidebar navigation → deploy ผ่าน localtunnel"),
        ], 1):
            with st.container(border=True):
                st.markdown(f"**ขั้นที่ {i}: {title}**")
                st.caption(detail)

        st.divider()
        st.subheader("📊 ผลการเปรียบเทียบ ML Models")
        st.markdown("""
        | วิธี | Avg Moves | Median | ข้อดี | ข้อจำกัด |
        |------|-----------|--------|-------|----------|
        | Hunt Only (Static) | ~50 | 49 | เรียบง่าย | ไม่ปรับตามสถานการณ์ |
        | Hunt + Target | ~38 | 37 | เข้าใจง่าย เร็ว | ยังเป็น heuristic |
        | Bayesian Update | ~33 | 32 | ปรับตามผู้เล่น | ช้ากว่าเล็กน้อย |
        | Q-Learning | ~30 | 29 | เรียนรู้ได้ | State space จำกัด |
        """)

    with tab4:
        st.header("📚 แหล่งอ้างอิง ML")
        st.subheader("📖 งานวิจัยและตำราที่อ้างอิง")
        for ref in [
            {"title": "Reinforcement Learning: An Introduction", "authors": "Sutton, R.S. & Barto, A.G. (2018)", "publisher": "MIT Press, 2nd Edition", "note": "ทฤษฎีหลักของ Q-Learning และ Bellman Equation", "url": "http://incompleteideas.net/book/the-book-2nd.html"},
            {"title": "Q-Learning (Watkins, 1989)", "authors": "Watkins, C.J.C.H. & Dayan, P. (1992)", "publisher": "Machine Learning, 8(3-4), 279–292", "note": "งานวิจัยต้นฉบับ Q-Learning algorithm", "url": "https://doi.org/10.1007/BF00992698"},
            {"title": "Bayesian Inference and Learning", "authors": "Bishop, C.M. (2006)", "publisher": "Pattern Recognition and Machine Learning, Springer", "note": "หลักการของ Bayesian update และ posterior probability", "url": "https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/"},
        ]:
            with st.container(border=True):
                st.markdown(f"**{ref['title']}**")
                st.markdown(f"_{ref['authors']}_ — {ref['publisher']}")
                st.caption(f"📌 {ref['note']}")
                st.markdown(f"🔗 {ref['url']}")

        st.subheader("🛠️ Library และ Framework")
        for name, url, desc in [
            ("Streamlit 1.x", "https://streamlit.io", "Web framework สำหรับ Python ML app"),
            ("NumPy", "https://numpy.org", "คำนวณ probability array และ matrix operations"),
            ("Pandas", "https://pandas.pydata.org", "โหลดและ preprocess CSV data"),
            ("Seaborn / Matplotlib", "https://seaborn.pydata.org", "Visualization และ heatmap"),
        ]:
            c1, c2, c3 = st.columns([2, 3, 3])
            with c1: st.markdown(f"**{name}**")
            with c2: st.caption(desc)
            with c3: st.caption(f"🔗 {url}")
            st.divider()

        st.subheader("📝 แนวทางการพัฒนาที่อ้างอิง")
        for title, author, url, note in [
            ("Battleship AI Strategy", "DataGenetics Blog", "http://www.datagenetics.com/blog/december32011/", "วิเคราะห์กลยุทธ์ Battleship อย่างละเอียดด้วยสถิติ"),
            ("Hunt and Target Algorithm", "Nick Berry, 2011", "http://www.datagenetics.com/blog/december32011/", "อธิบาย Hunt/Target mode ที่เป็น basis ของโปรเจกต์นี้"),
            ("Epsilon-Greedy Exploration", "Sutton & Barto Ch.2", "http://incompleteideas.net/book/the-book-2nd.html", "ทฤษฎี exploration vs exploitation trade-off"),
        ]:
            with st.container(border=True):
                st.markdown(f"**{title}** — _{author}_")
                st.caption(f"📌 {note}")
                st.markdown(f"🔗 {url}")

# ============================================================
# PAGE 4: เกี่ยวกับโมเดล Neural Network
# ============================================================
elif page == "🧠 เกี่ยวกับโมเดล Neural Network":
    st.title("🧠 Battleship AI — Neural Network (DQN) Model")
    st.markdown("อธิบายสถาปัตยกรรม Deep Q-Network ทฤษฎี และขั้นตอนการพัฒนาตั้งแต่ต้นจนจบ")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 การเตรียมข้อมูล",
        "🧠 ทฤษฎีอัลกอริทึม",
        "🔧 ขั้นตอนพัฒนาโมเดล",
        "📚 แหล่งอ้างอิง"
    ])

    with tab1:
        st.header("📁 การเตรียมข้อมูล (Data Preparation)")
        st.markdown("DQN ใช้ข้อมูลเดียวกับ ML Model แต่มีการแปลงเพิ่มเติมสำหรับ Neural Network")
        col_a, col_b, col_c = st.columns(3)
        with col_a: st.metric("battleship_games.csv", "59,710 เกม", "ข้อมูลรายเกม")
        with col_b: st.metric("battleship_game_squares.csv", "2,400 แถว", "สถิติรายช่อง")
        with col_c: st.metric("battleship_game_moves.csv", "1,008 แถว", "สถิติจำนวน moves")
        st.divider()
        st.subheader("การแปลงข้อมูลสำหรับ Neural Network Input")
        st.markdown("""
        State vector ขนาด **300 dimensions** ต่อ 1 game state ประกอบด้วย 3 ส่วน:
        """)
        for title, desc, example in [
            ("Board State [0:100]", "สถานะของแต่ละช่องบนกระดาน", "`0` = ยังไม่ยิง, `1` = ยิงโดน, `-1` = ยิงพลาด"),
            ("Probability Map [100:200]", "Historical probability ที่ normalize เป็น [0,1]", "นำมาจาก battleship_game_squares.csv ของผู้เล่นมนุษย์"),
            ("Action Mask [200:300]", "ช่องที่ยังยิงได้", "`1` = available, `0` = fired already"),
        ]:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.markdown(f"{desc}")
                st.caption(f"ตัวอย่าง: {example}")

        st.divider()
        st.subheader("Preprocessing Pipeline สำหรับ DQN")
        for i, (title, detail) in enumerate([
            ("โหลด CSV และสร้าง Base Probability", "groupby square → normalize → BASE_PROB_NORM array (100,) ใน [0,1]"),
            ("Normalize Probability", "(prob - min) / (max - min) → ใช้เป็น fixed channel ใน state"),
            ("Self-play Data Generation", "สร้างกระดานสุ่มทุก episode → AI เล่นกับตัวเอง → เก็บ transitions"),
            ("Replay Buffer", "50,000 transitions (s, a, r, s′, done) → sample mini-batch 128 แบบสุ่ม"),
            ("Pre-training (Behavioral Cloning)", "500 synthetic states จาก historical prob → warm-start weights ก่อน self-play"),
        ], 1):
            with st.container(border=True):
                st.markdown(f"**ขั้นที่ {i}: {title}**")
                st.caption(detail)

    with tab2:
        st.header("🧠 ทฤษฎีของ Deep Q-Network (DQN)")

        st.subheader("ทำไมต้องใช้ DQN แทน Q-Table?")
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("**❌ Q-Table (เดิม)**")
                st.markdown("""
                - ต้องเก็บ **2¹⁰⁰ ≈ 10³⁰ states** — เป็นไปไม่ได้
                - ไม่ generalize ได้ — state ที่ไม่เคยเห็นไม่รู้จัก
                - Memory: ∞
                """)
        with col2:
            with st.container(border=True):
                st.markdown("**✅ Deep Q-Network (DQN)**")
                st.markdown("""
                - Neural Net **ประมาณ** Q-values — ไม่ต้องเก็บทุก state
                - Generalize ได้ — state คล้ายกัน → Q-value ใกล้กัน
                - Memory: ~500K parameters
                """)

        st.divider()
        st.subheader("Dueling DQN Architecture")
        st.markdown("""
        ```
        Input (300 dims)
          → Dense(256) + BatchNorm + ReLU + Dropout(0.2)
          → Dense(256) + BatchNorm + ReLU + Dropout(0.2)
          → Dense(128) + ReLU
                    ↓               ↓
          Advantage A(s,a)    Value V(s)
          Dense(64) → Dense(100)  Dense(64) → Dense(1)
                    ↓               ↓
          Q(s,a) = V(s) + A(s,a) − mean(A(s,·))
        ```
        """)
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.markdown("**V(s) — Value Stream**")
                st.caption("มูลค่าของ state นี้โดยรวม — ไม่ขึ้นกับ action ที่เลือก")
        with c2:
            with st.container(border=True):
                st.markdown("**A(s,a) — Advantage Stream**")
                st.caption("ข้อได้เปรียบของ action นี้เมื่อเทียบกับค่าเฉลี่ย — ช่วยให้ converge เร็วขึ้น")

        st.divider()
        st.subheader("Double DQN — Bellman Equation")
        st.markdown("""
        ```
        Standard DQN : target = r + γ × max_a Q_target(s', a)
        Double DQN   : a* = argmax_a Q_online(s', a)
                       target = r + γ × Q_target(s', a*)
        ```
        Double DQN ใช้ Q_online เลือก action แต่ใช้ Q_target ประเมินค่า → ลด **overestimation bias**
        """)
        st.divider()
        st.subheader("⚡ Speed Techniques")
        for tech, desc in [
            ("@tf.function + GradientTape", "JIT-compile training step → เร็วขึ้น 10–30× vs model.fit()"),
            ("TRAIN_EVERY = 4", "learn() ทุก 4 steps แทนทุก step → ลด backward pass 75%"),
            ("Experience Replay (50,000)", "sample แบบสุ่ม → break correlation → stable training"),
            ("Target Network (sync ทุก 50 ep)", "ป้องกัน moving target problem → training ไม่ oscillate"),
            ("Vectorized Evaluation", "simulate 500 games พร้อมกัน → model() ~40 calls แทน 20,000 calls"),
        ]:
            with st.container(border=True):
                st.markdown(f"**{tech}**")
                st.caption(desc)

        st.divider()
        st.subheader("Reward Shaping")
        cols = st.columns(4)
        for col, (label, val, color) in zip(cols, [
            ("Miss", "-1", "🔴"), ("Hit", "+10", "🟡"),
            ("Sunk ship", "+50", "🟠"), ("Win", "+100", "🟢"),
        ]):
            col.metric(f"{color} {label}", val)

    with tab3:
        st.header("🔧 ขั้นตอนการพัฒนา DQN Model")
        for i, (title, detail) in enumerate([
            ("โหลดข้อมูล + สร้าง Base Prob Map", "อ่าน CSV → normalize → BASE_PROB_NORM array (100,) ใช้เป็น fixed input channel"),
            ("สร้าง BattleshipEnv", "Environment class: reset() → step(action) → reward + done signal"),
            ("สร้าง AdvantageMean Custom Layer", "แทน layers.Lambda → ใช้ @register_keras_serializable → serialize ได้ปลอดภัย"),
            ("Build Dueling DQN Network", "Shared layers → Advantage stream + Value stream → Q = V + (A - mean(A))"),
            ("สร้าง ReplayBuffer", "deque(maxlen=50,000) → push/sample transitions → break temporal correlation"),
            ("สร้าง DQNAgent + @tf.function", "JIT-compile train step → Double DQN update → ε-Greedy decay"),
            ("Pre-training จาก Historical Data", "500 synthetic states → target = base_prob × 5 → warm-start network weights"),
            ("Self-play Training 3,000 episodes", "ε: 1.0→0.05 → Target network sync ทุก 50 ep → log ep/s + ETA"),
            ("Evaluate + Compare", "Vectorized 500 games → compare vs Hunt+Target → Boxplot + Q-value heatmap"),
            ("Save + Deploy", "model.save(.keras) → %%writefile app.py → streamlit run"),
        ], 1):
            with st.container(border=True):
                st.markdown(f"**ขั้นที่ {i}: {title}**")
                st.caption(detail)

        st.divider()
        st.subheader("📊 ผลการเปรียบเทียบทุกโมเดล")
        st.markdown("""
        | วิธี | หลักการ | Avg Moves | ข้อดี | ข้อจำกัด |
        |------|---------|-----------|-------|----------|
        | Hunt Only | Static probability | ~50 | เรียบง่าย | ไม่ปรับตามเกม |
        | Hunt + Target | Rule-based | ~38 | เข้าใจง่าย | Heuristic |
        | Bayesian Map | Statistical | ~33 | ปรับตามผู้เล่น | ช้ากว่า |
        | Q-Learning | Tabular RL | ~30 | เรียนรู้ได้ | State space จำกัด |
        | **DQN Neural Network** | **Deep RL** | **~26** | **Generalize + เรียนรู้ Pattern** | **ต้องใช้เวลา Train** |
        """)

    with tab4:
        st.header("📚 แหล่งอ้างอิง Neural Network")
        st.subheader("📖 งานวิจัยหลัก DQN")
        for ref in [
            {"title": "Human-level control through deep reinforcement learning", "authors": "Mnih et al. (2015)", "publisher": "Nature, 518, 529–533", "note": "งานวิจัย DQN ต้นฉบับจาก DeepMind", "url": "https://doi.org/10.1038/nature14236"},
            {"title": "Dueling Network Architectures for Deep RL", "authors": "Wang et al. (2016)", "publisher": "ICML 2016", "note": "สถาปัตยกรรม Dueling DQN ที่แยก V(s) และ A(s,a)", "url": "https://arxiv.org/abs/1511.06581"},
            {"title": "Deep Reinforcement Learning with Double Q-learning", "authors": "van Hasselt et al. (2016)", "publisher": "AAAI 2016", "note": "Double DQN ที่ลด overestimation bias", "url": "https://arxiv.org/abs/1509.06461"},
            {"title": "Prioritized Experience Replay", "authors": "Schaul et al. (2016)", "publisher": "ICLR 2016", "note": "การ sample experiences ที่สำคัญมากกว่า", "url": "https://arxiv.org/abs/1511.05952"},
        ]:
            with st.container(border=True):
                st.markdown(f"**{ref['title']}**")
                st.markdown(f"_{ref['authors']}_ — {ref['publisher']}")
                st.caption(f"📌 {ref['note']}")
                st.markdown(f"🔗 {ref['url']}")

        st.subheader("🛠️ Library และ Framework")
        for name, url, desc in [
            ("TensorFlow / Keras", "https://tensorflow.org", "Deep Learning framework สำหรับสร้างและ train DQN"),
            ("NumPy", "https://numpy.org", "Vectorized state arrays และ batch operations"),
            ("Streamlit 1.x", "https://streamlit.io", "Web framework สำหรับ deploy model"),
            ("Pandas", "https://pandas.pydata.org", "โหลดและ preprocess CSV data"),
        ]:
            c1, c2, c3 = st.columns([2, 3, 3])
            with c1: st.markdown(f"**{name}**")
            with c2: st.caption(desc)
            with c3: st.caption(f"🔗 {url}")
            st.divider()

        st.subheader("📝 แนวทางการพัฒนาที่อ้างอิง")
        for title, author, url, note in [
            ("Playing Atari with Deep Reinforcement Learning", "Mnih et al., 2013", "https://arxiv.org/abs/1312.5602", "ต้นกำเนิดของ DQN + Experience Replay"),
            ("Deep Q-Network Tutorial", "TensorFlow Documentation", "https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial", "Tutorial การ implement DQN ด้วย TF"),
            ("Keras Custom Layers Guide", "Keras Documentation", "https://keras.io/guides/making_new_layers_and_models/", "การสร้าง Custom Layer ด้วย @register_keras_serializable"),
            ("Battleship Optimal Strategy", "DataGenetics Blog", "http://www.datagenetics.com/blog/december32011/", "กลยุทธ์ Battleship ที่เป็น baseline"),
        ]:
            with st.container(border=True):
                st.markdown(f"**{title}** — _{author}_")
                st.caption(f"📌 {note}")
                st.markdown(f"🔗 {url}")
