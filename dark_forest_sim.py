#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

STRATS = ["hide", "broadcast", "attack"]

# -------------------- Core Simulation --------------------

def simulate(params, rng):
    N0 = int(params["pop"])
    H0 = max(1, int(N0 * params["frac_hide"]))
    B0 = max(1, int(N0 * params["frac_broadcast"]))
    A0 = max(1, N0 - H0 - B0)
    y = np.array([H0, B0, A0], dtype=float)

    rounds = int(params["rounds"])
    traj = np.zeros((rounds, 3), dtype=float)

    pd_b = params["p_detect_broadcast"]
    pd_h = params["p_detect_hide"]
    p_k  = params["p_kill"]
    p_x  = params["p_expose_attacker"]
    gain_contact = params["contact_benefit"]
    gain_loot    = params["loot_benefit"]
    gain_hide    = params["hide_benefit"]
    mut          = params["mutation_rate"]
    cap          = params["carrying_cap"]
    noise        = params["env_noise"]

    for t in range(rounds):
        H, B, A = y

        visible_B = B * pd_b
        visible_H = H * pd_h
        total_visible = max(visible_B + visible_H, 1e-9)

        attacks_attempted = A * (total_visible / (total_visible + 1.0))
        share_B = visible_B / total_visible
        share_H = visible_H / total_visible

        attacks_on_B = attacks_attempted * share_B
        attacks_on_H = attacks_attempted * share_H

        kills_B = min(B, attacks_on_B * p_k)
        kills_H = min(H, attacks_on_H * p_k)
        attacker_losses = min(A, attacks_attempted * p_x)

        births_B = B * gain_contact
        births_A = A * (gain_loot * (kills_B + kills_H) / (A + 1e-9))
        births_H = H * gain_hide

        if noise > 0.0:
            births_B += rng.normal(0, noise * max(1.0, B*0.1))
            births_A += rng.normal(0, noise * max(1.0, A*0.1))
            births_H += rng.normal(0, noise * max(1.0, H*0.1))

        H_new = H - kills_H + max(0.0, births_H)
        B_new = B - kills_B + max(0.0, births_B)
        A_new = A - attacker_losses + max(0.0, births_A)

        total = H_new + B_new + A_new
        if cap > 0 and total > cap:
            scale = cap / total
            H_new *= scale; B_new *= scale; A_new *= scale

        if mut > 0:
            pool = H_new + B_new + A_new
            if pool > 3:
                def mutate(x):
                    out = x * (1.0 - mut)
                    spill = x * mut
                    return out, spill
                H_keep, H_spill = mutate(H_new)
                B_keep, B_spill = mutate(B_new)
                A_keep, A_spill = mutate(A_new)
                H_new = H_keep + 0.5*B_spill + 0.5*A_spill
                B_new = B_keep + 0.5*H_spill + 0.5*A_spill
                A_new = A_keep + 0.5*H_spill + 0.5*B_spill

        y = np.maximum([H_new, B_new, A_new], 0.0)
        traj[t,:] = y

    return traj

# -------------------- Adaptive UI with GridSpec --------------------

def build_and_run():
    rng = np.random.default_rng()

    params = dict(
        pop=3000,
        frac_hide=0.5,
        frac_broadcast=0.3,
        p_detect_broadcast=0.7,
        p_detect_hide=0.05,
        p_kill=0.6,
        p_expose_attacker=0.15,
        contact_benefit=0.02,
        loot_benefit=0.04,
        hide_benefit=0.01,
        mutation_rate=0.01,
        carrying_cap=5000,
        env_noise=0.0,
        rounds=200
    )

    # Figure + adaptive grid
    fig = plt.figure(figsize=(11, 6), constrained_layout=True)
    # 16 rows: 13 sliders, a bit of padding, 1 button row, padding
    gs = fig.add_gridspec(
        nrows=16, ncols=3,
        width_ratios=[1.3, 3.0, 1.4]  # sliders | plot | info box
    )

    # Main plot uses all rows in center column
    ax = fig.add_subplot(gs[:, 1])

    traj = simulate(params, rng)
    t = np.arange(traj.shape[0])
    line_H, = ax.plot(t, traj[:, 0], label="hide")
    line_B, = ax.plot(t, traj[:, 1], label="broadcast")
    line_A, = ax.plot(t, traj[:, 2], label="attack")
    ax.set_xlabel("rounds")
    ax.set_ylabel("population")
    ax.set_title("Dark Forest Toy Model — strategy populations over time")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # --- Sliders in left column (adaptive with window size) ---
    slider_rows = {
        "population":         0,
        "frac_hide":          1,
        "frac_broadcast":     2,
        "p_detect_broadcast": 3,
        "p_detect_hide":      4,
        "p_kill":             5,
        "p_expose_attacker":  6,
        "contact_benefit":    7,
        "loot_benefit":       8,
        "hide_benefit":       9,
        "mutation_rate":      10,
        "carrying_cap":       11,
        "env_noise":          12,
        "rounds":             13,
    }

    def add_slider(label, vmin, vmax, vinit, row):
        ax_s = fig.add_subplot(gs[row, 0])
        ax_s.set_anchor("W")
        return Slider(ax_s, label, vmin, vmax, valinit=vinit)

    s_pop   = add_slider("population",           100, 20000, params["pop"],                 slider_rows["population"])
    s_fracH = add_slider("frac_hide",            0.0, 1.0,   params["frac_hide"],          slider_rows["frac_hide"])
    s_fracB = add_slider("frac_broadcast",       0.0, 1.0,   params["frac_broadcast"],     slider_rows["frac_broadcast"])
    s_pdB   = add_slider("p_detect_broadcast",   0.0, 1.0,   params["p_detect_broadcast"], slider_rows["p_detect_broadcast"])
    s_pdH   = add_slider("p_detect_hide",        0.0, 0.3,   params["p_detect_hide"],      slider_rows["p_detect_hide"])
    s_pk    = add_slider("p_kill",               0.0, 1.0,   params["p_kill"],             slider_rows["p_kill"])
    s_pxa   = add_slider("p_expose_attacker",    0.0, 1.0,   params["p_expose_attacker"],  slider_rows["p_expose_attacker"])
    s_benC  = add_slider("contact_benefit",      0.0, 0.10,  params["contact_benefit"],    slider_rows["contact_benefit"])
    s_benL  = add_slider("loot_benefit",         0.0, 0.10,  params["loot_benefit"],       slider_rows["loot_benefit"])
    s_benH  = add_slider("hide_benefit",         0.0, 0.05,  params["hide_benefit"],       slider_rows["hide_benefit"])
    s_mut   = add_slider("mutation_rate",        0.0, 0.20,  params["mutation_rate"],      slider_rows["mutation_rate"])
    s_cap   = add_slider("carrying_cap",         0,   50000, params["carrying_cap"],       slider_rows["carrying_cap"])
    s_noise = add_slider("env_noise",            0.0, 0.50,  params["env_noise"],          slider_rows["env_noise"])
    s_round = add_slider("rounds",               50,  2000,  params["rounds"],             slider_rows["rounds"])

    # Button in left column, bottom-ish
    ax_btn = fig.add_subplot(gs[15, 0])
    btn = Button(ax_btn, "Run / Recompute")

    # Info box in right column, spanning many rows, adaptive size
    ax_info = fig.add_subplot(gs[2:14, 2])
    ax_info.axis("off")
    txt = ax_info.text(
        0.0, 0.5, "",
        transform=ax_info.transAxes,
        va="center", ha="left", family="monospace", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8)
    )

    def recompute(event=None):
        frac_hide = s_fracH.val
        frac_brd  = s_fracB.val
        if frac_hide + frac_brd > 0.98:
            total = frac_hide + frac_brd
            frac_hide /= total
            frac_brd  /= total
            s_fracH.set_val(frac_hide)
            s_fracB.set_val(frac_brd)
        frac_att = max(0.0, 1.0 - (frac_hide + frac_brd))

        new_params = dict(
            pop=int(s_pop.val),
            frac_hide=frac_hide,
            frac_broadcast=frac_brd,
            p_detect_broadcast=s_pdB.val,
            p_detect_hide=s_pdH.val,
            p_kill=s_pk.val,
            p_expose_attacker=s_pxa.val,
            contact_benefit=s_benC.val,
            loot_benefit=s_benL.val,
            hide_benefit=s_benH.val,
            mutation_rate=s_mut.val,
            carrying_cap=int(s_cap.val),
            env_noise=s_noise.val,
            rounds=int(s_round.val)
        )

        traj = simulate(new_params, rng)
        t = np.arange(traj.shape[0])
        line_H.set_data(t, traj[:, 0])
        line_B.set_data(t, traj[:, 1])
        line_A.set_data(t, traj[:, 2])
        ax.set_xlim(0, len(t) - 1)
        ymax = max(1.0, traj.max() * 1.05)
        ax.set_ylim(0, ymax)

        param_text = (
            "Init mix:\n"
            f"  H={frac_hide:.2f} | B={frac_brd:.2f} | A={frac_att:.2f}\n\n"
            "Detection & Conflict:\n"
            f"  pd(B)={s_pdB.val:.2f} | pd(H)={s_pdH.val:.2f}\n"
            f"  p_kill={s_pk.val:.2f} | expose_A={s_pxa.val:.2f}\n\n"
            "Benefits:\n"
            f"  contact={s_benC.val:.3f}\n"
            f"  loot={s_benL.val:.3f}\n"
            f"  hide={s_benH.val:.3f}\n\n"
            f"Mutation={s_mut.val:.3f}\n"
            f"Cap={int(s_cap.val)} | Rounds={int(s_round.val)}"
        )

        txt.set_text(param_text)
        fig.canvas.draw_idle()

    btn.on_clicked(recompute)
    recompute()
    plt.show()


if __name__ == "__main__":
    build_and_run()
