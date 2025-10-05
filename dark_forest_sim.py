#!/usr/bin/env python3
"""
Dark Forest Simulator (Interactive, matplotlib sliders)
-------------------------------------------------------

A simple game-theoretic toy model showing how "hide / broadcast / attack"
strategies compete over time under detection and conflict risks —
illustrating how Dark Forest behavior can emerge.

Run:
    python dark_forest_sim.py

Controls (sliders):
- Population + initial strategy mix
- Detection: broadcast vs hide
- Attack success and attacker exposure
- Benefits: contact (broadcast), loot (attack), survival (hide)
- Mutation rate (strategy switching noise)
- Rounds per run

Then press "Run" to recompute.

Note: This is a stylized model for intuition, not a scientific result.
"""

import math
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# -------------------- Core Simulation --------------------

STRATS = ["hide", "broadcast", "attack"]

def simulate(params, rng):
    """
    Simulate population dynamics across 'rounds' timesteps.
    State is the vector of counts [H, B, A].
    """
    N0 = int(params["pop"])
    # initial fractions -> counts
    H0 = max(1, int(N0 * params["frac_hide"]))
    B0 = max(1, int(N0 * params["frac_broadcast"]))
    A0 = max(1, N0 - H0 - B0)
    y = np.array([H0, B0, A0], dtype=float)

    rounds = int(params["rounds"])
    traj = np.zeros((rounds, 3), dtype=float)

    # unpack parameters
    pd_b = params["p_detect_broadcast"]   # detection prob vs broadcasters
    pd_h = params["p_detect_hide"]        # detection prob vs hiders
    p_k  = params["p_kill"]               # success of attack
    p_x  = params["p_expose_attacker"]    # attacker exposure risk per attack
    gain_contact = params["contact_benefit"]
    gain_loot    = params["loot_benefit"]
    gain_hide    = params["hide_benefit"]
    mut          = params["mutation_rate"]
    cap          = params["carrying_cap"]
    noise        = params["env_noise"]

    for t in range(rounds):
        H, B, A = y

        # 1) Detection pools: how many potential targets can be seen
        #    Attackers preferentially see broadcasters.
        visible_B = B * pd_b
        visible_H = H * pd_h
        total_visible = visible_B + visible_H
        # avoid zero
        total_visible = max(total_visible, 1e-9)

        # 2) Expected attacks executed (each attacker attempts ~1 attack per round scaled by visibility)
        attacks_attempted = A * (total_visible / (total_visible + 1.0))
        # 3) Distribute attacks across B and H by their visibility share
        share_B = visible_B / total_visible
        share_H = visible_H / total_visible

        attacks_on_B = attacks_attempted * share_B
        attacks_on_H = attacks_attempted * share_H

        # 4) Expected kills
        kills_B = min(B, attacks_on_B * p_k)
        kills_H = min(H, attacks_on_H * p_k)

        # 5) Attacker attrition due to exposure/counterattack
        attacker_losses = min(A, attacks_attempted * p_x)

        # 6) Payoffs / reproduction (very stylized)
        births_B = B * gain_contact
        births_A = A * (gain_loot * (kills_B + kills_H) / (A + 1e-9))
        births_H = H * gain_hide

        # 7) Environmental noise (can be negative or positive)
        if noise > 0.0:
            births_B += rng.normal(0, noise * max(1.0, B*0.1))
            births_A += rng.normal(0, noise * max(1.0, A*0.1))
            births_H += rng.normal(0, noise * max(1.0, H*0.1))

        # 8) Update populations
        H_new = H - kills_H + max(0.0, births_H)
        B_new = B - kills_B + max(0.0, births_B)
        A_new = A - attacker_losses + max(0.0, births_A)

        # 9) Soft carrying capacity (logistic cap)
        total = H_new + B_new + A_new
        if cap > 0 and total > cap:
            scale = cap / total
            H_new *= scale; B_new *= scale; A_new *= scale

        # 10) Mutation / strategy switching (noise in cultural evolution)
        if mut > 0:
            pool = H_new + B_new + A_new
            if pool > 3:
                # move a fraction mut of each into the others equally
                def mutate(x):
                    out = x * (1.0 - mut)
                    spill = x * mut
                    return out, spill
                H_keep, H_spill = mutate(H_new)
                B_keep, B_spill = mutate(B_new)
                A_keep, A_spill = mutate(A_new)
                # distribute spills equally among the other two strategies
                H_new = H_keep + 0.5*B_spill + 0.5*A_spill
                B_new = B_keep + 0.5*H_spill + 0.5*A_spill
                A_new = A_keep + 0.5*H_spill + 0.5*B_spill

        # 11) Prevent negatives and tiny drifts
        y = np.maximum([H_new, B_new, A_new], 0.0)
        traj[t,:] = y

    return traj

# -------------------- UI --------------------

def build_and_run():
    rng = np.random.default_rng()

    # Default parameters
    params = dict(
        pop=3000,
        frac_hide=0.5,
        frac_broadcast=0.3,  # frac_attack = 0.2 (implicit)
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

    # Create figure
    fig, ax = plt.subplots(figsize=(9, 6))
    plt.subplots_adjust(left=0.28, bottom=0.35)  # room for sliders

    # Initial run
    traj = simulate(params, rng)
    t = np.arange(traj.shape[0])
    line_H, = ax.plot(t, traj[:,0], label="hide")
    line_B, = ax.plot(t, traj[:,1], label="broadcast")
    line_A, = ax.plot(t, traj[:,2], label="attack")
    ax.set_xlabel("rounds")
    ax.set_ylabel("population")
    ax.set_title("Dark Forest Toy Model — strategy populations over time")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Sliders
    # left column x-position
    SL = 0.05
    SW = 0.20
    SH = 0.03
    GAP = 0.01
    y0 = 0.92

    def add_slider(label, valmin, valmax, valinit, ypos):
        ax_s = plt.axes([SL, ypos, SW, SH])
        return Slider(ax_s, label, valmin, valmax, valinit=valinit)

    s_pop   = add_slider("population",           100, 20000, params["pop"], y0)
    s_fracH = add_slider("frac_hide",            0.0, 1.0,   params["frac_hide"], y0-1*(SH+GAP))
    s_fracB = add_slider("frac_broadcast",       0.0, 1.0,   params["frac_broadcast"], y0-2*(SH+GAP))
    s_pdB   = add_slider("p_detect_broadcast",   0.0, 1.0,   params["p_detect_broadcast"], y0-3*(SH+GAP))
    s_pdH   = add_slider("p_detect_hide",        0.0, 0.3,   params["p_detect_hide"], y0-4*(SH+GAP))
    s_pk    = add_slider("p_kill",               0.0, 1.0,   params["p_kill"], y0-5*(SH+GAP))
    s_pxa   = add_slider("p_expose_attacker",    0.0, 1.0,   params["p_expose_attacker"], y0-6*(SH+GAP))
    s_benC  = add_slider("contact_benefit",      0.0, 0.10,  params["contact_benefit"], y0-7*(SH+GAP))
    s_benL  = add_slider("loot_benefit",         0.0, 0.10,  params["loot_benefit"], y0-8*(SH+GAP))
    s_benH  = add_slider("hide_benefit",         0.0, 0.05,  params["hide_benefit"], y0-9*(SH+GAP))
    s_mut   = add_slider("mutation_rate",        0.0, 0.20,  params["mutation_rate"], y0-10*(SH+GAP))
    s_cap   = add_slider("carrying_cap",         0,   50000, params["carrying_cap"], y0-11*(SH+GAP))
    s_noise = add_slider("env_noise",            0.0, 0.50,  params["env_noise"], y0-12*(SH+GAP))
    s_round = add_slider("rounds",               50,  2000,  params["rounds"], y0-13*(SH+GAP))

    # Recompute button
    ax_btn = plt.axes([SL, 0.05, SW, 0.05])
    btn = Button(ax_btn, "Run / Recompute")

    # Info box
    txt = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top", ha="left")

    def recompute(event=None):
        # enforce fractions sum <= 1, remaining goes to attack
        frac_hide = s_fracH.val
        frac_brd  = s_fracB.val
        if frac_hide + frac_brd > 0.98:
            # normalize
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
        line_H.set_data(t, traj[:,0])
        line_B.set_data(t, traj[:,1])
        line_A.set_data(t, traj[:,2])
        ax.set_xlim(0, len(t)-1)
        ymax = max(1.0, traj.max()*1.05)
        ax.set_ylim(0, ymax)

        txt.set_text(
            f"init mix: H={frac_hide:.2f}, B={frac_brd:.2f}, A={frac_att:.2f}\n"
            f"pd(B)={s_pdB.val:.2f}, pd(H)={s_pdH.val:.2f}, p_kill={s_pk.val:.2f}, expose_A={s_pxa.val:.2f}\n"
            f"benefits: contact={s_benC.val:.3f}, loot={s_benL.val:.3f}, hide={s_benH.val:.3f}\n"
            f"mutation={s_mut.val:.3f}, cap={int(s_cap.val)}, rounds={int(s_round.val)}"
        )

        fig.canvas.draw_idle()

    btn.on_clicked(recompute)

    # Also recompute when any slider changes (live)
    for s in [s_pop, s_fracH, s_fracB, s_pdB, s_pdH, s_pk, s_pxa, s_benC, s_benL, s_benH, s_mut, s_cap, s_noise, s_round]:
        s.on_changed(lambda val: None)  # required to register
    # We'll trigger actual recompute on button press to avoid excessive CPU.

    recompute()
    plt.show()


if __name__ == "__main__":
    build_and_run()
