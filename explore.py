"""
Corner Delivery Map - Cycle 10, Day 3 project
Where do corners land, and does the landing zone change what happens next?
Every corner in Copa America 2024 is mirrored to one side of the pitch,
placed in one of eight zones by where the ball was delivered, and linked to
the shots that followed.

Method:
  1. Corners = passes labelled as corners (periods 1 to 4, shootouts out).
     If the pass type is missing, the first pass of a possession that starts
     from a corner is used instead.
  2. Corners from the far side are mirrored so every corner is taken from the
     same side. The goal is at the top of the map, the corner flag at the top
     left.
  3. Zones from the end point of the delivery:
       Short corner   ball travels less than 20 yards
       Near post      within 9 yards of the goal line, near third of the box
       Central deep   within 9 yards of the goal line, central third
       Far post       within 9 yards of the goal line, far third
       Near edge, Penalty spot, Far edge   the same thirds, 9 to 18 yards out
       Outside the box   everything else
  4. Shots linked to a corner = non-penalty shots by the same team, in the
     same possession and period, taken within 15 seconds of the corner. Second
     balls are therefore counted, long build-ups are not.
  5. Per zone and per delivery type (inswing, outswing, straight, short):
     number of corners, share with a shot, and xG per corner. Ranges are the
     5th to 95th percentile from 5,000 bootstrap resamples of the corners and
     are shown only for groups with at least 8 corners.

Limitations: about 300 corners, so each zone has few corners and the ranges
are wide. Landing zone is not the same as the player who got the ball: the
data has no blocking, runs or marking. xG measures the shots that followed,
not the quality of the delivery itself. A zone may look good because stronger
teams aim there more often. The 15 second cut-off is a choice.

Outputs: corner_landing_map.png, corner_zone_value.png,
corner_replay.mp4 (corner-by-corner replay plus the two charts as end cards),
corner_deliveries.csv, corner_zone_summary.csv

Reuses: copa_america_2024_matches.csv, copa_america_2024_events.csv
"""

import os
import subprocess

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Rectangle, Arc
import imageio_ffmpeg

matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

FOCUS_TEAM = "Argentina"
SHOT_WINDOW = 15      # seconds after the corner
SHORT_DIST = 20       # yards
MIN_N = 8             # smallest group that gets a range
N_BOOT = 5000
SEED = 5
END_CARDS = ["corner_landing_map.png", "corner_zone_value.png"]
END_CARD_SECONDS = 4

BG = "#0d1b2a"
GRID = "#3a4a5a"
BLUE = "#4a90d9"
ORANGE = "#f4a300"
MUTED = "#a8b8c8"
GREEN = "#6fcf97"
RED = "#e76f51"

ZONES = ["Near post", "Central deep", "Far post", "Near edge", "Penalty spot",
         "Far edge", "Outside the box", "Short corner"]
TYPES = ["Inswing", "Outswing", "Straight", "Short"]
# label spots on the map (horizontal = y across the pitch, vertical = x towards goal)
LABEL_SPOT = {"Near post": (26, 115.5), "Central deep": (40, 115.5), "Far post": (54, 115.5),
              "Near edge": (26, 106.0), "Penalty spot": (40, 104.5), "Far edge": (54, 106.0),
              "Outside the box": (71, 94.0), "Short corner": (9, 112.0)}

matches = pd.read_csv("copa_america_2024_matches.csv")
events = pd.read_csv("copa_america_2024_events.csv", low_memory=False)
events = events[events["period"].isin([1, 2, 3, 4])].copy()
for col in ["pass_type", "pass_technique", "pass_end_location", "pass_outcome", "pass_length",
            "play_pattern", "shot_type", "shot_outcome", "shot_statsbomb_xg", "location",
            "minute", "second", "index", "possession"]:
    if col not in events.columns:
        events[col] = np.nan


def parse(col, a, b):
    xy = events[col].astype(str).str.extract(r"\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)")
    events[a] = pd.to_numeric(xy[0], errors="coerce")
    events[b] = pd.to_numeric(xy[1], errors="coerce")


parse("location", "sx", "sy")
parse("pass_end_location", "ex", "ey")
events["minute"] = events["minute"].fillna(0)
events["second"] = events["second"].fillna(0)
events["t"] = events["minute"] * 60 + events["second"]
events = events.sort_values(["match_id", "index"] if events["index"].notna().any() else ["match_id", "t"])
events = events.reset_index(drop=True)

# ---------- corners ----------
is_pass = events["type"] == "Pass"
if (events["pass_type"] == "Corner").any():
    corners = events[is_pass & (events["pass_type"] == "Corner")].copy()
else:
    print("pass_type not found, using the first pass of possessions that start from a corner")
    cand = events[is_pass & (events["play_pattern"] == "From Corner")]
    corners = cand.groupby(["match_id", "possession", "team"], as_index=False).first()
corners = corners[corners["sx"].notna() & corners["ex"].notna() & (corners["sx"] >= 100)].copy()
print(f"\nCorners found: {len(corners)} in {corners['match_id'].nunique()} matches")

# mirror to one side
flip = corners["sy"] > 40
corners["sy_m"] = np.where(flip, 80 - corners["sy"], corners["sy"])
corners["ey_m"] = np.where(flip, 80 - corners["ey"], corners["ey"])
dist = corners["pass_length"].copy()
calc = np.hypot(corners["ex"] - corners["sx"], corners["ey"] - corners["sy"])
corners["dist"] = dist.where(dist.notna(), calc)


def zone_of(x, y, d):
    if d < SHORT_DIST:
        return "Short corner"
    if x >= 102 and 18 <= y <= 62:
        deep = x >= 111
        side = "near" if y < 34 else ("far" if y > 46 else "central")
        if deep:
            return {"near": "Near post", "central": "Central deep", "far": "Far post"}[side]
        return {"near": "Near edge", "central": "Penalty spot", "far": "Far edge"}[side]
    return "Outside the box"


def type_of(tech, d):
    if d < SHORT_DIST:
        return "Short"
    t = str(tech)
    if "Inswing" in t:
        return "Inswing"
    if "Outswing" in t:
        return "Outswing"
    return "Straight"


corners["zone"] = [zone_of(x, y, d) for x, y, d in zip(corners["ex"], corners["ey_m"], corners["dist"])]
corners["dkind"] = [type_of(t, d) for t, d in zip(corners["pass_technique"], corners["dist"])]
corners["found_teammate"] = corners["pass_outcome"].isna()

# ---------- shots that followed ----------
shots = events[(events["type"] == "Shot") & events["shot_statsbomb_xg"].notna()
               & (events["shot_type"].fillna("") != "Penalty")]
sh = shots[["match_id", "possession", "team", "period", "index", "t", "shot_statsbomb_xg", "shot_outcome"]].rename(
    columns={"index": "s_index", "t": "s_t", "period": "s_period"})
cn = corners.reset_index(drop=True)
cn["cid"] = cn.index
mg = cn[["cid", "match_id", "possession", "team", "period", "index", "t"]].merge(
    sh, on=["match_id", "possession", "team"], how="inner")
mg = mg[(mg["s_period"] == mg["period"]) & (mg["s_index"] > mg["index"])
        & (mg["s_t"] - mg["t"] <= SHOT_WINDOW) & (mg["s_t"] - mg["t"] >= 0)]
agg = mg.groupby("cid").agg(n_shots=("s_index", "size"), xg=("shot_statsbomb_xg", "sum"),
                            goals=("shot_outcome", lambda s: int((s == "Goal").sum())))
cn = cn.merge(agg, left_on="cid", right_index=True, how="left")
cn["n_shots"] = cn["n_shots"].fillna(0).astype(int)
cn["xg"] = cn["xg"].fillna(0.0)
cn["goals"] = cn["goals"].fillna(0).astype(int)
cn["has_shot"] = cn["n_shots"] > 0
n_all = len(cn)
if n_all == 0:
    raise ValueError("No corners found in the events file")

# match info for labels
mi = matches.set_index("match_id")
cn["match_date"] = cn["match_id"].map(mi["match_date"]) if "match_date" in mi.columns else ""
cn["opp"] = [(mi.loc[m, "away_team"] if mi.loc[m, "home_team"] == t else mi.loc[m, "home_team"])
             if m in mi.index else "" for m, t in zip(cn["match_id"], cn["team"])]
cn[["match_id", "team", "opp", "period", "minute", "second", "zone", "dkind", "dist", "ex", "ey_m",
    "found_teammate", "n_shots", "xg", "goals"]].to_csv("corner_deliveries.csv", index=False)

rng = np.random.default_rng(SEED)


def boot_ci(x, scale=1.0):
    x = np.asarray(x, dtype=float)
    if len(x) < MIN_N:
        return np.nan, np.nan
    bm = rng.choice(x, (N_BOOT, len(x))).mean(axis=1) * scale
    lo, hi = np.percentile(bm, [5, 95])
    return float(lo), float(hi)


def summarise(df, key, order):
    rows = []
    for name in order:
        g = df[df[key] == name]
        n = len(g)
        sr_lo, sr_hi = boot_ci(g["has_shot"].astype(float), 100)
        xg_lo, xg_hi = boot_ci(g["xg"])
        rows.append(dict(group=name, n=n,
                         shot_rate=g["has_shot"].mean() * 100 if n else np.nan,
                         sr_lo=sr_lo, sr_hi=sr_hi,
                         xg_per_corner=g["xg"].mean() if n else np.nan, xg_lo=xg_lo, xg_hi=xg_hi,
                         goals=int(g["goals"].sum()), shots=int(g["n_shots"].sum())))
    return pd.DataFrame(rows)


zs = summarise(cn, "zone", ZONES)
ts = summarise(cn, "dkind", TYPES)
pd.concat([zs.assign(table="zone"), ts.assign(table="delivery_type")]).to_csv("corner_zone_summary.csv", index=False)
all_sr = cn["has_shot"].mean() * 100
all_xg = cn["xg"].mean()

print(f"\nAll corners: {n_all}, {cn['has_shot'].sum()} with a shot within {SHOT_WINDOW} s "
      f"({all_sr:.1f}%), {int(cn['n_shots'].sum())} shots, {int(cn['goals'].sum())} goals, "
      f"{all_xg:.3f} xG per corner")
print(f"Found a teammate with the delivery: {cn['found_teammate'].mean() * 100:.0f}%")
print("\nBy landing zone:")
print(zs.round(2).to_string(index=False))
print("\nBy delivery type:")
print(ts.round(2).to_string(index=False))
foc = cn[cn["team"] == FOCUS_TEAM]
if len(foc):
    print(f"\n{FOCUS_TEAM}: {len(foc)} corners, {int(foc['has_shot'].sum())} with a shot, "
          f"{foc['xg'].sum():.2f} xG ({foc['xg'].mean():.3f} per corner, tournament {all_xg:.3f})")
off_map = int((cn["ex"] < 72).sum())
if off_map:
    print(f"{off_map} corners ended more than 48 yards from goal and are not drawn on the map")


# ---------- pitch drawing (goal at the top, corner flag at the top left) ----------
def draw_attack_pitch(ax):
    ax.set_facecolor("#10253a")
    line = "#35506b"
    ax.plot([0, 0, 80, 80], [72, 120, 120, 72], color=line, linewidth=1.8)
    ax.add_patch(Rectangle((18, 102), 44, 18, fill=False, edgecolor=line, linewidth=1.6))
    ax.add_patch(Rectangle((30, 114), 20, 6, fill=False, edgecolor=line, linewidth=1.6))
    ax.add_patch(Arc((40, 108), 20, 20, theta1=216.87, theta2=323.13, color=line, linewidth=1.6))
    ax.scatter([40], [108], s=14, color=line, zorder=2)
    ax.plot([36, 44], [120, 120], color="white", linewidth=3.5)
    for u in (34, 46):
        ax.plot([u, u], [102, 120], color=line, linewidth=1.1, linestyle="--")
    ax.plot([18, 62], [111, 111], color=line, linewidth=1.1, linestyle="--")
    ax.set_xlim(-2, 82)
    ax.set_ylim(72, 122)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def zone_labels(ax, counts=None, size=11):
    for i, zn in enumerate(ZONES, start=1):
        u, v = LABEL_SPOT[zn]
        txt = str(i) if counts is None else str(counts.get(zn, 0))
        ax.text(u, v, txt, ha="center", va="center", color="white", fontsize=size, fontweight="bold",
                zorder=6, bbox=dict(boxstyle="circle,pad=0.25", facecolor="#1c3550", edgecolor=MUTED, linewidth=1))


# ---------- chart 1: landing map and zone table ----------
fig1 = plt.figure(figsize=(11, 9.5))
fig1.patch.set_facecolor(BG)
ax = fig1.add_axes([0.04, 0.40, 0.92, 0.50])
draw_attack_pitch(ax)
vis = cn[(cn["ex"] >= 72)]
miss = vis[~vis["has_shot"]]
hit = vis[vis["has_shot"]]
ax.scatter(miss["ey_m"], miss["ex"], s=34, color=BLUE, alpha=0.55, edgecolors="none", zorder=3)
ax.scatter(hit["ey_m"], hit["ex"], s=70, color=ORANGE, alpha=0.95, edgecolors="white", linewidths=0.8, zorder=4)
zone_labels(ax)
fig1.text(0.5, 0.965, "Where Do Corners Land?", ha="center", color="white", fontsize=19, fontweight="bold")
fig1.text(0.5, 0.925, f"{n_all} corners, Copa America 2024, all mirrored to one side. Goal at the top, corner flag top left",
          ha="center", color=MUTED, fontsize=10.5)
fig1.text(0.5, 0.372, "Blue: no shot within 15 seconds.   Orange: a shot followed.", ha="center",
          color=MUTED, fontsize=10.5)
axt = fig1.add_axes([0.04, 0.02, 0.92, 0.34])
axt.set_facecolor(BG)
axt.axis("off")
MONO = "DejaVu Sans Mono"
axt.text(0.0, 0.97, f"{'Zone':<20}{'Corners':>8}{'With a shot':>14}{'xG per corner':>16}", family=MONO,
         color=MUTED, fontsize=11.5, fontweight="bold", va="top", transform=axt.transAxes)
for i, r in enumerate(zs.itertuples(), start=1):
    sr = "-" if r.n == 0 else f"{r.shot_rate:.1f}%"
    xgv = "-" if r.n == 0 else f"{r.xg_per_corner:.3f}"
    axt.text(0.0, 0.97 - 0.105 * i, f"{i}  {r.group:<17}{r.n:>8}{sr:>14}{xgv:>16}", family=MONO,
             color="white", fontsize=11.5, va="top", transform=axt.transAxes)
axt.text(0.0, 0.97 - 0.105 * 9.2, f"{'All corners':<20}{n_all:>8}{all_sr:>13.1f}%{all_xg:>16.3f}", family=MONO,
         color=ORANGE, fontsize=11.5, fontweight="bold", va="top", transform=axt.transAxes)
fig1.savefig("corner_landing_map.png", dpi=150, facecolor=BG)
plt.close(fig1)
print("\nSaved chart: corner_landing_map.png")

# ---------- chart 2: value by zone and by delivery type ----------
fig2, axs = plt.subplots(2, 2, figsize=(11, 8.5), gridspec_kw={"height_ratios": [2, 1.15]})
fig2.patch.set_facecolor(BG)


def hbar(ax, tab, col_mean, col_lo, col_hi, ref, color, fmt):
    ax.set_facecolor(BG)
    tab = tab.iloc[::-1].reset_index(drop=True)
    vals = [0.0, ref]
    for yi, r in tab.iterrows():
        mean = r[col_mean]
        if not np.isfinite(mean):
            continue
        lo, hi = r[col_lo], r[col_hi]
        ax.barh(yi, mean, height=0.62, color=color)
        if np.isfinite(lo):
            ax.errorbar(mean, yi, xerr=[[mean - lo], [hi - mean]], fmt="none", ecolor="white",
                        capsize=3, linewidth=1.2)
        vals += [mean, hi if np.isfinite(hi) else mean]
    top = max(vals)
    ax.set_xlim(0, top * 1.28)
    for yi, r in tab.iterrows():
        mean = r[col_mean]
        if not np.isfinite(mean):
            continue
        end = r[col_hi] if np.isfinite(r[col_hi]) else mean
        ax.text(end + top * 0.02, yi, fmt.format(mean), va="center", color="white", fontsize=10)
    ax.axvline(ref, color="white", linewidth=1, linestyle="--")
    ax.set_yticks(range(len(tab)))
    ax.set_yticklabels([f"{g} (n={n})" for g, n in zip(tab["group"], tab["n"])], color="white", fontsize=10)
    ax.tick_params(axis="x", colors="white", labelsize=9)
    ax.tick_params(axis="y", length=0)
    for sp in ax.spines.values():
        sp.set_color(GRID)


hbar(axs[0, 0], zs, "shot_rate", "sr_lo", "sr_hi", all_sr, BLUE, "{:.1f}%")
hbar(axs[0, 1], zs, "xg_per_corner", "xg_lo", "xg_hi", all_xg, BLUE, "{:.3f}")
hbar(axs[1, 0], ts, "shot_rate", "sr_lo", "sr_hi", all_sr, ORANGE, "{:.1f}%")
hbar(axs[1, 1], ts, "xg_per_corner", "xg_lo", "xg_hi", all_xg, ORANGE, "{:.3f}")
axs[0, 0].set_title("Corners with a shot (%), by landing zone", color="white", fontsize=11, pad=6)
axs[0, 1].set_title("xG per corner, by landing zone", color="white", fontsize=11, pad=6)
axs[1, 0].set_title("Corners with a shot (%), by delivery type", color="white", fontsize=11, pad=6)
axs[1, 1].set_title("xG per corner, by delivery type", color="white", fontsize=11, pad=6)
fig2.suptitle("Does the Landing Zone Change What a Corner Is Worth?", color="white", fontsize=17,
              fontweight="bold", y=0.985)
fig2.text(0.5, 0.925, "Dashed line: all corners. Copa America 2024, shots within 15 seconds of the corner",
          ha="center", color=MUTED, fontsize=10.5)
fig2.text(0.5, 0.012, f"Whiskers: 5th to 95th percentile (5,000 resamples), shown for groups with {MIN_N}+ corners.",
          ha="center", color=MUTED, fontsize=9.5)
plt.tight_layout(rect=[0, 0.03, 1, 0.91])
fig2.savefig("corner_zone_value.png", dpi=150, facecolor=BG)
plt.close(fig2)
print("Saved chart: corner_zone_value.png")

# ---------- replay of one team's corners, one at a time ----------
rep = cn[cn["team"] == FOCUS_TEAM].copy()
rep_team = FOCUS_TEAM
if len(rep) < 5:
    rep = cn.copy()
    rep_team = "all teams"
sort_keys = ["match_date", "match_id", "period", "t"] if "match_date" in rep.columns else ["match_id", "period", "t"]
rep = rep.sort_values(sort_keys).reset_index(drop=True)
rep = rep[rep["ex"] >= 72].reset_index(drop=True)
n_rep = len(rep)
FPC = int(np.clip(280 // max(n_rep, 1), 6, 12))
FLIGHT = max(int(FPC * 0.65), 4)
print(f"\nReplaying {n_rep} corners of {rep_team}, {FPC} frames each")

fig = plt.figure(figsize=(10, 10), dpi=108)  # 1080 x 1080 pixels
fig.patch.set_facecolor(BG)
axp = fig.add_axes([0.04, 0.34, 0.92, 0.52])
title_name = rep_team if rep_team != "all teams" else "Corners"
fig.text(0.5, 0.965, f"Where Do {title_name}'s Corners Land?" if rep_team != "all teams" else "Where Do Corners Land?",
         ha="center", color="white", fontsize=23, fontweight="bold")
fig.text(0.5, 0.928, "Copa America 2024, every corner mirrored to one side", ha="center", color=MUTED, fontsize=14)
info_t = fig.text(0.5, 0.885, "", ha="center", color="white", fontsize=17, fontweight="bold")
note_t = fig.text(0.5, 0.315, "Blue: no shot within 15 seconds.   Orange: a shot followed.   Numbers: corners in each zone.",
                  ha="center", color=MUTED, fontsize=11)
sum_t = fig.text(0.5, 0.262, "", ha="center", color="white", fontsize=21, fontweight="bold")
last_t = fig.text(0.5, 0.208, "", ha="center", color=GREEN, fontsize=16)
last2_t = fig.text(0.5, 0.165, "", ha="center", color=MUTED, fontsize=14)
fig.text(0.5, 0.012, "StatsBomb open data, Copa America 2024. Shots within 15 seconds of the corner.",
         ha="center", color=MUTED, fontsize=9)


def ease(p):
    p = min(max(p, 0.0), 1.0)
    return p * p * (3 - 2 * p)


def bezier(p0, p1, p2, t):
    return ((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0],
            (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1])


def draw(i):
    k = min(i // FPC, n_rep - 1) if i < n_rep * FPC else n_rep - 1
    f = i - k * FPC if i < n_rep * FPC else FPC
    final = i >= n_rep * FPC
    axp.clear()
    draw_attack_pitch(axp)
    landed = rep.iloc[: k + (1 if (final or f >= FLIGHT) else 0)]
    counts = landed["zone"].value_counts().to_dict()
    zone_labels(axp, counts=counts, size=11)
    for r in landed.itertuples():
        if r.has_shot:
            axp.scatter([r.ey_m], [r.ex], s=120, color=ORANGE, edgecolors="white", linewidths=1, zorder=4)
        else:
            axp.scatter([r.ey_m], [r.ex], s=60, color=BLUE, alpha=0.6, edgecolors="none", zorder=3)
    cur = rep.iloc[k]
    if not final and f < FLIGHT:
        p0 = (cur.sy_m, 120.0)
        p2 = (cur.ey_m, cur.ex)
        mid = ((p0[0] + p2[0]) / 2, (p0[1] + p2[1]) / 2)
        bend = 7 if "Inswing" in cur.dkind else (-7 if "Outswing" in cur.dkind else 3)
        p1 = (mid[0] + bend, mid[1] + 6)
        tt = ease((f + 1) / FLIGHT)
        xs = [bezier(p0, p1, p2, s)[0] for s in np.linspace(0, tt, 14)]
        ys = [bezier(p0, p1, p2, s)[1] for s in np.linspace(0, tt, 14)]
        axp.plot(xs, ys, color="white", linewidth=2.2, alpha=0.85, zorder=5)
        axp.scatter([xs[-1]], [ys[-1]], s=90, color="white", edgecolors=BG, linewidths=1.5, zorder=7)
    elif not final and cur.has_shot:
        ring = (f - FLIGHT + 1) / max(FPC - FLIGHT, 1)
        axp.scatter([cur.ey_m], [cur.ex], s=120 + 900 * ring, facecolors="none", edgecolors=ORANGE,
                    linewidths=2.4, alpha=1 - 0.6 * ring, zorder=5)
    n_done = len(landed)
    shots_n = int(landed["has_shot"].sum())
    xg_sum = float(landed["xg"].sum())
    mnt = int(cur.minute)
    info_t.set_text(f"Corner {min(n_done + (0 if (final or f >= FLIGHT) else 1), n_rep)} of {n_rep}:  "
                    f"vs {cur.opp}, {mnt}'" if not final else f"{n_rep} corners")
    sum_t.set_text(f"{n_done} corners   |   {shots_n} with a shot   |   {xg_sum:.2f} xG")
    if not final and f >= FLIGHT:
        res = f"shot, {cur.xg:.2f} xG" if cur.has_shot else "no shot"
        last_t.set_text(f"Last corner: {cur.dkind.lower()}, landed in zone '{cur.zone}', {res}")
        last_t.set_color(ORANGE if cur.has_shot else GREEN)
    elif final:
        last_t.set_text(f"Tournament average: {all_sr:.1f}% of corners had a shot, {all_xg:.3f} xG per corner")
        last_t.set_color(ORANGE)
    else:
        last_t.set_text("")
    last2_t.set_text("" if not final else f"{rep_team}: {shots_n / max(n_done, 1) * 100:.1f}% with a shot, "
                     f"{xg_sum / max(n_done, 1):.3f} xG per corner")


n_frames = n_rep * FPC + 45
FPS = 20
RAW_VIDEO = "corner_replay_raw.mp4"
FINAL_VIDEO = "corner_replay.mp4"
anim = FuncAnimation(fig, draw, frames=n_frames, interval=1000 / FPS)
writer = FFMpegWriter(fps=FPS, codec="libx264", bitrate=4000, extra_args=["-pix_fmt", "yuv420p"])
print(f"Rendering {n_frames} frames ({n_frames / FPS:.0f} seconds), this takes a minute...")
anim.save(RAW_VIDEO, writer=writer, dpi=108, savefig_kwargs={"facecolor": BG})
print("Saved replay:", RAW_VIDEO)


# ---------- append the charts to the video as end cards ----------
def append_end_cards(video_in, video_out, cards, seconds, size=1080, fps=20, bg="0d1b2a"):
    cards = [c for c in cards if os.path.exists(c)]
    if not cards:
        print("No chart PNGs found, keeping the replay without end cards.")
        os.replace(video_in, video_out)
        return
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-y", "-i", video_in]
    for c in cards:
        cmd += ["-loop", "1", "-framerate", str(fps), "-t", str(seconds), "-i", c]
    norm = (f"scale={size}:{size}:force_original_aspect_ratio=decrease,"
            f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2:color=0x{bg},"
            f"setsar=1,fps={fps},format=yuv420p")
    n = len(cards) + 1
    parts_f = [f"[{i}:v]{norm}[v{i}]" for i in range(n)]
    labels = "".join(f"[v{i}]" for i in range(n))
    fc = ";".join(parts_f) + f";{labels}concat=n={n}:v=1:a=0[out]"
    cmd += ["-filter_complex", fc, "-map", "[out]", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-r", str(fps), "-movflags", "+faststart", video_out]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-1500:])
        print("End cards failed, keeping the replay without them.")
        os.replace(video_in, video_out)
        return
    os.remove(video_in)
    print("Added end cards:", ", ".join(cards))


append_end_cards(RAW_VIDEO, FINAL_VIDEO, END_CARDS, END_CARD_SECONDS, fps=FPS)
print("Saved video:", FINAL_VIDEO)