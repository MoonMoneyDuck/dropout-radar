"""📡 Dropout Radar · week 6 mission control for the study office.

Built on the hotel app (aaubs/tonights-front-desk). The model sits next to the code in model/: booster.json (XGBoost)
+ preprocess.json (filling gaps, scaling and one-hot as plain numbers), written by portable.export in the notebook.
The students are read straight from GitHub, like in the notebook. The jokes are on the office, never on the students.
"""
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from portable import Model

URL = "https://raw.githubusercontent.com/aaubs/ds-master/main/assignments/study-office/data/"
MODEL_DIR = Path(__file__).parent / "model"
GROUP = {0: "domestic", 1: "international"}
GIF = {k: f"https://media.giphy.com/media/{v}/giphy.gif" for k, v in {
    "welcome": "MCudzuADLuJWw", "thinking": "WRQBXSCnEFJIuxktnw", "fire": "Z1BTGhofioRxK",
    "nailed": "8VrtCswiLDNnO", "cheers": "DfLwM9kttDFEQ", "pikachu": "6nWhy3ulBL7GSCvKw6", "win": "3oEduKVQdG4c0JVPSo"}.items()}
LABEL = {"quiz_mean": "quiz average {v:.0f}", "logins_last3": "{v:.0f} logins in the last 3 weeks",
         "logins_total": "{v:.0f} logins in weeks 1–6", "logins_trend": "login trend {v:+.1f}",
         "submitted_share": "handed in {v:.0%} of assignments", "missed_last3": "missed {v:.0f} assignments lately",
         "weeks_since_login": "{v:.0f} weeks since last login", "admission_grade": "admission grade {v:.1f}",
         "age": "age {v:.0f}", "programme": "{v}", "gender": "{v}"}
YES_NO = {"fees_owed": ("no fees owed", "owes fees"), "su_scholarship": ("no SU", "has SU"),
          "international": ("domestic student", "international student"),
          "first_gen": ("not first in family", "first in family at university"),
          "moved_from_home": ("lives at home", "moved from home"), "married": ("not married", "married"),
          "evening_programme": ("day programme", "evening programme")}
HELP = {"quiz_mean": "📚 study help", "submitted_share": "📚 study help", "missed_last3": "📚 study help",
        "fees_owed": "💰 money talk", "su_scholarship": "💰 money talk", "logins_last3": "👋 just check in",
        "logins_total": "👋 just check in", "logins_trend": "👋 just check in", "weeks_since_login": "👋 just check in"}

st.set_page_config(page_title="Dropout Radar · week 6", page_icon="📡", layout="wide")


def html(markup):
    """Raw HTML/CSS through st.markdown (styles then apply to the whole page). Flattened: no newlines, no indents."""
    st.markdown(" ".join(line.strip() for line in markup.splitlines() if line.strip()), unsafe_allow_html=True)


# ---------------------------------------------------------------- the look: neon arcade, all CSS (no JavaScript needed)
html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bungee&family=Space+Grotesk:wght@400;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stMetric { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: radial-gradient(circle at 20% 0%, #2a1060 0%, #0b0b1e 45%), #0b0b1e; }
@property --n { syntax: '<integer>'; initial-value: 0; inherits: false; }

.hero { text-align:center; padding: 18px 0 6px; }
.hero h1 { font-family:'Bungee', cursive; font-size: clamp(2.2rem, 6vw, 4.4rem); margin:0; line-height:1.05;
  background: linear-gradient(90deg,#ff4fd8,#7b5cff,#36e4ff,#ff4fd8); background-size:300% 100%;
  -webkit-background-clip:text; background-clip:text; color:transparent; animation: shine 6s linear infinite; }
.hero p { color:#b9b6ff; font-size:1.1rem; margin:.4rem 0 0; }
.ticker { overflow:hidden; white-space:nowrap; border-top:1px solid #36e4ff44; border-bottom:1px solid #36e4ff44;
  margin: 10px 0 18px; padding:6px 0; color:#36e4ff; font-weight:600; letter-spacing:.04em; }
.ticker span { display:inline-block; padding-left:100%; animation: ticker 32s linear infinite; }
@keyframes shine { to { background-position: 300% 0; } }
@keyframes ticker { to { transform: translateX(-100%); } }

.card { background: linear-gradient(160deg,#1d1d4a,#14143a); border:1px solid #7b5cff55; border-radius:18px;
  padding:16px 18px; text-align:center; box-shadow: 0 0 22px #7b5cff22; transition: transform .25s, box-shadow .25s;
  animation: pop .6s cubic-bezier(.2,1.6,.4,1) both; }
.card:hover { transform: translateY(-6px) rotate(-1deg); box-shadow: 0 0 30px #ff4fd855; }
.card .emo { font-size:2.2rem; display:inline-block; animation: bob 2.2s ease-in-out infinite; }
.card .big { font-family:'Bungee', cursive; font-size:2.6rem; color:#fff; }
.card .lbl { color:#b9b6ff; font-size:.95rem; }
.count { animation: count 1.8s cubic-bezier(.2,.8,.2,1) forwards; counter-reset: n var(--n); }
.count::after { content: counter(n); }
@keyframes count { from { --n: 0; } to { --n: var(--to); } }
@keyframes pop { from { transform: scale(.3); opacity:0; } to { transform: scale(1); opacity:1; } }
@keyframes bob { 50% { transform: translateY(-6px) rotate(8deg); } }

.radar { position:relative; width:min(460px, 88vw); aspect-ratio:1; margin: 6px auto; border-radius:50%;
  background: repeating-radial-gradient(circle, transparent 0 18%, #36e4ff33 18% 18.4%),
              radial-gradient(circle, #0f2a3a 0%, #071420 70%);
  border:2px solid #36e4ff88; box-shadow: 0 0 40px #36e4ff44, inset 0 0 60px #36e4ff22; overflow:hidden; }
.radar::before { content:""; position:absolute; inset:0; border-radius:50%;
  background: conic-gradient(from 0deg, #36e4ff88, transparent 22%); animation: sweep 3.2s linear infinite; }
.radar::after { content:""; position:absolute; left:50%; top:0; bottom:0; width:1px; background:#36e4ff33;
  box-shadow: 0 0 0 0 transparent; }
.blip { position:absolute; width:12px; height:12px; margin:-6px; border-radius:50%; background:#ff4fd8;
  box-shadow:0 0 12px #ff4fd8; animation: ping 1.6s ease-out infinite; cursor:help; }
.blip.hot { background:#ffd23f; box-shadow:0 0 16px #ffd23f; width:16px; height:16px; margin:-8px; }
@keyframes sweep { to { transform: rotate(360deg); } }
@keyframes ping { 0% { box-shadow:0 0 0 0 #ff4fd8aa; } 100% { box-shadow:0 0 0 14px transparent; } }

.duel { display:flex; gap:14px; flex-wrap:wrap; }
.duel .card { flex:1 1 220px; }
.bar { height:26px; border-radius:13px; background:#25255a; overflow:hidden; margin:6px 0 14px; }
.bar div { height:100%; border-radius:13px; animation: grow 1.6s cubic-bezier(.2,.8,.2,1) both;
  background: linear-gradient(90deg,#7b5cff,#ff4fd8); }
.bar.cyan div { background: linear-gradient(90deg,#1a9bd8,#36e4ff); }
@keyframes grow { from { width:0; } }

.rain { position:relative; height:120px; overflow:hidden; border-radius:18px; margin-bottom:12px;
  background: linear-gradient(180deg,#17173d,#0b0b1e); border:1px dashed #ffd23f55; }
.rain i { position:absolute; top:-40px; font-style:normal; font-size:26px; pointer-events:none;
  animation: fall linear infinite; }
@keyframes fall { to { transform: translateY(170px) rotate(540deg); } }
.gunnar { font-size:3rem; display:inline-block; animation: wobble 1.2s ease-in-out infinite; }
@keyframes wobble { 25% { transform: rotate(-12deg); } 75% { transform: rotate(12deg); } }
.note { color:#b9b6ff; font-size:.9rem; }
</style>
""")


def card(emo, value, label, count=True, delay=0.0):
    num = f"{value:,}" if isinstance(value, (int, np.integer)) else value
    return (f'<div class="card" style="animation-delay:{delay}s"><div class="emo">{emo}</div>'
            f'<div class="big">{num}</div><div class="lbl">{label}</div></div>')


def cards(items):
    html('<div class="duel">' + "".join(card(*it, delay=i * 0.12) for i, it in enumerate(items)) + "</div>")


# ---------------------------------------------------------------- the model and the students, loaded once per server
@st.cache_resource
def load():
    model = Model(MODEL_DIR)                       # no pickle: loads with any recent pandas/xgboost
    history = pd.read_csv(URL + "history_week6.csv")
    new = pd.read_csv(URL + "new_week6.csv")
    last_year = history[history["cohort"] == 2025].copy()
    last_year["risk"] = model.predict_proba(last_year)
    new["risk"] = model.predict_proba(new)
    new = new.sort_values("risk", ascending=False).reset_index(drop=True)
    contrib = model.contributions(new).reset_index(drop=True)
    new["why"] = [reasons(new.iloc[i], contrib.iloc[i]) for i in range(len(new))]
    new["help"] = [HELP.get(contrib.iloc[i].idxmax(), "🤝 a friendly chat") for i in range(len(new))]
    return last_year, new


def reasons(row, contrib, top=2):
    """The features that push this student's risk up the most (XGBoost's own contributions, one-hot added back)."""
    best = contrib.sort_values(ascending=False).head(top)
    return " · ".join(YES_NO[f][int(row[f])] if f in YES_NO else LABEL.get(f, f + ": {v}").format(v=row[f])
                      for f, c in best.items() if c > 0)


def boxes(df, k):
    """The four boxes for 'talk to the k highest risks' (the list is drawn over the whole cohort)."""
    on_list = df["risk"].rank(ascending=False, method="first") <= k
    left = df["left"] == 1
    return {"reached": int((on_list & left).sum()), "worried": int((on_list & ~left).sum()),
            "missed": int((~on_list & left).sum()), "fine": int((~on_list & ~left).sum())}


def net_value(b, talk, worry, leave, helps):
    return b["reached"] * helps * leave - (b["reached"] + b["worried"]) * talk - b["worried"] * worry


def money_rain(symbols, n=36):
    rng = np.random.default_rng(1)
    drops = "".join(f'<i style="left:{rng.uniform(0, 97):.1f}%; animation-duration:{rng.uniform(2.2, 4.5):.2f}s; '
                    f'animation-delay:{rng.uniform(0, 1.5):.2f}s">{rng.choice(symbols)}</i>' for _ in range(n))
    html(f'<div class="rain">{drops}</div>')


last_year, new = load()
base_rate = last_year["left"].mean()

with st.sidebar:
    st.markdown("## 🕹️ Control panel")
    k = st.slider("💬 Conversations this week", 10, 150, 40, 1,
                  help="Three advisers can hold about 40. The office talks to the students with the highest risk.")
    advisers = k / 13.3
    st.markdown(f"That's about **{advisers:.1f} advisers** {'☕' * min(8, max(1, round(advisers)))}")
    st.markdown("## 💶 What things cost")
    talk = st.number_input("A conversation (DKK of adviser time)", 0, 5_000, 500, 100)
    worry = st.number_input("A student worried for nothing (DKK)", 0, 50_000, 2_000, 500,
                            help="An assumption, not a fact: who decides what a worried student 'costs'?")
    leave = st.number_input("A student who leaves (DKK for the university)", 0, 200_000, 60_000, 5_000)
    helps = st.slider("Share of at-risk students a conversation keeps", 0, 100, 30, 5, format="%d%%") / 100
    st.markdown("---")
    st.caption("🧑‍⚖️ The radar only *suggests*. **An adviser decides** who is contacted, students are told the "
               "system exists, and they can say no.")

html(f"""
<div class="hero"><h1>📡 DROPOUT RADAR</h1>
<p>Week 6 mission control · {len(new)} students on campus · {k} conversations in the tank</p></div>
<div class="ticker"><span>🛰️ RADAR ONLINE ·· ☕ ADVISERS FULLY CAFFEINATED ·· 🎩 GUT FEELING GUNNAR HAS ENTERED THE BUILDING ··
📉 ACCURACY HAS BEEN BANNED FROM THIS OFFICE ·· 🧑‍⚖️ HUMANS DECIDE, THE MODEL ONLY POINTS ·· 💌 EVERY STUDENT ON THE LIST GETS AN OFFER OF HELP, NOT A VERDICT ··</span></div>
""")

tab_radar, tab_replay, tab_fair, tab_money = st.tabs(
    ["📡 This week's radar", "🎬 Replay last year", "⚖️ Tug of fairness", "💸 Show me the money"])

# ---------------------------------------------------------------- 1 · this week's list, as a radar
with tab_radar:
    top = new.head(k)
    left_col, right_col = st.columns([1, 1.25], gap="large")
    with left_col:
        blips = []
        for _, r in top.iterrows():
            angle = (zlib.crc32(r["student_id"].encode()) % 3600) / 10 * np.pi / 180
            radius = 46 * (1 - (r["risk"] - top["risk"].min()) / max(1e-9, top["risk"].max() - top["risk"].min())) + 4
            x, y = 50 + radius * np.cos(angle), 50 + radius * np.sin(angle)
            hot = " hot" if r["risk"] >= 0.6 else ""
            blips.append(f'<div class="blip{hot}" style="left:{x:.1f}%; top:{y:.1f}%; animation-delay:{(angle % 1.6):.2f}s" '
                         f'title="{r["student_id"]} · {r["risk"]:.0%} · {r["why"]}"></div>')
        html(f'<div class="radar">{"".join(blips)}</div>'
                f'<p class="note" style="text-align:center">Each blip is a student on the list. Closer to the centre = '
                f'higher risk, 🟡 = above 60 %. Hover a blip to see who and why.</p>')
    with right_col:
        cards([("🎯", k, "students on the list"),
               ("🌡️", f"{top['risk'].min():.0%}", "lowest risk that made the cut", False),
               ("🌍", int(top["international"].sum()), "international students on it")])
        st.markdown("#### 🚨 Top 3 on the radar")
        for i, r in top.head(3).iterrows():
            st.markdown(f"**#{i + 1} · {r['student_id']}** · {r['programme']} · risk **{r['risk']:.0%}**  \n"
                        f"↳ {r['why']} → suggested: **{r['help']}**")
    st.markdown("#### 📝 The full list")
    only_list = st.toggle("Show only the students on the list", value=True)
    show = (top if only_list else new).copy()
    show.insert(0, "rank", range(1, len(show) + 1))
    show["on list"] = show["rank"] <= k
    show["group"] = show["international"].map(GROUP)
    show["risk"] = (100 * show["risk"]).round()
    st.dataframe(show[["rank", "on list", "student_id", "risk", "programme", "group", "why", "help"]],
                 column_config={"risk": st.column_config.ProgressColumn("risk", min_value=0, max_value=100, format="%d%%"),
                                "student_id": "student", "on list": st.column_config.CheckboxColumn("on list"),
                                "help": "suggested kind of help"},
                 hide_index=True, width="stretch", height=420)
    st.caption("On the 2025 students the model's risks ran about a quarter too high (8.4 % predicted, 6.9 % left): "
               "read them as a ranking more than as exact chances.")

# ---------------------------------------------------------------- 2 · the mistakes: replay last year
with tab_replay:
    st.markdown(f"### 🎬 What if we had used the radar **last year**?")
    st.markdown(f"In 2025, **{len(last_year)} students** were still enrolled at week 6 and **{int(last_year['left'].sum())}** "
                f"of them left later. We know who. So: what would **{k} conversations** with the highest risks have done?")
    if "replayed" not in st.session_state:
        st.session_state.replayed = False
    if not st.session_state.replayed:
        c1, c2 = st.columns([2, 1])
        if c1.button("▶️ ROLL THE TAPE", type="primary", width="stretch"):
            st.session_state.replayed = True
            st.balloons()
            st.rerun()
        c2.image(GIF["thinking"], caption="the advisers, waiting (via GIPHY)")
    else:
        b = boxes(last_year, k)
        n_left = b["reached"] + b["missed"]
        precision, recall = b["reached"] / k, b["reached"] / n_left
        cards([("🎯", b["reached"], "reached in time"), ("📞", b["worried"], "worried for nothing"),
               ("🚪", b["missed"], "missed"), ("😌", b["fine"], "fine, left alone")])
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(f"""
- **Precision {precision:.0%}** 🎯 of the {k} students the office talks to, **{b['reached']}** were really about to leave.
- **Recall {recall:.0%}** 🕸️ of the {n_left} students who left, the office reached **{b['reached']}**.
""")
        c2.image(GIF["nailed"] if recall >= 0.4 else GIF["cheers"] if recall >= 0.2 else GIF["pikachu"], caption="via GIPHY")

        st.markdown("### 🥊 The challengers")
        gunnar = k * base_rate
        html(f"""<div class="duel">
          {card("📡", b['reached'], f"the radar reaches<br><b>{b['reached']}</b> of {n_left}")}
          <div class="card" style="animation-delay:.15s"><div class="gunnar">🎩</div><div class="big">≈{gunnar:.0f}</div>
            <div class="lbl"><b>Gut Feeling Gunnar</b> picks {k} names out of a hat.<br>He reaches about {gunnar:.0f}.</div></div>
          <div class="card" style="animation-delay:.3s"><div class="emo">🏆</div><div class="big">{1 - base_rate:.0%}</div>
            <div class="lbl"><b>Doing nothing</b> is {1 - base_rate:.0%} "accurate"<br>and reaches <b>zero</b> students.
            That's why accuracy is banned here.</div></div></div>""")

        st.markdown("### 📈 More conversations: more reached, more worried")
        curve = pd.DataFrame([{"conversations": j, **boxes(last_year, j)} for j in range(5, 301, 5)]).set_index("conversations")
        st.line_chart(curve[["reached", "worried"]].rename(columns={"reached": "🎯 reached in time", "worried": "📞 worried for nothing"}),
                      x_label="conversations", y_label="students", color=["#36e4ff", "#ff4fd8"])
        if st.button("⏪ Rewind"):
            st.session_state.replayed = False
            st.rerun()

# ---------------------------------------------------------------- 3 · per group: a tug of war
with tab_fair:
    st.markdown(f"### ⚖️ Does the radar see **everyone** equally well? ({k} conversations, 2025 students)")
    on_list_idx = last_year.nlargest(k, "risk").index
    rows = []
    for g, part in last_year.groupby("international"):
        on_list = part.index.isin(on_list_idx)
        left = part["left"] == 1
        reached, missed = int((on_list & left).sum()), int((~on_list & left).sum())
        rows.append({"group": GROUP[g], "students": len(part), "left": int(left.sum()),
                     "share who left": left.mean(), "average predicted risk": part["risk"].mean(),
                     "reached in time": reached, "worried for nothing": int((on_list & ~left).sum()), "missed": missed,
                     "recall": reached / max(1, reached + missed)})
    groups = pd.DataFrame(rows).set_index("group")
    c1, c2 = st.columns(2, gap="large")
    for col, (g, r), style, emo in zip([c1, c2], groups.iterrows(), ["", " cyan"], ["🏠", "🌍"]):
        with col:
            html(f"""<div class="card"><div class="emo">{emo}</div><div class="big">{r['recall']:.0%}</div>
              <div class="lbl"><b>{g}</b> students who left and were reached<br>({int(r['reached in time'])} of {int(r['left'])})</div></div>
              <p class="note">really left</p><div class="bar{style}"><div style="width:{100 * r['share who left'] / 0.15:.0f}%"></div></div>
              <p class="note">the radar's average risk</p><div class="bar{style}"><div style="width:{100 * r['average predicted risk'] / 0.15:.0f}%"></div></div>
              <p class="note">reached (recall)</p><div class="bar{style}"><div style="width:{100 * r['recall']:.0f}%"></div></div>""")
    st.dataframe(groups.style.format({"share who left": "{:.1%}", "average predicted risk": "{:.1%}", "recall": "{:.0%}"}),
                 width="stretch")
    st.info("🧐 Both groups leave about equally often, but the radar scores domestic students higher, so they grab more "
            "of the places on the list. International students log in much less in general: a quiet student is normal "
            "there, not a warning sign. Check this per group every year, and let **people** decide which kind of fairness "
            "the office wants: equal recall, or risks that mean the same in both groups. Few international students "
            "leave each year, so these numbers wobble a lot.")

# ---------------------------------------------------------------- 4 · one thing of our own: is it worth it?
with tab_money:
    st.markdown("### 💸 Does the rule pay for itself?")
    st.markdown("Twist the cost knobs in the control panel. The numbers are **assumptions**: the office should choose "
                "them, ideally together with students.")
    b = boxes(last_year, k)
    value_k = net_value(b, talk, worry, leave, helps)
    js = range(5, 301, 1)
    values = [net_value(boxes(last_year, j), talk, worry, leave, helps) for j in js]
    j_best = list(js)[int(np.argmax(values))]
    denom = helps * leave + worry
    break_even = (talk + worry) / denom if denom else 1.0
    above = int((new["risk"] >= break_even).sum())
    money_rain(["💸", "💰", "🪙", "💶"] if value_k > 0 else ["🔥", "📉", "😬"])
    html('<div class="duel">'
            + card("💰" if value_k > 0 else "🔥", f"{value_k:,.0f} DKK", f"value of {k} conversations (2025)", count=False)
            + card("🧮", j_best, "best number of conversations (2025)", delay=0.12)
            + card("⚖️", f"{break_even:.1%}", "a conversation pays off above this risk", count=False, delay=0.24)
            + "</div>")
    if j_best > k + 10:
        st.success(f"🧑‍💼 The costs say the office should hold about **{j_best}** conversations, "
                   f"**{j_best - k} more** than now: roughly **{(j_best - k) / 13.3:.1f} extra advisers** would pay for themselves.")
    elif j_best < k - 10:
        st.warning(f"🧘 With these costs, fewer conversations (about **{j_best}**) would be worth more than {k}.")
    else:
        st.info(f"👌 {k} conversations is close to the best number with these costs.")
    st.line_chart(pd.DataFrame({"net value (DKK)": values}, index=pd.Index(list(js), name="conversations")),
                  x_label="conversations", y_label="net value, DKK", color=["#ffd23f"])
    st.markdown(f"""
- **Net value** = students kept × value of a student − all conversations − students worried for nothing.
- With these costs, a conversation pays off for any student with a risk above **{break_even:.1%}**:
  this week that is **{above} students**, against room for **{k}**.
""")
