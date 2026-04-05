"""
app.py – Interface Streamlit "Next-Gen" de CareerBridge AI – Maroc Edition.
Lancer avec : streamlit run app.py
═══════════════════════════════════════════════════════════════════════════════
Features :
  • Système d'onglets : Orientation | Analyse Marché | Feuille de Route
  • Sidebar dynamique avec logs temps réel
  • Diagramme de Sankey (flux de carrière) via Plotly
  • Analyse multimodale : bulletin image + audio
  • Messages bilingues Français / Darija
  • Interface 100% réactive aux résultats des agents
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import time
import json
import logging
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# ─── Configuration Streamlit ──────────────────────────────────────────────────

st.set_page_config(
    page_title="CareerBridge AI – Maroc",
    page_icon="🇲🇦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS Next-Gen ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .hero-title {
    font-size: 3rem; font-weight: 800;
    background: linear-gradient(135deg, #c0392b 0%, #e74c3c 30%, #2ecc71 70%, #27ae60 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; margin-bottom: 0.3rem;
  }

  .hero-subtitle {
    text-align: center; color: #666; font-size: 1.1rem; margin-bottom: 0.5rem;
  }

  .flag-bar { text-align: center; font-size: 1.8rem; margin-bottom: 1.5rem; }

  .log-container {
    background: #0d1117; color: #58a6ff;
    font-family: 'Courier New', monospace; font-size: 0.78rem;
    border-radius: 8px; padding: 1rem; height: 200px;
    overflow-y: auto; border: 1px solid #30363d;
  }

  .log-entry { margin-bottom: 3px; }
  .log-success { color: #3fb950; }
  .log-warning { color: #d29922; }
  .log-error { color: #f85149; }
  .log-info { color: #58a6ff; }

  .domain-card {
    border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem;
    border: 1px solid #e0e0e0; background: #fafafa;
    transition: all 0.3s ease; cursor: pointer;
  }

  .domain-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    border-color: #3498db;
  }

  .domain-card-gold { border-left: 5px solid #f39c12; background: linear-gradient(to right, #f39c1211, #ffffff); }
  .domain-card-silver { border-left: 5px solid #95a5a6; background: linear-gradient(to right, #95a5a611, #ffffff); }
  .domain-card-bronze { border-left: 5px solid #e67e22; background: linear-gradient(to right, #e67e2211, #ffffff); }
  .domain-card-rejected { border-left: 5px solid #e74c3c; background: linear-gradient(to right, #e74c3c11, #ffffff); }

  .score-badge {
    display: inline-block; padding: 0.4rem 0.9rem; border-radius: 20px;
    font-weight: 700; font-size: 0.9rem; color: white;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
  }

  .badge-green { background: linear-gradient(135deg, #27ae60, #2ecc71); }
  .badge-orange { background: linear-gradient(135deg, #e67e22, #f39c12); }
  .badge-red { background: linear-gradient(135deg, #c0392b, #e74c3c); }

  .darija-box {
    background: linear-gradient(135deg, #c0392b11, #27ae6011);
    border: 2px solid #c0392b33; border-radius: 12px;
    padding: 1.5rem; margin: 1rem 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
  }

  .darija-text { font-size: 1.1rem; color: #2c3e50; line-height: 1.8; font-weight: 500; }

  .milestone-card {
    background: white; border-radius: 10px; padding: 1rem;
    border-left: 4px solid #3498db; margin-bottom: 0.8rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    transition: all 0.2s ease;
  }

  .milestone-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.1); }

  .salary-highlight {
    background: #e8f5e9; color: #1b5e20;
    border-radius: 20px; padding: 0.3rem 0.9rem;
    font-weight: 600; display: inline-block;
    box-shadow: 0 2px 4px rgba(27, 94, 32, 0.2);
  }

  .rejected-badge {
    background: #fdecea; color: #c0392b;
    border: 2px solid #e74c3c33; border-radius: 8px;
    padding: 0.6rem 1.2rem; font-size: 0.85rem;
    font-weight: 500;
  }

  .tab-header { font-size: 1.3rem; font-weight: 700; margin-bottom: 1.5rem; color: #2c3e50; }

  .stProgress > div > div > div {
    background: linear-gradient(90deg, #c0392b, #e74c3c, #2ecc71);
  }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────

st.markdown('<h1 class="hero-title">🎓 CareerBridge AI</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Plateforme d\'orientation académique & professionnelle pour le Maroc</p>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="flag-bar">🇲🇦 &nbsp; Multi-Agents IA &nbsp;|&nbsp; RAG ChromaDB &nbsp;|&nbsp; Darija &nbsp; 🇲🇦</div>',
    unsafe_allow_html=True
)

# ─── Sidebar CONFIGURATION ─────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    api_key = st.text_input(
        "🔑 Clé API OpenAI",
        type="password",
        placeholder="sk-...",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Optionnel - Ollama local en priorité",
        key="sidebar_openai_key"
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    tavily_key = st.text_input(
        "🌐 Clé Tavily (optionnel)",
        type="password",
        placeholder="tvly-...",
        key="sidebar_tavily_key"
    )
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key

    st.divider()
    st.markdown("### 🤖 Agents Actifs")
    agents_status = [
        ("🔬 Student Profiler", True),
        ("🏭 Industry Matcher", True),
        ("🔄 Plan B Architect", True),
        ("🎨 Career Visualizer", True),
        ("🇲🇦 Darija Translator", True),
    ]
    for name, active in agents_status:
        st.success(f"✅ {name}") if active else st.error(f"❌ {name}")

    st.divider()
    st.markdown("### 📡 Système RAG")
    chroma_path = Path("data/chroma_db/chroma.sqlite3")
    if chroma_path.exists():
        st.success("✅ ChromaDB indexé")
    else:
        st.info("🔄 Sera créé au 1er lancement")

    pdf_count = len(list(Path("data/pdfs").glob("*.pdf"))) if Path("data/pdfs").exists() else 0
    st.metric("PDFs indexés", pdf_count)

    st.divider()
    st.markdown("### 🌍 Maroc")
    st.info("8 domaines | 30+ écoles | Salaires MAD | Darija")

    st.caption("CareerBridge AI v2.0 – 2024")

# ─── Session State ─────────────────────────────────────────────────────────────

if "log_messages" not in st.session_state:
    st.session_state.log_messages = []
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "is_running" not in st.session_state:
    st.session_state.is_running = False


def add_log(msg: str, level: str = "info"):
    """Ajoute un message de log avec timestamp"""
    timestamp = time.strftime("%H:%M:%S")
    emoji = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}.get(level, "•")
    st.session_state.log_messages.append(f"[{timestamp}] {emoji} {msg}")


def render_student_profile(profile):
    """Affiche le profil étudiant de manière dynamique"""
    if not profile:
        st.warning("❌ Profil étudiant non disponible")
        return

    col_p1, col_p2 = st.columns([2, 1])

    with col_p1:
        st.markdown("### 👤 Profil Étudiant")
        summary = getattr(profile, 'student_summary', '')
        st.info(summary if summary else "Aucun résumé disponible")

        top_skills = getattr(profile, 'top_skills', [])
        if top_skills:
            st.markdown("**🛠️ Compétences identifiées :**")
            skill_cols = st.columns(min(len(top_skills), 5))
            for i, skill in enumerate(top_skills[:5]):
                with skill_cols[i % 5]:
                    st.markdown(f"""
                    <div style='background:#e3f2fd;color:#1565c0;padding:0.4rem 0.8rem;
                                border-radius:15px;text-align:center;font-weight:500;'>{skill}</div>
                    """, unsafe_allow_html=True)

        col_str, col_gap = st.columns(2)
        with col_str:
            strengths = getattr(profile, 'profile_strengths', [])
            if strengths:
                st.markdown("**✅ Points forts :**")
                for s in strengths[:4]:
                    st.markdown(f"  • {s}")
        with col_gap:
            gaps = getattr(profile, 'profile_gaps', [])
            if gaps:
                st.markdown("**🔧 À développer :**")
                for g in gaps[:4]:
                    st.markdown(f"  • {g}")

    with col_p2:
        motivation = getattr(profile, 'motivation_level', 7)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=motivation,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Motivation", 'font': {'size': 16}},
            gauge={
                'axis': {'range': [0, 10]},
                'bar': {'color': "#27ae60"},
                'steps': [
                    {'range': [0, 4], 'color': "#fdecea"},
                    {'range': [4, 7], 'color': "#fff8e1"},
                    {'range': [7, 10], 'color': "#e8f5e9"},
                ],
            }
        ))
        fig_gauge.update_layout(height=250, margin=dict(t=30, b=10, l=10, r=10))
        st.plotly_chart(fig_gauge, use_container_width=True, key="gauge_motivation")

        profile_type = getattr(profile, 'profile_type', 'mixte')
        type_emojis = {'scientifique': '🔬', 'tech': '💻', 'humaniste': '📚', 'mixte': '🎭', 'artistique': '🎨'}
        st.markdown(f"**Profil :** {type_emojis.get(profile_type, '🎭')} {profile_type.upper()}")


def render_top_domains(match, ranked):
    """Affiche les domaines recommandés"""
    st.markdown("### 🏆 Top 3 Domaines Recommandés")

    if not ranked or len(ranked) == 0:
        st.warning("❌ Aucun domaine recommandé")
        return

    medals = ["🥇", "🥈", "🥉"]
    colors = ["gold", "silver", "bronze"]

    for i, dom in enumerate(ranked[:3]):
        if i < len(medals):
            score = dom.get('combined_score', 0)
            label = dom.get('label', 'Domaine inconnu')

            badge_class = "badge-green" if score >= 7 else "badge-orange" if score >= 5 else "badge-red"

            sal = dom.get('salary_mad', {})
            sal_junior = sal.get('junior', 'N/A') if isinstance(sal, dict) else 'N/A'
            sal_senior = sal.get('senior', 'N/A') if isinstance(sal, dict) else 'N/A'

            st.markdown(f"""
            <div class="domain-card domain-card-{colors[i]}">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem">
                <h4 style="margin:0">{medals[i]} {label}</h4>
                <span class="score-badge {badge_class}">{score}/10</span>
              </div>
              <p style="color:#666;font-size:0.9rem;margin-bottom:0.5rem">
                💰 Junior: <strong>{sal_junior}</strong> MAD<br/>
                💼 Senior: <strong>{sal_senior}</strong> MAD
              </p>
            </div>
            """, unsafe_allow_html=True)


def render_market_analysis(ranked):
    """Affiche l'analyse du marché"""
    st.markdown('<div class="tab-header">📊 Analyse du Marché Marocain</div>', unsafe_allow_html=True)

    if not ranked or len(ranked) == 0:
        st.warning("❌ Aucunes données disponibles")
        return

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        labels = [d["label"][:20] for d in ranked[:8]]
        scores = [d["combined_score"] for d in ranked[:8]]
        sat_colors = [
            "#e74c3c" if d["saturation_index"] >= 7
            else "#e67e22" if d["saturation_index"] >= 5
            else "#27ae60"
            for d in ranked[:8]
        ]

        fig_bar = go.Figure(go.Bar(
            x=scores, y=labels, orientation="h",
            marker_color=sat_colors,
            text=[f"{s}/10" for s in scores],
            textposition="outside",
        ))
        fig_bar.update_layout(
            title="Scores de Compatibilité",
            xaxis=dict(range=[0, 10], title="Score /10"),
            height=400,
            plot_bgcolor="white",
        )
        st.plotly_chart(fig_bar, use_container_width=True, key="bar_market")

    with col_c2:
        top3_domains = ranked[:3]
        categories = ["Score Combiné", "Score Profil", "Score Marché"]

        fig_radar = go.Figure()
        colors_radar = ["#c0392b", "#2ecc71", "#3498db"]

        for i, d in enumerate(top3_domains):
            values = [
                d.get("combined_score", 0),
                d.get("profile_score", 0),
                d.get("market_score", 0),
            ]
            label = d.get("label", "Domaine")
            fig_radar.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself", name=label[:20],
                line_color=colors_radar[i],
                opacity=0.7,
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(range=[0, 10])),
            title="Radar – Top 3",
            height=400,
        )
        st.plotly_chart(fig_radar, use_container_width=True, key="radar_market")

    st.markdown("### 📋 Tableau Complet")
    table_data = []
    for d in ranked:
        sat_label = "🔴 Saturé" if d["saturation_index"] >= 7 else "🟡 Modéré" if d["saturation_index"] >= 5 else "🟢 Porteur"
        sal = d.get("salary_mad", {})
        table_data.append({
            "Domaine": d["label"],
            "Score": f"{d['combined_score']}/10",
            "Marché": f"{d['market_score']}/10",
            "Saturation": sat_label,
            "Jr": sal.get("junior", "N/A") if isinstance(sal, dict) else "N/A",
        })

    if table_data:
        st.dataframe(table_data, use_container_width=True, hide_index=True)


# ─── Formulaire d'entrée ──────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## 📋 Profil de l'Étudiant")

col_left, col_right = st.columns(2)

with col_left:
    subjects = st.text_area(
        "📖 Matières préférées",
        placeholder="Ex: mathématiques, informatique, physique...",
        height=90,
        key="form_subjects"
    )
    skills = st.text_area(
        "🛠️ Compétences",
        placeholder="Ex: python, programmation, communication...",
        height=90,
        key="form_skills"
    )
    career_goal = st.text_input(
        "🎯 Objectif de carrière",
        placeholder="Ex: Ingénieur IA, Data Scientist...",
        key="form_goal"
    )

with col_right:
    interests = st.text_area(
        "💡 Centres d'intérêt",
        placeholder="Ex: IA, énergie solaire, agriculture...",
        height=90,
        key="form_interests"
    )

    bulletin_file = st.file_uploader(
        "📄 Bulletin scolaire",
        type=["png", "jpg", "jpeg", "pdf"],
        key="form_bulletin"
    )

    audio_file = st.file_uploader(
        "🎙️ Message audio",
        type=["wav", "mp3", "m4a", "ogg"],
        key="form_audio"
    )

# ─── Traitement Multimodal ────────────────────────────────────────────────────

bulletin_analysis = None
audio_transcript = None

if bulletin_file is not None:
    with st.spinner("🔍 Analyse du bulletin..."):
        try:
            from utils.multimodal import get_bulletin_analyzer
            use_vision = bool(os.getenv("OPENAI_API_KEY"))
            analyzer = get_bulletin_analyzer(use_openai_vision=use_vision)
            bulletin_analysis = analyzer.analyze_bulletin_bytes(bulletin_file.read())

            col_bul1, col_bul2, col_bul3 = st.columns(3)
            with col_bul1:
                avg = bulletin_analysis.get("averages", {}).get("general", "N/A")
                st.metric("📊 Moyenne", f"{avg}/20" if avg != "N/A" else "N/A")
            with col_bul2:
                strong = bulletin_analysis.get("strong_subjects", [])
                st.metric("💪 Matières fortes", len(strong))
            with col_bul3:
                tags = bulletin_analysis.get("profile_tags", [])
                st.metric("🏷️ Profil", tags[0] if tags else "N/A")

            st.success("✅ Bulletin analysé")
        except Exception as e:
            st.warning(f"⚠️ Erreur: {e}")

if audio_file is not None:
    with st.spinner("🎙️ Transcription..."):
        try:
            import tempfile
            from utils.multimodal import get_stt_processor
            suffix = "." + audio_file.name.split(".")[-1]
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name
            stt = get_stt_processor()
            result = stt.transcribe_audio_file(tmp_path, language="fr")
            audio_transcript = result.get("text", "")
            if audio_transcript:
                st.success(f"🎙️ Transcription: *\"{audio_transcript}\"*")
            os.unlink(tmp_path)
        except Exception as e:
            st.warning(f"⚠️ Erreur: {e}")

# ─── Bouton principal ─────────────────────────────────────────────────────────

st.markdown("---")
col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
with col_b2:
    analyze_btn = st.button(
        "🚀 Lancer l'Analyse Multi-Agents",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.is_running,
        key="btn_analyze"
    )

# ─── Lancement de l'analyse ───────────────────────────────────────────────────

if analyze_btn:
    if not subjects and not skills and not interests:
        st.error("❌ Remplissez au moins un champ")
        st.stop()

    st.session_state.log_messages = []
    st.session_state.is_running = True
    st.session_state.analysis_result = None

    student_data = {
        "subjects": subjects,
        "skills": skills,
        "interests": interests,
        "career_goal": career_goal,
        "bulletin_analysis": bulletin_analysis,
        "audio_transcript": audio_transcript,
    }

    st.sidebar.markdown("### 📟 Logs Temps Réel")
    log_placeholder = st.sidebar.empty()

    def update_logs(msg: str):
        add_log(msg)
        log_html = "<div class='log-container'>"
        for entry in st.session_state.log_messages[-15:]:
            css_class = "log-success" if "✅" in entry else "log-error" if "❌" in entry else "log-warning" if "⚠️" in entry else "log-info"
            log_html += f"<div class='log-entry {css_class}'>{entry}</div>"
        log_html += "</div>"
        log_placeholder.markdown(log_html, unsafe_allow_html=True)

    progress = st.progress(0, text="Initialisation...")

    with st.spinner("🧠 Analyse en cours..."):
        try:
            from crew_logic import run_careerbridge_crew

            update_logs("🚀 Démarrage...")
            progress.progress(10, "Analyse du profil...")

            result = run_careerbridge_crew(
                student_data=student_data,
                log_callback=update_logs,
            )

            progress.progress(100, "✅ Terminée!")
            st.session_state.analysis_result = result
            st.session_state.is_running = False

            mode_msg = "🤖 IA Complète" if result.get("mode") == "full_ai" else "📊 Analytique"
            st.toast(f"✅ Analyse terminée! {mode_msg}", icon="🎉")

        except Exception as e:
            st.error(f"❌ Erreur: {e}")
            update_logs(f"❌ Erreur: {e}")
            st.session_state.is_running = False

# ─── AFFICHAGE DYNAMIQUE DES RÉSULTATS ────────────────────────────────────────

if st.session_state.analysis_result:
    result = st.session_state.analysis_result
    profile = result.get("profile")
    match = result.get("match")
    plan_b = result.get("plan_b")
    vision = result.get("vision")
    darija = result.get("darija")
    ranked = result.get("ranked_domains", [])
    mode = result.get("mode", "analytical")

    mode_color = "🟢" if mode == "full_ai" else "🟡"
    elapsed = result.get('elapsed_seconds', 'N/A')
    st.markdown(f"**{mode_color} Mode : {'IA Complète' if mode == 'full_ai' else 'Analytique'}** | ⏱️ {elapsed}s")

    st.markdown("---")

    # KPIs globaux DYNAMIQUES
    if ranked and len(ranked) > 0:
        top = ranked[0]
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            lbl = top['label'][:18] + "..." if len(top['label']) > 18 else top['label']
            st.metric("🥇 Top", lbl)
        with k2:
            st.metric("🎯 Score", f"{top['combined_score']:.1f}/10")
        with k3:
            st.metric("📈 Marché", f"{top['market_score']:.1f}/10")
        with k4:
            sal = top.get('salary_mad', {})
            sal_jr = sal.get('junior', 'N/A') if isinstance(sal, dict) else 'N/A'
            st.metric("💰 Jr", str(sal_jr)[:12])
        with k5:
            sat = top.get('saturation_index', 5)
            color = "🔴" if sat >= 7 else "🟡" if sat >= 5 else "🟢"
            st.metric(color, f"{sat}/10")

    st.markdown("---")

    # ONGLETS PRINCIPAUX
    tab1, tab2, tab3 = st.tabs([
        "🎓 Orientation",
        "📊 Marché",
        "🗺️ Feuille de Route",
    ])

    with tab1:
        st.markdown('<div class="tab-header">🎓 Profil & Recommandations</div>', unsafe_allow_html=True)
        render_student_profile(profile)
        st.markdown("---")
        render_top_domains(match, ranked)

    with tab2:
        render_market_analysis(ranked)

    with tab3:
        st.markdown('<div class="tab-header">🗺️ Feuille de Route</div>', unsafe_allow_html=True)

        if vision and getattr(vision, 'vision_narrative', ''):
            st.markdown("### ✨ Votre Vision")
            st.write(f"*{getattr(vision, 'vision_narrative', '')}*")

        if plan_b:
            st.markdown("### 🔄 Plan B")
            col_pb1, col_pb2 = st.columns(2)
            with col_pb1:
                certs = getattr(plan_b, 'online_certifications', [])
                if certs:
                    st.markdown("**🎓 Certifications :**")
                    for c in certs[:4]:
                        st.markdown(f"  • {c}")
            with col_pb2:
                bootcamps = getattr(plan_b, 'bootcamps_morocco', [])
                if bootcamps:
                    st.markdown("**💻 Bootcamps :**")
                    for b in bootcamps[:4]:
                        st.markdown(f"  • {b}")

        if darija:
            st.markdown("---")
            st.markdown("### 🇲🇦 Darija")
            greeting = getattr(darija, 'greeting_darija', '')
            if greeting:
                st.markdown(f"""
                <div class="darija-box">
                  <div class="darija-text">{greeting}</div>
                </div>
                """, unsafe_allow_html=True)

    if mode == "full_ai":
        st.balloons()

else:
    # Accueil
    st.markdown("---")
    col_feat1, col_feat2, col_feat3, col_feat4 = st.columns(4)

    features = [
        ("🤖", "5 Agents IA", "Multi-agents orchestration"),
        ("📚", "RAG ChromaDB", "Recherche sémantique"),
        ("🔍", "Vision IA", "Analyse bulletins"),
        ("🇲🇦", "Maroc First", "Contexte local"),
    ]

    for col, (icon, title, desc) in zip([col_feat1, col_feat2, col_feat3, col_feat4], features):
        with col:
            st.markdown(f"""
            <div style="text-align:center;padding:1.5rem;border:1px solid #e0e0e0;
                        border-radius:12px;height:160px;background:linear-gradient(135deg, #f5f7fa, #ecf0f1);">
              <div style="font-size:2.5rem">{icon}</div>
              <div style="font-weight:700;margin:0.5rem 0">{title}</div>
              <div style="font-size:0.85rem;color:#666">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#888;font-size:0.9rem">
    💡 <em>Remplissez le formulaire et cliquez "Lancer l'Analyse"</em><br>
    🎯 <em>Système utilise Ollama local + fallback OpenAI</em>
    </div>
    """, unsafe_allow_html=True)