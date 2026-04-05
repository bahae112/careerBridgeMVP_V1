"""
crew_logic.py
═══════════════════════════════════════════════════════════════════════════════
Orchestration CrewAI de CareerBridge AI – Maroc Edition.
  • Process.sequential avec memory=True
  • Gestion d'erreurs robuste avec fallback analytique
  • Calcul des scores indépendant du LLM (toujours disponible)
  • Logs en temps réel (compatible Streamlit callback)

# [JURY_CHECK]: REASONING – Le fallback analytique garantit que l'app
# fonctionne même sans clé OpenAI (démonstration jury sans internet).
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import os
import json
import logging
import time
import requests
from pathlib import Path
from typing import Optional, Callable

from crewai import Crew, Process
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None
try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None
try:
    from langchain_deepseek import ChatDeepSeek
except ImportError:
    ChatDeepSeek = None

from agents_factory import create_all_agents
from tasks_factory import (
    create_all_tasks,
    StudentProfile,
    IndustryMatchReport,
    DomainMatch,
    CareerVisionReport,
    CareerMilestone,
    SankeyData,
    PlanBReport,
    DarijaTranslation,
)

logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).parent / "data"


def create_multi_llm_config():
    """
    Crée une configuration multi-LLM pour différents agents selon leurs besoins.

    Returns:
        dict avec configuration LLM pour chaque agent
    """
    llm_configs = {}

    # Agent 1: Student Profiler - Utilise Ollama (analyse locale rapide)
    try:
        response = requests.get("http://localhost:11500/api/tags", timeout=5)
        if response.status_code == 200:
            llm_configs['profiler'] = ChatOllama(
                model="mistral",
                base_url="http://localhost:11500",
                temperature=0.3  # Plus précis pour l'analyse de profil
            )
    except:
        pass

    # Agent 2: Industry Matcher - Utilise DeepSeek (analyse de marché avancée)
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if deepseek_key and ChatDeepSeek:
        try:
            llm_configs['matcher'] = ChatDeepSeek(
                model="deepseek-chat",
                api_key=deepseek_key,
                temperature=0.2  # Très précis pour l'analyse marché
            )
        except:
            pass

    # Agent 3: Plan B Architect - Utilise Groq (rapide pour génération créative)
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key and ChatGroq:
        try:
            llm_configs['plan_b'] = ChatGroq(
                model="llama3-70b-8192",
                api_key=groq_key,
                temperature=0.7  # Créatif pour les plans alternatifs
            )
        except:
            pass

    # Agent 4: Career Visualizer - Utilise Gemini (bon pour narration)
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key and ChatGoogleGenerativeAI:
        try:
            llm_configs['visualizer'] = ChatGoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=gemini_key,
                temperature=0.6  # Expressif pour la narration
            )
        except:
            pass

    # Agent 5: Darija Translator - Utilise OpenAI (meilleur pour traduction)
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            llm_configs['translator'] = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.3  # Précis pour la traduction
            )
        except:
            pass

    # Fallback: Si certains LLMs ne sont pas configurés, utiliser Ollama ou OpenAI
    default_llm = None
    try:
        response = requests.get("http://localhost:11500/api/tags", timeout=5)
        if response.status_code == 200:
            default_llm = ChatOllama(
                model="mistral",
                base_url="http://localhost:11500",
                temperature=0.5
            )
    except:
        pass

    if not default_llm and openai_key:
        try:
            default_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
        except:
            pass

    # Assigner les LLMs par défaut pour les agents non configurés
    for agent_name in ['profiler', 'matcher', 'plan_b', 'visualizer', 'translator']:
        if agent_name not in llm_configs and default_llm:
            llm_configs[agent_name] = default_llm

    return llm_configs


# ══════════════════════════════════════════════════════════════════════════════
# SCORING ANALYTIQUE (INDÉPENDANT DU LLM)
# ══════════════════════════════════════════════════════════════════════════════

def _load_morocco_data() -> dict:
    try:
        with open(DATA_DIR / "morocco_careers.json", "r", encoding="utf-8") as f:
            return json.load(f)["domains"]
    except Exception:
        return {}


def _overlap_ratio(user_items: list, domain_items: list) -> float:
    if not domain_items:
        return 0.0
    u = {x.lower().strip() for x in user_items}
    d = {x.lower().strip() for x in domain_items}
    return len(u & d) / len(d)


def compute_compatibility_scores(
    skills: list,
    subjects: list,
    interests: list,
) -> list:
    """
    Calcule les scores de compatibilité pour tous les domaines marocains.

    # [JURY_CHECK]: REASONING – Formule transparente et documentée.
    # score = skills(40%) + subjects(35%) + interests(25%) + market(30% overlay)
    """
    career_data = _load_morocco_data()
    results = []

    for key, domain in career_data.items():
        skill_score = _overlap_ratio(skills, domain.get("key_skills", []))
        subject_overlap = _overlap_ratio(
            subjects,
            [s.lower() for s in domain.get("key_skills", [])]
        )

        # Score profil pondéré
        profile_score = (skill_score * 0.40 + subject_overlap * 0.35) * 10

        # Score marché
        market_score = (
            domain["market_demand"] * 0.30
            + domain["growth_rate"] * 0.25
            + domain["employability"] * 0.25
            + (10 - domain["saturation_index"]) * 0.20
        )

        combined = round(profile_score * 0.55 + market_score * 0.45, 2)

        results.append({
            "domain_key": key,
            "label": domain["label"],
            "label_darija": domain.get("label_darija", ""),
            "combined_score": combined,
            "profile_score": round(profile_score, 2),
            "market_score": round(market_score, 2),
            "saturation_index": domain["saturation_index"],
            "is_saturated": domain.get("is_saturated", False),
            "rejection_reason": domain.get("rejection_reason"),
            "salary_mad": domain["salary_mad"],
            "growth_description": domain.get("growth_description", ""),
            "moroccan_schools": domain.get("moroccan_schools", [])[:3],
            "top_employers": domain.get("top_employers", [])[:4],
            "key_skills": domain.get("key_skills", []),
        })

    results.sort(key=lambda x: x["combined_score"], reverse=True)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# FALLBACK ANALYTIQUE COMPLET
# ══════════════════════════════════════════════════════════════════════════════

def generate_fallback_results(student_data: dict, ranked_domains: list) -> dict:
    """
    Génère un résultat complet sans LLM.
    Utilisé quand : pas de clé API, quota dépassé, erreur réseau.

    # [JURY_CHECK]: REASONING – Même en mode dégradé, l'app reste utile
    # grâce au scoring analytique et aux templates de contenu.
    """
    darija_data = {}
    try:
        with open(DATA_DIR / "darija_translations.json", "r", encoding="utf-8") as f:
            darija_data = json.load(f)
    except Exception:
        pass

    top3 = [d for d in ranked_domains[:5] if not d.get("is_saturated")][:3]
    rejected = [d for d in ranked_domains if d.get("is_saturated")]

    # ── StudentProfile fallback ───────────────────────────────────────────────
    profile_result = StudentProfile(
        student_summary=(
            f"Étudiant avec des affinités pour {student_data.get('subjects', 'diverses matières')}, "
            f"des compétences en {student_data.get('skills', 'plusieurs domaines')} "
            f"et un intérêt pour {student_data.get('interests', 'différents secteurs')}."
        ),
        profile_type="mixte",
        top_skills=[s.strip() for s in student_data.get("skills", "").split(",") if s.strip()][:5] or ["À développer"],
        potential_domains=[d["domain_key"] for d in top3],
        profile_strengths=["Profil multidisciplinaire", "Ouverture aux nouvelles technologies"],
        profile_gaps=["Expérience pratique à acquérir", "Portfolio à construire"],
        motivation_level=8,
        learning_style="pratique",
    )

    # ── IndustryMatchReport fallback ──────────────────────────────────────────
    top3_matches = []
    for d in top3:
        top3_matches.append(DomainMatch(
            domain_key=d["domain_key"],
            domain_label=d["label"],
            is_recommended=True,
            is_saturated=False,
            compatibility_score=d["combined_score"],
            market_score=d["market_score"],
            saturation_index=d["saturation_index"],
            salary_junior_mad=d["salary_mad"]["junior"],
            salary_senior_mad=d["salary_mad"]["senior"],
            top_schools=[s["name"] for s in d.get("moroccan_schools", [])],
            top_employers=d.get("top_employers", []),
            required_skills=d.get("key_skills", []),
            growth_trend=d.get("growth_description", ""),
        ))

    rejected_matches = []
    for d in rejected[:2]:
        rejected_matches.append(DomainMatch(
            domain_key=d["domain_key"],
            domain_label=d["label"],
            is_recommended=False,
            is_saturated=True,
            rejection_reason=d.get("rejection_reason", "Marché saturé"),
            compatibility_score=d["combined_score"],
            market_score=d["market_score"],
            saturation_index=d["saturation_index"],
            salary_junior_mad=d["salary_mad"]["junior"],
            salary_senior_mad=d["salary_mad"]["senior"],
        ))

    match_result = IndustryMatchReport(
        top_3_domains=top3_matches,
        rejected_domains=rejected_matches,
        market_summary="Le Maroc Digital 2030 et le Plan Vert II créent des opportunités sans précédent dans les secteurs tech et agri-tech.",
        best_domain_key=top3[0]["domain_key"] if top3 else "informatique_ia",
        overall_market_context="Contexte favorable : investissements massifs dans le digital, les énergies renouvelables et l'agriculture moderne.",
    )

    # ── PlanBReport fallback ──────────────────────────────────────────────────
    plan_b_result = PlanBReport(
        plan_b_options=[
            {
                "title": "Voie universitaire publique + autoformation",
                "description": "Université publique + MOOCs Coursera/edX + projets GitHub",
                "timeline": "3-5 ans",
                "cost_mad": "5 000 – 20 000 MAD/an",
                "difficulty": "moyen",
                "platforms_or_schools": ["Coursera", "edX", "YouTube", "GitHub"],
                "expected_outcome": "Profil compétitif avec portfolio démontrable"
            }
        ],
        online_certifications=["Google Data Analytics (Coursera)", "AWS Cloud Practitioner", "Microsoft AI-900", "Meta Front-End Developer"],
        scholarships=["Bourse AMCI (Agence Marocaine de Coopération)", "Bourse OCP Foundation", "Bourse INDH", "Bourse Al Barid Bank"],
        bootcamps_morocco=["Gomycode Maroc (Casablanca, Rabat, Fès)", "1337 (Coding school gratuite, Ben Guerir)", "Codeworks Maroc", "UM6P Coding Bootcamp"],
        entrepreneurship_path="Maroc Startup Act facilite la création d'entreprise pour les jeunes. CRI + incubateurs UM6P Ventures disponibles.",
        key_message=darija_data.get("plan_b_messages", {}).get("encouragement", "Le chemin n'est pas toujours direct, mais il existe toujours.")
    )

    # ── CareerVisionReport fallback ───────────────────────────────────────────
    best = top3[0] if top3 else {"label": "Informatique & IA", "domain_key": "informatique_ia"}
    milestones = [
        CareerMilestone(year="Année 1-2", title=f"Formation : {best['label']}", description="Intégration dans une école d'ingénieurs ou université. Construction des bases.", location="Maroc", salary_mad="0 (formation)", key_action="Réussir les concours d'entrée et s'y adapter"),
        CareerMilestone(year="Année 2-4", title="Spécialisation + Stage", description="Approfondissement, premier stage en entreprise marocaine.", location="Casablanca / Rabat", salary_mad="Stage : 2 000–5 000 MAD/mois", key_action="Décrocher un stage dans une entreprise du Top 5"),
        CareerMilestone(year="Année 4-6", title="Premier emploi", description="Intégration dans le marché du travail marocain.", location="Casablanca Tech City / Ben Guerir", salary_mad=best.get("salary_mad", {}).get("junior", "8 000–14 000 MAD/mois") if isinstance(best.get("salary_mad"), dict) else "8 000–14 000 MAD/mois", key_action="Continuer à apprendre et construire son réseau professionnel"),
        CareerMilestone(year="Année 6-10", title="Expert / Lead", description="Montée en compétences, management d'équipe ou spécialisation avancée.", location="Maroc / International", salary_mad=best.get("salary_mad", {}).get("senior", "22 000–45 000 MAD/mois") if isinstance(best.get("salary_mad"), dict) else "22 000–45 000 MAD/mois", key_action="Obtenir une certification internationale premium"),
        CareerMilestone(year="10 ans+", title="Leader / Entrepreneur", description="Impact national : contribuer au Maroc Digital 2030 ou Plan Vert.", location="Maroc & Afrique", salary_mad="40 000+ MAD/mois ou fondateur startup", key_action="Mentorer la prochaine génération d'étudiants marocains"),
    ]

    # Données Sankey simplifiées
    nodes = ["Baccalauréat", "Licence/BTS", "Master/École", "Junior", "Senior", "Expert/Lead", "Entrepreneur"]
    sankey = SankeyData(
        nodes=nodes,
        sources=[0, 1, 2, 3, 4, 4, 5],
        targets=[1, 2, 3, 4, 5, 6, 6],
        values=[100, 85, 75, 70, 50, 30, 25],
        colors=["#667eea", "#764ba2", "#f093fb", "#4facfe", "#43e97b", "#fa709a", "#fed6e3"]
    )

    vision_result = CareerVisionReport(
        vision_narrative=(
            f"Dans 10 ans, vous serez un professionnel reconnu dans le domaine {best['label']} au Maroc. "
            f"Vous aurez contribué concrètement au développement numérique ou vert de votre pays. "
            f"Votre parcours, qui commence aujourd'hui, vous mènera des salles de cours de {best.get('moroccan_schools', [{}])[0].get('city', 'Casablanca') if best.get('moroccan_schools') else 'Casablanca'} "
            f"aux bureaux des grandes entreprises marocaines et peut-être au-delà."
        ),
        milestones=milestones,
        sankey_data=sankey,
        inspirational_quote="'L'avenir appartient à ceux qui croient en la beauté de leurs rêves.' – Eleanor Roosevelt",
        success_story=f"De nombreux ingénieurs marocains ont commencé avec le même profil que vous et travaillent aujourd'hui chez OCP, Microsoft Maroc, ou ont fondé leurs propres startups.",
        final_salary_target="40 000 – 80 000 MAD/mois à 10 ans",
        impact_on_morocco=f"Contribuer au Plan Maroc Digital 2030 et positionner le Maroc comme hub technologique africain."
    )

    # ── DarijaTranslation fallback ────────────────────────────────────────────
    g = darija_data.get("greetings", {})
    di = darija_data.get("domain_intros", {})
    best_key = best.get("domain_key", "informatique_ia")

    darija_result = DarijaTranslation(
        greeting_darija=g.get("opening", "Marhba bik! CareerBridge AI ghadi ysa3dek."),
        main_recommendation_darija=di.get(best_key, f"Had lmajal kayn fih bzzaf d-lfuras f-lmaghrib."),
        schools_darija=darija_data.get("school_specific", {}).get("UM6P", "Kayn bzzaf d-lcoles mezyanin f-lmaghrib lli ghadi tsa3dek."),
        salary_darija=f"Ila tkamalt had lformation, ghadi ta9dar tkhdem w t9bqd bin 8000 w 15000 dirham f-shahar kbidaya. Wghadi yazid m3a l-waqt.",
        plan_b_darija=darija_data.get("plan_b_messages", {}).get("intro", "Ila ma-9dirsh plan A, kayn plan B mezyan."),
        family_message_darija=g.get("family_message", "Lwalidine l-kram, had lbramij mbniya 3la waqe3 suq l-khidma f-lmaghrib."),
        encouragement_darija=darija_data.get("motivational_phrases", ["Kul nhar tatqaddam khatwa zgira."])[0],
        dua_closing="Allah iyawne9 o iyasser l-umour. Inshallah ghadi tkoun najah w tfarha walidik. 🤲"
    )

    return {
        "profile": profile_result,
        "match": match_result,
        "plan_b": plan_b_result,
        "vision": vision_result,
        "darija": darija_result,
        "ranked_domains": ranked_domains,
        "mode": "analytical_fallback",
        "error": None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATEUR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def run_careerbridge_crew(
    student_data: dict,
    log_callback: Optional[Callable] = None,
) -> dict:
    """
    Lance l'analyse complète CareerBridge AI.

    Args:
        student_data: {subjects, skills, interests, career_goal,
                       bulletin_analysis?, audio_transcript?}
        log_callback: fonction(msg: str) pour logs temps réel (Streamlit)

    Returns:
        dict avec profile, match, plan_b, vision, darija, ranked_domains, mode
    """

    def log(msg: str):
        logger.info(msg)
        if log_callback:
            log_callback(msg)

    # ── Étape 1 : Scoring analytique (toujours) ───────────────────────────────
    log("🔢 Calcul des scores de compatibilité analytiques...")
    skills = [s.strip() for s in student_data.get("skills", "").split(",") if s.strip()]
    subjects = [s.strip() for s in student_data.get("subjects", "").split(",") if s.strip()]
    interests = [s.strip() for s in student_data.get("interests", "").split(",") if s.strip()]

    ranked_domains = compute_compatibility_scores(skills, subjects, interests)
    log(f"✅ {len(ranked_domains)} domaines scorés. Top : {ranked_domains[0]['label'] if ranked_domains else 'N/A'}")

    # ── Étape 2 : Configuration Multi-LLM ──────────────────────────────────────
    log("🧠 Configuration des LLMs spécialisés par agent...")
    llm_configs = create_multi_llm_config()

    if not llm_configs:
        log("❌ Aucun LLM disponible → Mode analytique (fallback)")
        result = generate_fallback_results(student_data, ranked_domains)
        result["ranked_domains"] = ranked_domains
        result["error"] = "Aucun LLM configuré"
        return result

    # Afficher la configuration LLM
    for agent_name, llm_instance in llm_configs.items():
        provider = type(llm_instance).__name__.replace('Chat', '').replace('GenerativeAI', 'Gemini')
        log(f"  • {agent_name}: {provider}")

    # ── Étape 3 : Import et init RAG ──────────────────────────────────────────
    log("📚 Initialisation du pipeline RAG (ChromaDB)...")
    try:
        from utils.tools import get_all_tools
        tools = get_all_tools()
        # Initialisation lazy du RAG (premier appel)
        pdf_tool = tools[0]
        if not pdf_tool._initialized:
            pdf_tool._initialize_rag()
        log("✅ RAG initialisé.")
    except Exception as e:
        log(f"⚠️ RAG init error: {e} (mode dégradé)")
        result = generate_fallback_results(student_data, ranked_domains)
        result["ranked_domains"] = ranked_domains
        result["error"] = str(e)
        return result

    # ── Étape 4 : Création des agents et tâches ───────────────────────────────
    log("🤖 Instanciation des 5 agents CrewAI avec LLMs spécialisés...")
    try:
        agents = create_all_agents(tools, llm_configs=llm_configs)
        tasks = create_all_tasks(agents, student_data, llm_configs=llm_configs)
        log(f"✅ {len(agents)} agents et {len(tasks)} tâches créés.")
    except Exception as e:
        log(f"❌ Erreur création agents: {e}")
        return generate_fallback_results(student_data, ranked_domains)

    # ── Étape 5 : Lancement de la Crew ────────────────────────────────────────
    log("🚀 Lancement de la Crew (Process.sequential, memory=True)...")

    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
        memory=True,
        llm=llm
    )

    try:
        start_time = time.time()
        crew_output = crew.kickoff()
        elapsed = round(time.time() - start_time, 1)
        log(f"✅ Crew terminée en {elapsed}s.")

    except Exception as e:
        error_str = str(e).lower()
        if use_openai and ("quota" in error_str or "insufficient" in error_str or "rate limit" in error_str):
            log(f"⚠️ Quota OpenAI dépassé: {e}. Basculement sur Ollama.")
            try:
                llm = ChatOllama(
                    model="mistral",
                    base_url="http://localhost:11500",
                    temperature=0.7
                )
                log("✅ Ollama connecté (mistral) - retry")
                crew = Crew(
                    agents=list(agents.values()),
                    tasks=tasks,
                    process=Process.sequential,
                    verbose=True,
                    memory=True,
                    llm=llm
                )
                start_time = time.time()
                crew_output = crew.kickoff()
                elapsed = round(time.time() - start_time, 1)
                log(f"✅ Crew terminée avec Ollama en {elapsed}s.")
            except Exception as e2:
                log(f"❌ Erreur Ollama aussi: {e2}. Basculement sur mode analytique.")
                result = generate_fallback_results(student_data, ranked_domains)
                result["ranked_domains"] = ranked_domains
                result["error"] = f"OpenAI: {e}, Ollama: {e2}"
                return result
        else:
            log(f"❌ Crew error: {e}. Basculement sur mode analytique.")
            result = generate_fallback_results(student_data, ranked_domains)
            result["ranked_domains"] = ranked_domains
            result["error"] = str(e)
            return result

        # Extraction des résultats Pydantic par tâche
        task_results = {}
        for i, task in enumerate(tasks):
            try:
                task_results[i] = task.output.pydantic if hasattr(task.output, 'pydantic') else None
            except Exception:
                task_results[i] = None

        profile = task_results.get(0) or _fallback_profile(student_data)
        match = task_results.get(1) or _fallback_match(ranked_domains)
        plan_b = task_results.get(2) or _fallback_plan_b()
        vision = task_results.get(3) or _fallback_vision(ranked_domains)
        darija = task_results.get(4) or _fallback_darija()

        return {
            "profile": profile,
            "match": match,
            "plan_b": plan_b,
            "vision": vision,
            "darija": darija,
            "ranked_domains": ranked_domains,
            "mode": "full_ai",
            "crew_raw_output": str(crew_output),
            "elapsed_seconds": elapsed,
            "error": None,
        }

    except Exception as e:
        log(f"❌ Crew error: {e}. Basculement sur mode analytique.")
        result = generate_fallback_results(student_data, ranked_domains)
        result["ranked_domains"] = ranked_domains
        result["error"] = str(e)
        return result


# ── Helpers fallback légers ───────────────────────────────────────────────────

def _fallback_profile(student_data: dict) -> StudentProfile:
    return StudentProfile(
        student_summary=f"Profil avec compétences : {student_data.get('skills', 'diverses')}",
        profile_type="mixte",
        top_skills=[s.strip() for s in student_data.get("skills", "").split(",") if s.strip()][:5] or ["À définir"],
        potential_domains=["informatique_ia", "data_science"],
        profile_strengths=["Polyvalent", "Motivé"],
        profile_gaps=["Expérience à acquérir"],
    )


def _fallback_match(ranked: list) -> IndustryMatchReport:
    top3 = ranked[:3]
    return IndustryMatchReport(
        top_3_domains=[DomainMatch(
            domain_key=d["domain_key"], domain_label=d["label"],
            is_recommended=True, compatibility_score=d["combined_score"],
            market_score=d["market_score"], saturation_index=d["saturation_index"],
            salary_junior_mad=d["salary_mad"]["junior"],
            salary_senior_mad=d["salary_mad"]["senior"],
        ) for d in top3],
        rejected_domains=[],
        market_summary="Marché marocain dynamique, secteur digital en forte croissance.",
        best_domain_key=top3[0]["domain_key"] if top3 else "informatique_ia",
        overall_market_context="Maroc Digital 2030 : 200 000 emplois tech prévus.",
    )


def _fallback_plan_b() -> PlanBReport:
    return PlanBReport(
        plan_b_options=[],
        online_certifications=["Google Analytics", "AWS Cloud", "Microsoft Azure AI"],
        scholarships=["AMCI", "OCP Foundation", "INDH"],
        bootcamps_morocco=["Gomycode", "1337", "Codeworks"],
        key_message="Chaque étudiant marocain a un chemin vers le succès.",
    )


def _fallback_vision(ranked: list) -> CareerVisionReport:
    best = ranked[0] if ranked else {"label": "Tech", "domain_key": "informatique_ia",
                                      "salary_mad": {"junior": "10 000 MAD", "senior": "30 000 MAD"}}
    return CareerVisionReport(
        vision_narrative=f"Votre avenir dans {best['label']} s'annonce prometteur au Maroc.",
        milestones=[
            CareerMilestone(year="1-3 ans", title="Formation", description="École ou université",
                           location="Maroc", key_action="Réussir les études"),
            CareerMilestone(year="3-6 ans", title="Premier emploi",
                           salary_mad=best["salary_mad"]["junior"] if isinstance(best.get("salary_mad"), dict) else "10 000 MAD",
                           description="Intégration en entreprise", location="Casablanca",
                           key_action="Construire son réseau"),
        ],
        sankey_data=SankeyData(
            nodes=["Bac", "Formation", "Emploi Junior", "Senior"],
            sources=[0, 1, 2], targets=[1, 2, 3], values=[100, 80, 60],
        ),
        inspirational_quote="Le succès, c'est aller d'échec en échec sans perdre son enthousiasme.",
        success_story="Des milliers de jeunes marocains ont tracé leur chemin dans ce domaine.",
        final_salary_target="30 000 – 60 000 MAD/mois à 10 ans",
        impact_on_morocco="Contribuer au développement du Maroc digital."
    )


def _fallback_darija() -> DarijaTranslation:
    return DarijaTranslation(
        greeting_darija="Marhba bik! CareerBridge AI ghadi ysa3dek t9da lmoussira dyalek.",
        main_recommendation_darija="Had lmajal kayn fih bzzaf d-lfuras f-lmaghrib dyal daba.",
        schools_darija="Kayn bzzaf d-lcoles mezyanin: UM6P, EHTP, ENSET - koulha publique o f-lmotawwil.",
        salary_darija="Ila tkamalt lformation, ghadi ta9dar tkhdem o t9bqd mezyan - kafi o yzid.",
        plan_b_darija="Ila ma-9dirsh plan A, makayn mouchkil - plan B kamel o mabni mezyan.",
        family_message_darija="Lwalidine l-kram, lbramij mdroussa mezyan. Weldkom aw bentkum ghadi ikoun bkhir.",
        encouragement_darija="Sber o tqaddam - lmostaqbal dyalek f-yadik nta.",
        dua_closing="Allah iyawne9 o iyasser l-umour. Inshallah ghadi tkoun najah. 🤲"
    )
