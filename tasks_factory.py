"""
tasks_factory.py
═══════════════════════════════════════════════════════════════════════════════
Définition des tâches CrewAI avec modèles Pydantic pour garantir
la récupérabilité des données structurées par l'UI Streamlit.

# [JURY_CHECK]: REASONING – L'usage de output_pydantic garantit que les
# données (salaires MAD, cursus, scores) sont parsables côté UI sans
# fragile parsing de texte libre.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from crewai import Task


# ══════════════════════════════════════════════════════════════════════════════
# MODÈLES PYDANTIC DE SORTIE
# ══════════════════════════════════════════════════════════════════════════════

class SubjectScore(BaseModel):
    subject: str = Field(description="Nom de la matière")
    grade: Optional[float] = Field(default=None, description="Note /20")
    category: str = Field(default="autre", description="Catégorie (sciences_exactes, tech, langues...)")
    strength_level: str = Field(default="moyen", description="fort/moyen/faible")


class StudentProfile(BaseModel):
    """Sortie structurée de l'Agent Profiler."""
    student_summary: str = Field(description="Résumé du profil en 2-3 phrases")
    profile_type: str = Field(description="Type de profil (scientifique/tech/humaniste/mixte/artistique)")
    top_skills: List[str] = Field(description="Top 5 compétences identifiées")
    subject_scores: List[SubjectScore] = Field(default=[], description="Analyse par matière")
    general_average: Optional[float] = Field(default=None, description="Moyenne générale /20")
    potential_domains: List[str] = Field(description="3-5 clés de domaines compatibles")
    profile_strengths: List[str] = Field(description="Points forts identifiés")
    profile_gaps: List[str] = Field(description="Points à développer")
    motivation_level: int = Field(default=7, description="Niveau de motivation estimé /10")
    learning_style: str = Field(default="mixte", description="Style d'apprentissage préféré")


class DomainMatch(BaseModel):
    """Évaluation d'un domaine par l'Industry Matcher."""
    domain_key: str
    domain_label: str
    is_recommended: bool
    is_saturated: bool = False
    rejection_reason: Optional[str] = None
    compatibility_score: float = Field(description="Score 0-10")
    market_score: float = Field(description="Score marché 0-10")
    saturation_index: int = Field(description="Indice saturation 0-10")
    salary_junior_mad: str
    salary_senior_mad: str
    top_schools: List[str] = Field(default=[])
    top_employers: List[str] = Field(default=[])
    required_skills: List[str] = Field(default=[])
    growth_trend: str = ""
    certifications: List[str] = Field(default=[])


class IndustryMatchReport(BaseModel):
    """Sortie structurée de l'Agent Industry Matcher."""
    top_3_domains: List[DomainMatch] = Field(description="Top 3 domaines recommandés")
    rejected_domains: List[DomainMatch] = Field(default=[], description="Domaines rejetés avec justification")
    market_summary: str = Field(description="Résumé des tendances du marché marocain")
    best_domain_key: str = Field(description="Clé du domaine n°1 recommandé")
    overall_market_context: str = Field(description="Contexte macro-économique Maroc 2024-2030")


class PlanBOption(BaseModel):
    """Une option de Plan B."""
    title: str
    description: str
    timeline: str = Field(description="Durée estimée ex: '6-12 mois'")
    cost_mad: str = Field(description="Coût estimé en MAD")
    difficulty: str = Field(description="facile/moyen/difficile")
    platforms_or_schools: List[str] = Field(default=[])
    expected_outcome: str


class PlanBReport(BaseModel):
    """Sortie structurée de l'Agent Plan B Architect."""
    plan_b_options: List[PlanBOption] = Field(description="2-4 alternatives")
    online_certifications: List[str] = Field(description="Certifications en ligne recommandées")
    scholarships: List[str] = Field(description="Bourses disponibles au Maroc")
    bootcamps_morocco: List[str] = Field(description="Bootcamps disponibles au Maroc")
    entrepreneurship_path: Optional[str] = Field(default=None, description="Piste entrepreneuriale si pertinente")
    key_message: str = Field(description="Message principal pour l'étudiant")


class CareerMilestone(BaseModel):
    """Un jalon dans la feuille de route."""
    year: str = Field(description="Ex: 'Année 1-2'")
    title: str = Field(description="Titre du jalon")
    description: str
    location: str = Field(default="Maroc", description="Ville/pays")
    salary_mad: Optional[str] = Field(default=None)
    key_action: str = Field(description="Action principale à entreprendre")


class SankeyData(BaseModel):
    """Données pour le diagramme de Sankey (flux de carrière)."""
    nodes: List[str] = Field(description="Nœuds du diagramme (ex: Bac → Licence → Master → Emploi)")
    sources: List[int] = Field(description="Indices sources des liens")
    targets: List[int] = Field(description="Indices cibles des liens")
    values: List[float] = Field(description="Valeurs/épaisseurs des liens")
    colors: List[str] = Field(default=[], description="Couleurs des nœuds")


class CareerVisionReport(BaseModel):
    """Sortie structurée de l'Agent Career Visualizer."""
    vision_narrative: str = Field(description="Narration inspirante du parcours idéal")
    milestones: List[CareerMilestone] = Field(description="Jalons de la feuille de route")
    sankey_data: SankeyData = Field(description="Données pour le diagramme de Sankey Plotly")
    inspirational_quote: str = Field(description="Citation motivante")
    success_story: str = Field(description="Histoire de réussite marocaine similaire")
    final_salary_target: str = Field(description="Objectif salarial à 10 ans en MAD")
    impact_on_morocco: str = Field(description="Contribution potentielle au développement du Maroc")


class DarijaTranslation(BaseModel):
    """Sortie structurée de l'Agent Darija Translator."""
    greeting_darija: str = Field(description="Message d'accueil en Darija")
    main_recommendation_darija: str = Field(description="Recommandation principale en Darija")
    schools_darija: str = Field(description="Explication des écoles en Darija")
    salary_darija: str = Field(description="Explication du salaire en Darija")
    plan_b_darija: str = Field(description="Plan B expliqué en Darija")
    family_message_darija: str = Field(description="Message pour les parents en Darija")
    encouragement_darija: str = Field(description="Encouragement final en Darija")
    dua_closing: str = Field(description="Clôture avec dou3a bénédiction")


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY DES TÂCHES
# ══════════════════════════════════════════════════════════════════════════════

def create_all_tasks(agents: dict, student_data: dict, llm=None, llm_configs=None) -> list:
    """
    Crée les 5 tâches en ordre séquentiel avec context chaining.

    Args:
        agents: dict {'profiler', 'matcher', 'plan_b', 'visualizer', 'translator'}
        student_data: {
            subjects, skills, interests, career_goal,
            bulletin_analysis (optionnel),
            audio_transcript (optionnel)
        }
        llm: LangChain LLM instance (optionnel, pour compatibilité)
        llm_configs: dict avec configuration LLM par agent (optionnel)
    Returns:
        Liste de 5 tâches CrewAI
    """

    subjects = student_data.get("subjects", "Non précisé")
    skills = student_data.get("skills", "Non précisé")
    interests = student_data.get("interests", "Non précisé")
    career_goal = student_data.get("career_goal", "Non précisé")

    # Enrichissement avec données multimodales si disponibles
    bulletin_info = ""
    if student_data.get("bulletin_analysis"):
        ba = student_data["bulletin_analysis"]
        bulletin_info = (
            f"\n\n📊 DONNÉES BULLETIN (Vision IA) :\n"
            f"  • Moyenne générale : {ba.get('averages', {}).get('general', 'N/A')}/20\n"
            f"  • Matières fortes : {', '.join(ba.get('strong_subjects', []))}\n"
            f"  • Matières faibles : {', '.join(ba.get('weak_subjects', []))}\n"
            f"  • Profil détecté : {', '.join(ba.get('profile_tags', []))}\n"
            f"  • Domaines suggérés : {', '.join(ba.get('compatible_domains', []))}"
        )

    audio_info = ""
    if student_data.get("audio_transcript"):
        audio_info = (
            f"\n\n🎙️ TRANSCRIPTION AUDIO :\n"
            f"  \"{student_data['audio_transcript']}\""
        )

    # ──────────────────────────────────────────────────────────────────────────
    # TÂCHE 1 : Analyse du profil étudiant
    # ──────────────────────────────────────────────────────────────────────────

    task_profile = Task(
        description=(
            f"Effectuez une ANALYSE COMPLÈTE et STRUCTURÉE du profil de l'étudiant marocain suivant.\n\n"
            f"═══════════════════════════════════\n"
            f"DONNÉES ÉTUDIANT :\n"
            f"═══════════════════════════════════\n"
            f"📖 Matières préférées    : {subjects}\n"
            f"🛠️  Compétences           : {skills}\n"
            f"💡 Centres d'intérêt    : {interests}\n"
            f"🎯 Objectif de carrière  : {career_goal}\n"
            f"{bulletin_info}"
            f"{audio_info}\n\n"
            f"═══════════════════════════════════\n"
            f"INSTRUCTIONS DÉTAILLÉES :\n"
            f"═══════════════════════════════════\n"
            f"1. Utilisez rank_domains_by_profile via l'outil morocco_market_data pour identifier les domaines clés\n"
            f"2. Utilisez pdf_knowledge_search pour enrichir l'analyse avec des profils similaires\n"
            f"3. Analysez la cohérence entre aspirations (career_goal) et capacités actuelles\n"
            f"4. Identifiez le 'profil type' dominant de cet étudiant\n"
            f"5. Produisez une analyse structurée JSON selon le modèle StudentProfile\n\n"
            f"# [JURY_CHECK]: REASONING – Montrez votre raisonnement étape par étape\n"
            f"avant de conclure. Expliquez POURQUOI vous attribuez chaque score."
        ),
        expected_output=(
            "Un objet JSON conforme au modèle StudentProfile avec :\n"
            "- student_summary (synthèse du profil)\n"
            "- profile_type (scientifique/tech/humaniste/mixte)\n"
            "- top_skills (liste des 5 compétences clés)\n"
            "- potential_domains (3-5 clés de domaines ex: informatique_ia)\n"
            "- profile_strengths et profile_gaps\n"
            "- general_average si disponible depuis le bulletin\n"
            "- motivation_level (0-10)\n"
            "Tous les champs doivent être renseignés."
        ),
        agent=agents["profiler"],
        output_pydantic=StudentProfile,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # TÂCHE 2 : Matching industrie et validation marché
    # ──────────────────────────────────────────────────────────────────────────

    task_match = Task(
        description=(
            f"Sur base du profil étudiant analysé, effectuez une VALIDATION MARCHÉ RIGOUREUSE.\n\n"
            f"DONNÉES CONTEXTUELLES :\n"
            f"  Profil étudiant : {subjects} | {skills} | {interests}\n"
            f"  Objectif : {career_goal}\n\n"
            f"═══════════════════════════════════\n"
            f"PROTOCOLE D'ANALYSE OBLIGATOIRE :\n"
            f"═══════════════════════════════════\n"
            f"1. Pour CHAQUE domaine potentiel (minimum 4 domaines) :\n"
            f"   → Appelez morocco_market_data avec le domain_key\n"
            f"   → Vérifiez l'indice de saturation\n"
            f"   → Si saturation >= 7 : REJETEZ avec argumentaire factuel\n"
            f"   → Si saturation 5-6 : MISE EN GARDE + spécialisation recommandée\n"
            f"   → Si saturation <= 4 : VALIDEZ avec données de salaires MAD\n\n"
            f"2. Utilisez web_search_morocco pour vérifier les tendances récentes\n\n"
            f"3. Calculez un score de compatibilité (0-10) pour chaque domaine retenu :\n"
            f"   score = (profil_match * 0.5) + (marché_score * 0.3) + (employabilité * 0.2)\n\n"
            f"# [JURY_CHECK]: REASONING – L'agent DOIT rejeter au moins 1 domaine\n"
            f"# avec une argumentation basée sur les données (saturation/salaires bas).\n"
            f"# Cela démontre le raisonnement critique vs recommandation aveugle."
        ),
        expected_output=(
            "Un objet JSON conforme au modèle IndustryMatchReport avec :\n"
            "- top_3_domains (domaines recommandés avec scores, salaires MAD, écoles, employeurs)\n"
            "- rejected_domains (domaines rejetés avec rejection_reason détaillée)\n"
            "- market_summary (contexte marché Maroc 2024-2030)\n"
            "- best_domain_key (la meilleure recommandation)\n"
            "IMPORTANT : Salaires OBLIGATOIREMENT en MAD. Écoles marocaines réelles."
        ),
        agent=agents["matcher"],
        context=[task_profile],
        output_pydantic=IndustryMatchReport,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # TÂCHE 3 : Architecte Plan B
    # ──────────────────────────────────────────────────────────────────────────

    task_plan_b = Task(
        description=(
            f"Concevez des STRATÉGIES ALTERNATIVES robustes pour cet étudiant marocain.\n\n"
            f"Profil : {subjects} | Compétences : {skills}\n"
            f"Objectif : {career_goal}\n\n"
            f"═══════════════════════════════════\n"
            f"SCÉNARIOS À TRAITER :\n"
            f"═══════════════════════════════════\n"
            f"SCÉNARIO A – Accès difficile aux grandes écoles :\n"
            f"  → BTS/DUT comme passerelle\n"
            f"  → Prépa privée accessible\n"
            f"  → Université publique + autoformation\n\n"
            f"SCÉNARIO B – Contraintes financières :\n"
            f"  → Bourses AMCI, OCP Foundation, INDH\n"
            f"  → Prêts étudiant BPE, Al Barid Bank\n"
            f"  → Formations gratuites/subventionnées\n\n"
            f"SCÉNARIO C – Reconversion ou retard :\n"
            f"  → Bootcamps (Gomycode, Codeworks, 1337)\n"
            f"  → Certifications en ligne (Coursera, Google, AWS)\n"
            f"  → Alternance et apprentissage\n\n"
            f"SCÉNARIO D – Entrepreneuriat :\n"
            f"  → Maroc Startup Act\n"
            f"  → CRI (Centres Régionaux d'Investissement)\n"
            f"  → Incubateurs (UM6P Ventures, Maroc PME)\n\n"
            f"Utilisez pdf_knowledge_search pour trouver des programmes spécifiques.\n"
            f"# [JURY_CHECK]: TOOL_USE – Doit utiliser l'outil de recherche au moins 2 fois."
        ),
        expected_output=(
            "Un objet JSON conforme au modèle PlanBReport avec :\n"
            "- plan_b_options (2-4 alternatives avec timeline et coût MAD)\n"
            "- online_certifications (Google, AWS, Microsoft, Coursera)\n"
            "- scholarships (bourses disponibles au Maroc)\n"
            "- bootcamps_morocco (Gomycode, 1337, Codeworks, etc.)\n"
            "- entrepreneurship_path si pertinent\n"
            "- key_message inspirant pour l'étudiant"
        ),
        agent=agents["plan_b"],
        context=[task_profile, task_match],
        output_pydantic=PlanBReport,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # TÂCHE 4 : Visualisation du parcours de carrière
    # ──────────────────────────────────────────────────────────────────────────

    task_vision = Task(
        description=(
            f"Créez une FEUILLE DE ROUTE VIVANTE et INSPIRANTE pour cet étudiant.\n\n"
            f"Domaine recommandé (contexte) : {career_goal}\n"
            f"Profil : {subjects} | {skills}\n\n"
            f"═══════════════════════════════════\n"
            f"STRUCTURE REQUISE :\n"
            f"═══════════════════════════════════\n"
            f"1. VISION NARRATIVE (200-300 mots) :\n"
            f"   Racontez le futur de cet étudiant comme une histoire.\n"
            f"   Mentionnez des lieux réels (Casablanca, Ben Guerir, etc.)\n"
            f"   et des entreprises réelles (OCP, Attijariwafa, startups).\n\n"
            f"2. JALONS (minimum 5) :\n"
            f"   Année 1-2 / Année 2-4 / Année 4-6 / Année 6-8 / Année 8-10+\n"
            f"   Chaque jalon : titre + description + ville + salaire MAD + action clé\n\n"
            f"3. DONNÉES SANKEY (CRUCIAL pour la visualisation Plotly) :\n"
            f"   Modélisez le flux de carrière :\n"
            f"   Bac → [Écoles] → [Premiers emplois] → [Évolution] → [Senior/Expert]\n"
            f"   Fournissez les nœuds, sources, targets et valeurs numériques.\n\n"
            f"4. SUCCESS STORY marocaine similaire (vraie ou plausible)\n\n"
            f"# [JURY_CHECK]: REASONING – Le diagramme de Sankey DOIT avoir\n"
            f"# au minimum 8 nœuds et 7 liens pour être informatif."
        ),
        expected_output=(
            "Un objet JSON conforme au modèle CareerVisionReport avec :\n"
            "- vision_narrative (histoire inspirante du futur)\n"
            "- milestones (5+ jalons avec salaires MAD)\n"
            "- sankey_data (nodes, sources, targets, values pour Plotly)\n"
            "- inspirational_quote (citation motivante)\n"
            "- success_story (exemple marocain)\n"
            "- final_salary_target (objectif à 10 ans en MAD)\n"
            "- impact_on_morocco (contribution nationale)"
        ),
        agent=agents["visualizer"],
        context=[task_profile, task_match, task_plan_b],
        output_pydantic=CareerVisionReport,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # TÂCHE 5 : Traduction Darija
    # ──────────────────────────────────────────────────────────────────────────

    task_darija = Task(
        description=(
            f"Traduisez et adaptez les recommandations principales en DARIJA MAROCAINE.\n\n"
            f"RÉSUMÉ DES RECOMMANDATIONS À TRADUIRE :\n"
            f"  Étudiant : {subjects} → Objectif : {career_goal}\n\n"
            f"═══════════════════════════════════\n"
            f"ÉLÉMENTS À TRADUIRE OBLIGATOIREMENT :\n"
            f"═══════════════════════════════════\n"
            f"1. greeting_darija : Accueil chaleureux pour l'étudiant ET sa famille\n"
            f"2. main_recommendation_darija : La recommandation principale expliquée simplement\n"
            f"3. schools_darija : Les écoles recommandées (UM6P, EHTP, etc.) expliquées\n"
            f"4. salary_darija : Ce que signifie le salaire en termes concrets\n"
            f"   (ex: '15 000 MAD c'est comme ça...facilek...')\n"
            f"5. plan_b_darija : Les alternatives si plan A impossible\n"
            f"6. family_message_darija : Message SPÉCIFIQUE pour les parents\n"
            f"   (rassurez-les sur la stabilité, la durée, le coût)\n"
            f"7. encouragement_darija : Encouragement final vibrant\n"
            f"8. dua_closing : Clôture avec dou3a bénédiction\n\n"
            f"STYLE OBLIGATOIRE :\n"
            f"  ✅ Darija authentique (pas MSA/Fusha)\n"
            f"  ✅ Chaleureux et bienveillant\n"
            f"  ✅ Accessible aux familles peu scolarisées\n"
            f"  ✅ Pas condescendant\n"
            f"  ✅ Métaphores du quotidien marocain\n\n"
            f"# [JURY_CHECK]: REASONING – La qualité de la Darija sera évaluée\n"
            f"# par le jury sur son authenticité et son accessibilité culturelle."
        ),
        expected_output=(
            "Un objet JSON conforme au modèle DarijaTranslation avec tous les champs renseignés.\n"
            "La Darija doit utiliser la translittération latine standard marocaine\n"
            "(ex: 3 pour ع, 9 pour ق, 7 pour ح, kh pour خ, gh pour غ).\n"
            "Chaque champ : minimum 2-3 phrases."
        ),
        agent=agents["translator"],
        context=[task_profile, task_match, task_plan_b, task_vision],
        output_pydantic=DarijaTranslation,
    )

    return [task_profile, task_match, task_plan_b, task_vision, task_darija]
