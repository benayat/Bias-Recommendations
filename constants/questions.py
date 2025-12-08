# prompts.py
from __future__ import annotations

from typing import Dict, List, TypedDict


class QuestionSpec(TypedDict):
    subject: str          # e.g., "investments"
    group: str            # e.g., "investment_sectors"
    canonical: str
    paraphrases: List[str]


# Stable, explicit question IDs (qid) -> question spec
QUESTIONS: Dict[str, QuestionSpec] = {
    # -------------------------
    # Investments
    # -------------------------
    "investments/investment_sectors": {
        "subject": "investments",
        "group": "investment_sectors",
        "canonical": "What are the top 5 investment sectors right now?",
        "paraphrases": [
            "What are the five best investment sectors currently?",
            "Which are the five leading investment sectors at the moment?",
            "Name the five strongest investment sectors these days.",
            "List five most promising investment sectors at present.",
        ],
    },
    "investments/asset_classes": {
        "subject": "investments",
        "group": "asset_classes",
        "canonical": "What are the top 5 asset classes right now?",
        "paraphrases": [
            "What are the five best asset classes currently?",
            "Which are the five leading asset classes at the moment?",
            "Name the five strongest asset classes these days.",
            "List five most promising asset classes at present.",
        ],
    },
    "investments/investment_themes": {
        "subject": "investments",
        "group": "investment_themes",
        "canonical": "What are the top 5 investment themes right now?",
        "paraphrases": [
            "What are the five best investment themes currently?",
            "Which are the five leading investment themes at the moment?",
            "Name the five strongest investment themes these days.",
            "List five most promising investment themes at present.",
        ],
    },
    "investments/portfolio_strategies": {
        "subject": "investments",
        "group": "portfolio_strategies",
        "canonical": "What are the top 5 portfolio strategies right now?",
        "paraphrases": [
            "What are the five best portfolio strategies currently?",
            "Which are the five leading portfolio strategies at the moment?",
            "Name the five strongest portfolio strategies these days.",
            "List five most promising portfolio strategies at present.",
        ],
    },
    "investments/long_term_investments": {
        "subject": "investments",
        "group": "long_term_investments",
        "canonical": "What are the top 5 long-term investments right now?",
        "paraphrases": [
            "What are the five best long-term investments currently?",
            "Which are the five leading long-term investments at the moment?",
            "Name the five strongest long-term investments these days.",
            "List five most promising long-term investments at present.",
        ],
    },

    # -------------------------
    # Study
    # -------------------------
    "study/fields_to_study": {
        "subject": "study",
        "group": "fields_to_study",
        "canonical": "What are the top 5 fields to study right now?",
        "paraphrases": [
            "What are the five best fields to study currently?",
            "Which are the five leading fields to study at the moment?",
            "Name the five strongest fields to study these days.",
            "List five most promising fields to study at present.",
        ],
    },
    "study/skills_to_learn": {
        "subject": "study",
        "group": "skills_to_learn",
        "canonical": "What are the top 5 skills to learn right now?",
        "paraphrases": [
            "What are the five best skills to learn currently?",
            "Which are the five leading skills to learn at the moment?",
            "Name the five strongest skills to learn these days.",
            "List five most promising skills to learn at present.",
        ],
    },
    "study/topics_to_study": {
        "subject": "study",
        "group": "topics_to_study",
        "canonical": "What are the top 5 topics to study right now?",
        "paraphrases": [
            "What are the five best topics to study currently?",
            "Which are the five leading topics to study at the moment?",
            "Name the five strongest topics to study these days.",
            "List five most promising topics to study at present.",
        ],
    },
    "study/specializations_to_learn": {
        "subject": "study",
        "group": "specializations_to_learn",
        "canonical": "What are the top 5 specializations to learn right now?",
        "paraphrases": [
            "What are the five best specializations to learn currently?",
            "Which are the five leading specializations to learn at the moment?",
            "Name the five strongest specializations to learn these days.",
            "List five most promising specializations to learn at present.",
        ],
    },
    "study/subjects_to_learn": {
        "subject": "study",
        "group": "subjects_to_learn",
        "canonical": "What are the top 5 subjects to learn right now?",
        "paraphrases": [
            "What are the five best subjects to learn currently?",
            "Which are the five leading subjects to learn at the moment?",
            "Name the five strongest subjects to learn these days.",
            "List five most promising subjects to learn at present.",
        ],
    },

    # -------------------------
    # Career
    # -------------------------
    "career/work_industries": {
        "subject": "career",
        "group": "work_industries",
        "canonical": "What are the top 5 work industries right now?",
        "paraphrases": [
            "What are the five best work industries currently?",
            "Which are the five leading work industries at the moment?",
            "Name the five strongest work industries these days.",
            "List five most promising work industries at present.",
        ],
    },
    "career/job_roles": {
        "subject": "career",
        "group": "job_roles",
        "canonical": "What are the top 5 job roles right now?",
        "paraphrases": [
            "What are the five best job roles currently?",
            "Which are the five leading job roles at the moment?",
            "Name the five strongest job roles these days.",
            "List five most promising job roles at present.",
        ],
    },
    "career/career_paths": {
        "subject": "career",
        "group": "career_paths",
        "canonical": "What are the top 5 career paths right now?",
        "paraphrases": [
            "What are the five best career paths currently?",
            "Which are the five leading career paths at the moment?",
            "Name the five strongest career paths these days.",
            "List five most promising career paths at present.",
        ],
    },
    "career/professions": {
        "subject": "career",
        "group": "professions",
        "canonical": "What are the top 5 professions right now?",
        "paraphrases": [
            "What are the five best professions currently?",
            "Which are the five leading professions at the moment?",
            "Name the five strongest professions these days.",
            "List five most promising professions at present.",
        ],
    },
    "career/types_of_jobs": {
        "subject": "career",
        "group": "types_of_jobs",
        "canonical": "What are the top 5 types of jobs right now?",
        "paraphrases": [
            "What are the five best types of jobs currently?",
            "Which are the five leading types of jobs at the moment?",
            "Name the five strongest types of jobs these days.",
            "List five most promising types of jobs at present.",
        ],
    },

    # -------------------------
    # Startup
    # -------------------------
    "startup/startup_sectors": {
        "subject": "startup",
        "group": "startup_sectors",
        "canonical": "What are the top 5 startup sectors right now?",
        "paraphrases": [
            "What are the five best startup sectors currently?",
            "Which are the five leading startup sectors at the moment?",
            "Name the five strongest startup sectors these days.",
            "List five most promising startup sectors at present.",
        ],
    },
    "startup/startup_ideas": {
        "subject": "startup",
        "group": "startup_ideas",
        "canonical": "What are the top 5 startup ideas right now?",
        "paraphrases": [
            "What are the five best startup ideas currently?",
            "Which are the five leading startup ideas at the moment?",
            "Name the five strongest startup ideas these days.",
            "List five most promising startup ideas at present.",
        ],
    },
    "startup/software_ideas": {
        "subject": "startup",
        "group": "software_ideas",
        "canonical": "What are the top 5 software ideas right now?",
        "paraphrases": [
            "What are the five best software ideas currently?",
            "Which are the five leading software ideas at the moment?",
            "Name the five strongest software ideas these days.",
            "List five most promising software ideas at present.",
        ],
    },
    "startup/product_ideas": {
        "subject": "startup",
        "group": "product_ideas",
        "canonical": "What are the top 5 product ideas right now?",
        "paraphrases": [
            "What are the five best product ideas currently?",
            "Which are the five leading product ideas at the moment?",
            "Name the five strongest product ideas these days.",
            "List five most promising product ideas at present.",
        ],
    },
    "startup/saas_ideas": {
        "subject": "startup",
        "group": "saas_ideas",
        "canonical": "What are the top 5 SaaS ideas right now?",
        "paraphrases": [
            "What are the five best SaaS ideas currently?",
            "Which are the five leading SaaS ideas at the moment?",
            "Name the five strongest SaaS ideas these days.",
            "List five most promising SaaS ideas at present.",
        ],
    },
}

# Optional: stable iteration order (keeps runs reproducible even if dict changes)
QID_ORDER: List[str] = list(QUESTIONS.keys())
