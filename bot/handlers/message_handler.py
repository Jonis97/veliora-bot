import json
import logging
from typing import Any, Optional, Union

from openai import AsyncOpenAI
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InputFile, Message, Update
from telegram.ext import ContextTypes

from bot.services.pipeline_service import ContentPipelineService
from bot.services.youtube_service import YouTubeTranscriptService, extract_video_id
from bot.utils.active_source import NeedActiveSourceError
from bot.utils.dedup import MessageDeduplicator
from bot.utils.errors import GenerationFailedError, TranscriptUnavailableError


LOGGER = logging.getLogger(__name__)

# Onboarding only — separate from active_source / pipeline memory.
user_state: dict[int, dict[str, Optional[str]]] = {}

# Temporary guided-preview state — separate from active_source.
preview_state: dict[int, dict[str, Any]] = {}

_PREVIEW_TRANSCRIPT_MAX = 14_000

_PREVIEW_SYSTEM_DEFAULT = (
    "You are a helpful teacher. Output ONE JSON object only, no markdown.\n"
    "From the source transcript below ONLY (no invented facts):\n"
    '- "topic": one short line (teacher-friendly)\n'
    '- "key_ideas": exactly 3 short strings, each one bullet-worthy idea from the source\n'
    '- "words": 3 to 5 useful English words or short phrases that appear in or are clearly grounded in the source\n'
    "Keep everything short. No card layout, no images.\n"
    "Follow any additional instruction without breaking source-only rules."
)

_PREVIEW_SYSTEM_LESSON = (
    "You are a helpful teacher. Output ONE JSON object only, no markdown.\n"
    "Use the transcript below ONLY as the source (no invented facts).\n"
    'Return ONLY these keys for a lesson preview:\n'
    '- "topic": one short line (teacher-friendly)\n'
    '- "warmup_questions": exactly 5 short warm-up questions grounded in source\n'
    '- "choices": 3 to 4 this-or-that choices grounded in source, format: "Option A or Option B?"\n'
    '- "support_words": at least 5 simple English words or short chunks from source\n'
    "HARD RULES:\n"
    "- Never return '—' or an empty string as a question or choice.\n"
    "- warmup_questions must be exactly 5 real questions.\n"
    "- support_words must have at least 5 real items (words or chunks).\n"
    "- If the source has limited content, derive questions, choices, and words from the topic while staying consistent with the transcript.\n"
    "Do not include key_ideas, discussion_questions, or vocabulary_items."
)

_PREVIEW_SYSTEM_LESSON_A1 = (
    "You are a helpful teacher. Output ONE JSON object only, no markdown.\n"
    "Use the A1 FILTERED SOURCE in the user message ONLY as the source (no invented facts beyond it). "
    "It is a filtered daily-life topic and scenes — not a raw transcript; do not use or assume any other text.\n"
    "This preview is for CEFR level A1 only. Follow ALL rules below.\n\n"
    "GENERAL:\n"
    "- Use only CEFR A1 vocabulary.\n"
    "- Use ONLY Present Simple.\n"
    "- Max sentence length: 6 words (for every English sentence you output, including topic line if possible).\n"
    '- No abstract words (e.g. "freedom", "success").\n'
    "- No idioms or phrasal verbs.\n"
    "- All content must relate to ONE topic.\n"
    '- No placeholders like "—".\n'
    "- No empty fields.\n"
    "- Do not regenerate existing content; only extend if asked (initial generation: fill all fields from source).\n"
    "- Follow exact counts strictly (no more, no less).\n\n"
    "COHERENCE:\n"
    "- All questions, choices, and vocabulary must stay inside the same daily situation as the topic. "
    "Do not introduce unrelated elements.\n"
    "- Example: if topic is 'morning routine at home', every question, word, and choice must relate "
    "only to morning routine at home.\n"
    "- Forbidden: neighbors, strangers, unrelated places or people.\n\n"
    "TOPIC LINE:\n"
    "- The topic must always be concrete, personal, and easy to answer (everyday life).\n"
    "- If the filtered source suggests abstract, psychological, emotional, or conceptual themes, "
    "express the topic as a simple daily-life situation, not an abstract label.\n"
    "- Examples: self-talk → talking to yourself at home; motivation → things that help you every day; "
    "stress → feeling bad at school or work.\n\n"
    'Return ONLY these keys for a lesson preview:\n'
    '- "topic": one short line (teacher-friendly), one concrete daily-life situation, same ONE topic as all items below\n'
    '- "warmup_questions": exactly 5 questions\n'
    '- "core_questions": exactly 4 questions\n'
    '- "choices": exactly 4 items\n'
    '- "support_words": exactly 6 items\n\n'
    "WARM-UP (warmup_questions): exactly 5 questions\n"
    "- Each max 6 words.\n"
    '- Patterns: "Do you...?" / "Is it...?" / "Can you...?"\n'
    "- Personal, about daily life.\n"
    '- Forbidden: "Why" questions, abstract topics.\n\n'
    "CORE QUESTIONS (core_questions): exactly 4 questions\n"
    "- Each max 7 words.\n"
    '- Patterns: "What do you...?" / "Where do you...?" / "When do you...?"\n'
    "- Directly related to topic.\n"
    "- Forbidden: hypothetical questions, complex grammar.\n\n"
    "THIS OR THAT (choices): exactly 4 items\n"
    '- Format ONLY: "X or Y?" (question mark at end).\n'
    "- X and Y: single words or max 2-word phrases.\n"
    '- Concrete only (e.g. "coffee or tea?").\n'
    "- Forbidden: abstract choices, long phrases.\n\n"
    "VOCABULARY (support_words): exactly 6 items\n"
    '- Format each string: "English — Ukrainian" (em dash between English and Ukrainian).\n'
    "- Nouns or verbs only.\n"
    "- Directly related to topic.\n"
    "- Forbidden: rare or abstract words.\n\n"
    "Do NOT include neutral or generic daily routine elements that are not directly related to the main situation "
    "and main problem from the source.\n"
    "If a word, question, or choice does not clearly reflect the core situation and core problem, it must be excluded.\n"
    "Prioritize relevance over variety.\n\n"
    "Example:\n"
    "If topic is 'waking up early and feeling tired':\n"
    "KEEP: tired, sleep, wake up, feel bad, school\n"
    "REMOVE: breakfast, toast, cereal, coffee\n"
    "(unless they directly appear in the source)\n\n"
    "QUALITY CONTROL:\n"
    "- Every single element — each question, each choice, each vocabulary word — must directly relate to "
    "the simplified topic AND the core problem.\n"
    "- Test before including any element:\n"
    "  'Does this word/question help student talk about THIS specific situation?'\n"
    "- If answer is NO → remove it.\n"
    "- Do not add generic filler just to make the lesson feel complete.\n"
    "- Prioritize relevant content over filler content.\n\n"
    "If the filtered source has limited detail, derive items from the topic and scenes while staying consistent with them.\n"
    "Do not include key_ideas, discussion_questions, or vocabulary_items."
)

_PREVIEW_SYSTEM_LESSON_A2 = (
    "You are a helpful teacher. Output ONE JSON object only, no markdown.\n"
    "Use the A2 FILTERED SOURCE in the user message ONLY as the source (no invented facts beyond it). "
    "It is a filtered topic and scenes — not a raw transcript; do not use or assume any other text.\n"
    "This preview is for CEFR level A2 only. Apply ONLY these rules for lesson + A2.\n\n"
    "GLOBAL:\n"
    "- Every element must reflect the main situation and core meaning from the filtered source.\n"
    "- Do NOT introduce unrelated elements.\n"
    "- Do NOT add generic filler.\n"
    "- Simplify the language, NOT the topic.\n"
    "- Follow exact counts strictly (no more, no less).\n"
    "- Do not regenerate existing content; only extend if asked (initial generation: fill all fields from source).\n"
    '- No placeholders like "—".\n'
    "- No empty fields.\n\n"
    'Return ONLY these keys for a lesson preview:\n'
    '- "topic": one short line (teacher-friendly), same situation as the source (simplified, not replaced)\n'
    '- "warmup_questions": exactly 5 questions\n'
    '- "core_questions": exactly 4 questions\n'
    '- "choices": exactly 4 items\n'
    '- "support_words": exactly 6 items (each "English — Ukrainian")\n\n'
    "WARM-UP (warmup_questions): exactly 5 questions\n"
    '- Patterns ONLY: "Do you...?" / "Is it...?" / "Can you...?"\n'
    "- Must be personal and concrete.\n"
    '- Forbidden: "Have you ever", abstract topics.\n\n'
    "CORE QUESTIONS (core_questions): exactly 4 questions\n"
    "- Why and How questions are allowed.\n"
    "- BUT: must stay concrete, simple, and directly tied to the source situation.\n"
    "- Must relate directly to topic.\n"
    "- Do NOT move into abstract reasoning or theory.\n"
    "- Simple cause-effect is allowed (because, so, helps, makes).\n"
    "- Why and How questions must be about personal experience or simple real-life situations.\n"
    "- Do NOT generate scientific, academic, or theoretical questions.\n"
    "- Wrong: 'What are the phases of hair growth cycle?'\n"
    "- Wrong: 'How does genetics affect hair loss?'\n"
    "- Right: 'Do you know why some people lose hair?'\n"
    "- Right: 'Why is hair loss a problem for some people?'\n"
    "- Questions must be answerable from personal experience, not from scientific knowledge.\n"
    "- Questions must focus on the PERSON, not on general explanations.\n"
    "- Do NOT ask:\n"
    "- how something works in general\n"
    "- why something happens in general\n"
    "- medical or scientific causes\n"
    "- anything requiring specialized knowledge\n"
    "- Instead ask about:\n"
    "- what people feel\n"
    "- what people do\n"
    "- what people think\n"
    "- what happens in their life\n"
    "- Shift from explanation to experience.\n"
    "- Wrong: 'How does chemotherapy relate to hair loss?'\n"
    "- Wrong: 'Why does stress cause hair loss?'\n"
    "- Right: 'How do you feel when you see someone losing hair?'\n"
    "- Right: 'Do you know someone who worries about hair loss?'\n\n"
    "THIS OR THAT (choices): exactly 4 items\n"
    '- Format ONLY: "X or Y?" (question mark at end).\n'
    "- May reflect simple contrast from the source.\n\n"
    "VOCABULARY (support_words): exactly 6 items\n"
    '- Format each string: "English — Ukrainian".\n'
    "- Slightly wider than A1 but still practical.\n"
    "- Must be directly related to topic.\n"
    "- Vocabulary must stay everyday and student-usable.\n"
    "- Do NOT include scientific or academic terms.\n"
    "- All English vocabulary words must be lowercase.\n"
    "FORBIDDEN vocabulary for A2:\n"
    "- scientific terms (genetics, cycle, pattern, hormone)\n"
    "- academic words (intelligence, mechanism, process)\n"
    "- medical terms\n\n"
    "ALLOWED vocabulary for A2:\n"
    "- words a student can use in real conversation\n"
    "- words that describe visible, personal situations\n"
    "- simple everyday nouns, verbs, and adjectives only\n\n"
    "Test: Can an A2 student use this word when talking to a friend?\n"
    "If NO → remove it.\n\n"
    "If the filtered source has limited detail, derive items from the topic and scenes while staying consistent with them.\n"
    "Do not include key_ideas, discussion_questions, or vocabulary_items."
)

_PREVIEW_A2_FILTER_SYSTEM = (
    "You follow the user instructions exactly. Output ONE JSON object only, no markdown."
)

_PREVIEW_A2_FILTER_USER = (
    "Filter this content for A2 English students.\n\n"
    "Output ONLY:\n"
    "- 1 simplified topic (same situation as source, not replaced)\n"
    "- 3–5 scenes or cause-effect points from the source\n\n"
    "Rules:\n"
    "- Stay semantically close to the original source\n"
    "- Do NOT replace topic with generic daily routine\n"
    "- Do NOT invent context that does not exist in source\n"
    "- Topic may include simple cause-effect (why/how)\n"
    "- Language must be simple and concrete\n"
    "- If topic cannot be simplified → keep closest real-life version\n"
    "- NEVER switch to unrelated situations\n\n"
    'Return one JSON object with keys "topic" (string) and "scenes" (array of 3 to 5 strings). '
    "Each scene or cause-effect point is one short English line. No extra keys."
)

_PREVIEW_SYSTEM_LESSON_B1 = (
    "You are a helpful teacher. Output ONE JSON object only, no markdown.\n"
    "Use the B1 FILTERED SOURCE in the user message ONLY as the source (no invented facts beyond it). "
    "It is a filtered topic and scenes — not a raw transcript; do not use or assume any other text.\n"
    "This preview is for CEFR level B1 only. Apply ONLY these rules for lesson + B1.\n\n"
    "GLOBAL:\n"
    "- Every element must reflect the main situation and core meaning from the filtered source.\n"
    "- Do NOT introduce unrelated elements.\n"
    "- Do NOT add generic filler.\n"
    "- Follow exact counts strictly (no more, no less).\n"
    "- Do not regenerate existing content; only extend if asked (initial generation: fill all fields from source).\n"
    '- No placeholders like "—".\n'
    "- No empty fields.\n"
    "- Simplify the language, but allow deeper meaning.\n"
    "- Do NOT turn the lesson into an essay or discussion club.\n\n"
    'Return ONLY these keys for a lesson preview:\n'
    '- "topic": one short line (teacher-friendly), same situation as the source (simplified, not replaced)\n'
    '- "warmup_questions": exactly 5 questions\n'
    '- "core_questions": exactly 4 questions\n'
    '- "choices": exactly 4 items\n'
    '- "support_words": exactly 6 items (each English word — Ukrainian translation)\n\n'
    "WARM-UP (warmup_questions): exactly 5 questions\n"
    '- Patterns: "Do you...?" / "Is it...?" / "Can you...?" / "Have you...?"\n'
    "- Personal, relatable, real-life.\n"
    "- May include past or present experience.\n"
    "- No abstract or academic phrasing.\n\n"
    "CORE QUESTIONS (core_questions): exactly 4 questions\n"
    "- Must include opinions, reasons, or simple arguments.\n"
    "- Allow: Why / How questions with explanations.\n"
    "- Allow: present, past, future references.\n"
    "- Focus on: personal experience, opinion, real-life situations.\n"
    "- Questions must stay connected to the student's life or realistic human situations.\n"
    "- Allowed: because, so, I think, I believe.\n"
    "- FORBIDDEN: academic, scientific, theoretical explanations.\n"
    "- Do NOT move into abstract debate.\n"
    "FORBIDDEN phrases in questions:\n"
    "- genetic factors, biological mechanism, hormonal process\n"
    "- any phrase that sounds like a textbook or lecture\n"
    "Replace with personal versions:\n"
    "- Wrong: 'How do genetic factors affect hair loss?'\n"
    "- Right: 'Do you think hair loss runs in families?'\n"
    "CRITICAL CHECK before finalizing core questions:\n"
    "Questions must focus on:\n"
    "- the student\n"
    "- people in real life\n"
    "- personal opinion, experience, or everyday situations\n"
    "If a question is about scientists, researchers, society, systems, or general facts → REMOVE IT.\n"
    "Replace it with a personal or real-life version.\n"
    "- Wrong: 'Why is it important for scientists...'\n"
    "- Right: 'Would you try something new to keep your hair?'\n\n"
    "THIS OR THAT (choices): exactly 4 items\n"
    '- Format ONLY: "X or Y?" (question mark at end).\n'
    "- May reflect contrast, preference, or opinion.\n"
    "- Must stay relevant to topic.\n"
    "FORBIDDEN in This or That for B1:\n"
    "- scientific phases or cycles\n"
    "- medical procedures\n"
    "- technical processes\n"
    "FORBIDDEN words in This or That options:\n"
    "- treatment, treatments\n"
    "- medical, medicine\n"
    "- remedy, remedies (when used in medical sense)\n"
    "Replace with:\n"
    "- help\n"
    "- care\n"
    "- ways\n"
    "- things people do\n"
    "CRITICAL CHECK for This or That:\n"
    "Both options must be things a regular person can choose in real life.\n"
    "No medical procedures, no expert solutions.\n\n"
    "VOCABULARY (support_words): exactly 6 items\n"
    '- Format each string: English word — Ukrainian translation (same as "word — переклад").\n'
    "- Everyday, practical, usable in conversation.\n"
    "- Allowed: simple abstract words (problem, reason, choice, result).\n"
    "- FORBIDDEN: scientific, academic, technical terms.\n"
    "- All English words must be lowercase.\n"
    "FORBIDDEN in vocabulary for B1:\n"
    "- research, genetic, hormonal, biological, mechanism, phase, cycle\n"
    "- any word requiring scientific background to understand\n"
    "FORBIDDEN words:\n"
    "- treatment, treatments\n"
    "- medical, medicine\n"
    "- remedy, remedies (when used in medical sense)\n"
    "Replace with:\n"
    "- help\n"
    "- care\n"
    "- ways\n"
    "- things people do\n\n"
    "Test: Can a B1 student use this word when talking to a friend about this topic?\n"
    "If NO → remove it.\n\n"
    "If the filtered source has limited detail, derive items from the topic and scenes while staying consistent with them.\n"
    "Do not include key_ideas, discussion_questions, or vocabulary_items."
)

_PREVIEW_B1_FILTER_SYSTEM = (
    "You follow the user instructions exactly. Output ONE JSON object only, no markdown."
)

_PREVIEW_B1_FILTER_USER = (
    "Filter this content for B1 English students.\n\n"
    "Output ONLY:\n"
    "- 1 simplified topic (same situation as source, not replaced)\n"
    "- 3–5 scenes, cause-effect points, or reasoning from source\n\n"
    "Rules:\n"
    "- Stay semantically close to the original source\n"
    "- Do NOT replace topic with generic scenario\n"
    "- Do NOT invent context that does not exist in source\n"
    "- Topic may include simple abstract ideas and cause-effect\n"
    "- Language must remain clear and understandable\n"
    "- If topic cannot be simplified → keep closest real-life version\n"
    "- NEVER switch to unrelated topics\n\n"
    'Return one JSON object with keys "topic" (string) and "scenes" (array of 3 to 5 strings). '
    "Each line is one short English scene, cause-effect point, or reasoning note. No extra keys."
)

_B2_VOCABULARY_FORBIDDEN_TERMS = (
    "research, study, clinical, trial, phase, mechanism, hormone, genetic, genetics, "
    "biological, treatment, treatments, diagnosis, symptom, symptoms, patient, therapy, therapies, "
    "dosage, dose, molecule, molecules, cell, cells, protein, proteins, enzyme, enzymes, "
    "receptor, receptors, pathway, pathways, inflammation, metabolism, immune system, nervous system, "
    "cardiovascular, dermatology, trichology, alopecia, androgen, androgens, follicle, follicles, "
    "miniaturization, stem cell, stem cells, regeneration, nanotechnology, biotechnology, "
    "pharmaceutical, pharmaceuticals, FDA-approved, peer-reviewed, hypothesis, hypotheses, "
    "correlation, causation, longitudinal study, double-blind, placebo, placebos, control group, "
    "statistical significance, p-value, abstract, methodology, results section, conclusion, "
    "limitations, future research, advancement, advancements, scientific breakthrough, clinical study, "
    "medical study, research paper, peer review, evidence-based, empirical, quantitative, qualitative, "
    "longitudinal, cross-sectional, meta-analysis, systematic review, hypothesis testing, "
    "control variable, dependent variable, independent variable, placebo effect, double-blind trial, "
    "randomized controlled trial, RCT, phase I, phase II, phase III, phase IV, FDA approval, "
    "off-label, contraindication, side effect profile, pharmacokinetics, pharmacodynamics, "
    "bioavailability, half-life, dosage form, titration, tapering, withdrawal, remission, "
    "exacerbation, prognosis, pathophysiology, etiology, idiopathic, congenital, hereditary, "
    "heritability, polygenic, multifactorial, gene expression, transcription, translation, mutation, "
    "allele, chromosome, karyotype, genotype, phenotype, epigenetics, DNA methylation, "
    "histone modification, RNA interference, CRISPR, gene therapy, stem cell therapy, "
    "regenerative medicine, tissue engineering, scaffold, biomaterial, biocompatibility, "
    "xenograft, allograft, autograft, micrograft, follicular unit extraction, FUE, "
    "follicular unit transplantation, FUT, strip surgery, scalp reduction, flap surgery, "
    "tissue expansion, laser therapy, LLLT, photobiomodulation, PRP, platelet-rich plasma, "
    "mesotherapy, microneedling, derma roller, stem cell serum, growth factor, cytokine, chemokine, "
    "interleukin, tumor necrosis factor, vascular endothelial growth factor, VEGF, "
    "fibroblast growth factor, FGF, insulin-like growth factor, IGF, transforming growth factor beta, "
    "TGF-beta, Wnt signaling, Hedgehog signaling, Notch signaling, Sonic hedgehog, Shh pathway, "
    "beta-catenin, DKK1, BMP, bone morphogenetic protein, DHT, dihydrotestosterone, "
    "5-alpha reductase, finasteride, dutasteride, minoxidil, ketoconazole, spironolactone, "
    "cyproterone acetate, flutamide, bicalutamide, oral contraceptives, anti-androgens, "
    "estrogen therapy, progesterone, testosterone, cortisol, corticosteroids, "
    "topical corticosteroids, intralesional corticosteroids, anthralin, tretinoin, adapalene, "
    "benzoyl peroxide, salicylic acid, glycolic acid, lactic acid, mandelic acid, kojic acid, "
    "hydroquinone, arbutin, niacinamide, panthenol, biotin, collagen, keratin, elastin, ceramide, "
    "hyaluronic acid, peptides, amino acids, vitamins, minerals, supplements, nutraceuticals, "
    "cosmeceuticals, functional foods, superfoods, adaptogens, nootropics, smart drugs, "
    "cognitive enhancers, mood stabilizers, antidepressants, anxiolytics, antipsychotics, "
    "benzodiazepines, SSRIs, SNRIs, MAOIs, tricyclics, lithium, valproate, carbamazepine, "
    "lamotrigine, gabapentin, pregabalin, topiramate, opioid, opioid analgesic, NSAID, "
    "acetaminophen, aspirin, ibuprofen, naproxen, celecoxib, meloxicam, diclofenac, indomethacin, "
    "ketorolac, tramadol, codeine, morphine, fentanyl, oxycodone, hydrocodone, hydromorphone, "
    "oxymorphone, methadone, buprenorphine, naloxone, naltrexone, acamprosate, disulfiram, "
    "modafinil, armodafinil, methylphenidate, amphetamine, dextroamphetamine, lisdexamfetamine, "
    "atomoxetine, guanfacine, clonidine, propranolol, atenolol, metoprolol, carvedilol, labetalol, "
    "nicardipine, nifedipine, amlodipine, diltiazem, verapamil, hydralazine, minoxidil (oral), "
    "spironolactone (hair), cyproterone, flutamide, bicalutamide, nilutamide, enzalutamide, "
    "apalutamide, darolutamide, abiraterone, prednisone, dexamethasone, hydrocortisone, "
    "methylprednisolone, triamcinolone, betamethasone, clobetasol, mometasone, fluticasone, "
    "budesonide, beclomethasone, ciclesonide, halobetasol, halcinonide, amcinonide, desonide, "
    "desoximetasone, diflorasone, fluocinolone, fluocinonide, flurandrenolide, halcinonide, "
    "hydrocortisone butyrate, hydrocortisone probutate, hydrocortisone valerate, mometasone furoate, "
    "prednicarbate, ulobetasol"
)

_PREVIEW_SYSTEM_LESSON_B2 = (
    "You are a helpful teacher. Output ONE JSON object only, no markdown.\n"
    "Use the B2 FILTERED SOURCE in the user message ONLY as the source (no invented facts beyond it). "
    "It is a filtered topic and scenes — not a raw transcript; do not use or assume any other text.\n"
    "This preview is for CEFR level B2 only. Apply ONLY these rules for lesson + B2.\n\n"
    "GLOBAL:\n"
    "- Every element must reflect the main situation and core meaning from the filtered source.\n"
    "- Do NOT introduce unrelated elements.\n"
    "- Do NOT add generic filler.\n"
    "- Follow exact counts strictly (no more, no less).\n"
    "- Do not regenerate existing content; only extend if asked (initial generation: fill all fields from source).\n"
    '- No placeholders like "—".\n'
    "- No empty fields.\n"
    "- Simplify the language, but allow deeper meaning.\n"
    "- Do NOT turn the lesson into an essay or lecture.\n"
    "Global forbidden rule for ALL elements in B2:\n"
    "- natural remedies\n"
    "- genetics\n"
    "- hormones\n"
    "- medications\n"
    "These words must never appear in any block (topic, warm-up, core, choices, vocabulary).\n\n"
    'Return ONLY these keys for a lesson preview:\n'
    '- "topic": one short line (teacher-friendly), same situation as the source (not replaced)\n'
    '- "warmup_questions": exactly 5 questions\n'
    '- "core_questions": exactly 4 questions\n'
    '- "choices": exactly 4 items\n'
    '- "support_words": exactly 6 items (each "word — переклад": English — Ukrainian)\n\n'
    "TOPIC (topic line):\n"
    "The topic must sound like a real-life situation or problem people talk about — "
    "NOT like a documentary title, science headline, or academic chapter name.\n"
    "FORBIDDEN in topic name:\n"
    '- "The Science of..." (or similar)\n'
    "- Any topic starting with academic framing\n"
    "FORBIDDEN topic patterns:\n"
    '- "Understanding..."\n'
    '- "Exploring..."\n'
    '- "The Future of..."\n'
    '- "Breakthroughs in..."\n'
    '- "How X Works" (unless clearly casual and human-centered)\n'
    "Replace with simple, human phrasing.\n"
    "- Wrong: 'The Science of Hair Loss'\n"
    "- Right: 'Worries about losing hair'\n\n"
    "WARM-UP (warmup_questions): exactly 5 questions\n"
    '- Patterns: "Do you...?" / "Is it...?" / "Can you...?" / "Have you...?"\n'
    "- Personal, relatable, real-life.\n"
    "- May include past, present, or future experience.\n"
    "- May include simple opinions.\n"
    "- No academic or theoretical phrasing.\n\n"
    "CORE QUESTIONS (core_questions): exactly 4 questions\n"
    "- Must include opinions, reasons, and simple arguments.\n"
    "- Must allow defending a point of view.\n"
    "- Allow: Why / How questions with explanation and comparison.\n"
    "- Allow: present, past, future references.\n"
    "- Focus on: personal experience, opinion, real-life situations, and critical thinking.\n"
    "- Allow: because, so, I think, I believe, I agree, I disagree.\n"
    "- May include comparison of ideas or choices.\n"
    "- Questions must stay connected to the student's life or realistic human situations.\n"
    "- Do NOT move into abstract theory, academic analysis, or lecture style.\n"
    "- Do NOT require scientific or specialized knowledge.\n"
    "FORBIDDEN in core questions:\n"
    "- \"advancements in science\"\n"
    "- \"scientific solutions\"\n"
    "- Any phrase requiring expert knowledge\n"
    "Replace with personal, real-life versions.\n"
    "FORBIDDEN in core questions and in This or That:\n"
    "- medications, medication\n"
    "- remedies, remedy\n"
    "- hair treatments, treatment\n"
    "- investing in hair\n"
    "Replace with everyday language:\n"
    "- things people do → instead of treatments\n"
    "- ways to help → instead of remedies\n"
    "- products → only if in casual context\n\n"
    "THIS OR THAT (choices): exactly 4 items\n"
    '- Format ONLY: "X or Y?" (question mark at end).\n'
    "- May reflect contrast, preference, or opinion.\n"
    "- May include more nuanced or complex choices.\n"
    "- Both options must be realistic and relatable.\n"
    "- No academic, medical, or expert-only concepts.\n"
    "FORBIDDEN in This or That:\n"
    "- medical pattern names (e.g. male pattern baldness)\n"
    "- scientific comparisons\n"
    "- natural remedies\n"
    "- genetics\n"
    "- hormones\n"
    "- medications\n\n"
    "VOCABULARY (support_words): exactly 6 items\n"
    "- Everyday, practical, usable in conversation.\n"
    "- May include simple abstract words (choice, result, opinion, reason, effect, change).\n"
    "- Must be directly related to topic.\n"
    + (
        "- FORBIDDEN: scientific, academic, technical terms.\n"
        "FORBIDDEN in vocabulary:\n- "
        + _B2_VOCABULARY_FORBIDDEN_TERMS
        + "\n\nALLOWED vocabulary for B2:\n"
        "- words about feelings, thoughts, habits, daily life, choices, problems, worries, hopes, opinions, comparisons, simple explanations\n\n"
        "Test: Can a B2 student use this word when talking to a friend about this topic?\n"
        "If NO → remove it.\n"
        "- All English words must be lowercase.\n"
        '- Format each string: English word — Ukrainian translation ("word — переклад").\n\n'
        "If the filtered source has limited detail, derive items from the topic and scenes while staying consistent with them.\n"
        "Do not include key_ideas, discussion_questions, or vocabulary_items."
    )
)

_PREVIEW_B2_FILTER_SYSTEM = (
    "You follow the user instructions exactly. Output ONE JSON object only, no markdown."
)

_PREVIEW_B2_FILTER_USER = (
    "Filter this content for B2 English students.\n\n"
    "Output ONLY:\n"
    "- 1 topic (same situation as source, not replaced)\n"
    "- 3–5 scenes, cause-effect points, or nuanced reasoning from source\n\n"
    "Rules:\n"
    "- Stay semantically close to the original source\n"
    "- Do NOT replace topic with generic scenario\n"
    "- Do NOT invent context that does not exist in source\n"
    "- Topic may include nuanced ideas, perspectives, and real-life complexity\n"
    "- Language must remain clear and understandable\n"
    "- If topic cannot be simplified → keep closest real-life version\n"
    "- NEVER switch to unrelated topics\n"
    "The topic line must sound like a real-life situation or problem people talk about — "
    "NOT like a documentary title, science headline, or academic chapter name.\n"
    "FORBIDDEN topic patterns:\n"
    '- "The Science of..."\n'
    '- "Understanding..."\n'
    '- "Exploring..."\n'
    '- "The Future of..."\n'
    '- "Breakthroughs in..."\n'
    '- "How X Works" (unless clearly casual and human-centered)\n'
    "Replace with simple, human phrasing.\n\n"
    'Return one JSON object with keys "topic" (string) and "scenes" (array of 3 to 5 strings). '
    "Each line is one short English scene, cause-effect point, or nuanced reasoning note. No extra keys."
)

_PREVIEW_A1_FILTER_SYSTEM = (
    "You follow the user instructions exactly. Output ONE JSON object only, no markdown."
)

_PREVIEW_A1_FILTER_USER = (
    "Filter this content for A1 English students.\n\n"
    "Output ONLY:\n"
    "- 1 simplified daily-life topic (concrete situation)\n"
    "- 3–5 daily-life scenes or actions from the source\n\n"
    "Rules:\n"
    "- No abstract or mental concepts\n"
    "- Do not copy abstract or complex wording from the source\n"
    "- No psychology, theory, or emotions explanations\n"
    "- Use only: actions, places, times\n"
    "  (talk, go, eat / home, school, work / morning, evening)\n"
    "- Keep only concrete, visible, real-life situations\n"
    "- Each scene must be a simple physical action or situation\n"
    "- Output must be something an A1 student\n"
    "  can imagine, answer, and talk about\n"
    "- Scenes are better than ideas\n\n"
    "CRITICAL: Do NOT replace the topic with a more common or easier scenario.\n"
    "Do NOT generalize.\n"
    "Do NOT invent context that does not exist in the source "
    "(e.g. gym, routine, lifestyle).\n\n"
    "You must simplify the SAME situation from the source.\n"
    "Not a different situation.\n\n"
    "Wrong: source about muscles → output 'exercising at gym'\n"
    "Right: source about muscles → output 'muscles get tired and grow'\n\n"
    "Stay strictly inside the meaning of the source.\n"
    "Simplify the language, not the topic.\n\n"
    "The simplified topic MUST stay semantically close to the original source.\n"
    "Do NOT replace the topic with a generic daily routine.\n"
    "Do NOT invent a new situation unrelated to the source.\n"
    "You must simplify the SAME situation, not change it.\n\n"
    "If the topic cannot be simplified into A1-level daily-life scenes without losing meaning:\n"
    "- keep only the closest possible real-life version\n"
    "- do NOT switch to unrelated topics\n\n"
    "Example:\n"
    "Source topic: teenagers not getting enough sleep\n"
    "A1 version: going to bed late and feeling tired\n"
    "NOT: morning routine or waking up\n\n"
    "All questions, choices, and vocabulary must reflect the main situation AND the main problem from the source.\n"
    "For A1, keep the core difficulty or feeling in a simple, concrete form.\n\n"
    "Example:\n"
    "Source: teenagers not getting enough sleep\n"
    "A1 focus: tired, sleep, wake up early, want to sleep\n"
    "Questions: Do you feel tired in the morning?\n"
    "           Is it hard to wake up for school?\n"
    "NOT: neutral morning routine without the core problem\n\n"
    "Do NOT shift to generic daily routine if the source has a clear main problem.\n\n"
    "If no usable daily-life scenes are found in the source:\n"
    "- ignore the transcript completely\n"
    "- use only the simplified topic\n"
    "- generate 3–5 basic daily-life scenes that\n"
    "  an A1 student can imagine and talk about\n\n"
    "Example output:\n"
    "Topic: talking to yourself at home\n"
    "Scenes:\n"
    "- you talk before sleep\n"
    "- you say words in the mirror\n"
    "- you repeat things you need to do\n\n"
    "Output only the simplified topic and scenes.\n"
    "Nothing else.\n\n"
    'Return one JSON object with keys "topic" (string) and "scenes" (array of 3 to 5 strings). '
    "Each scene is one short English line. No extra keys."
)

_PREVIEW_SYSTEM_QUESTIONS = (
    "You are a helpful teacher. Output ONE JSON object only, no markdown.\n"
    "Use the transcript below ONLY as the source (no invented facts).\n"
    'Return ONLY these keys for a speaking / discussion preview:\n'
    '- "topic": one short line (teacher-friendly)\n'
    '- "discussion_questions": 3 to 5 short discussion questions grounded in the source\n'
    "Do not include key_ideas, words, warmup_questions, vocabulary_items, or grammar_patterns."
)

_PREVIEW_SPEAKING_FILTER_SYSTEM = (
    "You follow the user instructions exactly. Output ONE JSON object only, no markdown."
)

_PREVIEW_SPEAKING_FILTER_USER = (
    "Filter this content for speaking practice.\n\n"
    "Output ONLY:\n"
    "- 1 clear topic (from source, no replacement)\n"
    "- 3–5 key situations or angles from source\n\n"
    "Rules:\n"
    "- Stay semantically close to the original source\n"
    "- Do NOT replace topic with generic scenario\n"
    "- Do NOT invent unrelated context\n"
    "- Focus on real-life situations and experiences\n"
    "- If topic is complex → simplify to real-life version\n"
    "- NEVER switch topic\n\n"
    'Return one JSON object with keys "topic" (string) and "scenes" (array of 3 to 5 strings). '
    "Each line is one short English situation or angle. No extra keys."
)

_SPEAKING_GLOBAL_FORBIDDEN_RULES = (
    "GLOBAL FORBIDDEN (applies to ALL speaking levels: A1, A2, B1, B2 — individual level rules still apply on top):\n\n"
    "FORBIDDEN vocabulary:\n"
    "- treatments, treatment\n"
    "- medical, medicine, medication\n"
    "- remedy, remedies\n"
    "- genetics, genetic\n"
    "- hormones, hormonal\n"
    "- research, researchers\n"
    "- scientific, scientist\n\n"
    "FORBIDDEN patterns:\n"
    "- meta-questions:\n"
    '  - "Can you talk about..."\n'
    '  - "Can you describe..."\n'
    '  - "Can you share about..."\n'
    "- academic framing in topic name\n"
    "- questions requiring expert knowledge\n"
    "- questions not answerable from personal experience\n"
    "- trivial or obvious questions\n"
    "- questions answerable with only yes/no\n"
    "- generic speaking tasks not connected to topic\n\n"
    "CONTEXTUAL RESTRICTIONS:\n"
    '- "causes" allowed ONLY in simple everyday context '
    '(e.g. "What causes you stress?") NOT in scientific/explanatory form\n'
    '- "temporary" / "permanent" allowed ONLY in everyday context NOT in medical/scientific meaning\n\n'
)

_SPEAKING_GLOBAL_HARD_CONSTRAINTS_LINE = (
    "- GLOBAL (all speaking levels A1–B2): same forbidden vocabulary, forbidden patterns, and contextual "
    "restrictions as in GLOBAL FORBIDDEN above (treatments, medical, medication, remedy, genetics, hormones, "
    "research, scientific; meta talk/describe/share; academic topic; expert-only; non-experience; trivial; "
    "yes/no only; generic task; causes / temporary / permanent — everyday only).\n"
)

_PREVIEW_SYSTEM_SPEAKING_A1 = (
    "You are a helpful teacher. Output ONE JSON object only, no markdown.\n"
    "Use the SPEAKING FILTERED SOURCE in the user message ONLY as the source (no invented facts beyond it). "
    "It is a filtered topic and key situations — not a raw transcript; do not use or assume any other text.\n"
    "This preview is CEFR A1 speaking only. Apply ONLY these STRICT rules.\n\n"
    + _SPEAKING_GLOBAL_FORBIDDEN_RULES
    + "A1 QUESTIONS QUALITY:\n\n"
    "- Questions must be simple BUT meaningful.\n"
    "- A1 does not mean trivial or obvious.\n\n"
    "A1 SPEAKING STRUCTURE:\n\n"
    "- Topic (very simple, daily-life)\n"
    "- Discussion questions: exactly 6\n"
    "- Speaking task: exactly 1\n\n"
    "TOPIC SIMPLIFICATION:\n\n"
    "- Topic must use ONLY A1 words\n"
    "- Replace:\n"
    '  - "hair growth" → "hair"\n'
    '  - "hair loss" → avoid or simplify (e.g. worry about hair, thin hair)\n'
    "- Keep topic short and simple: prefer 1–3 words; never more than 4 English words\n"
    "- Must describe a daily-life situation\n"
    "- No abstract or complex phrasing\n\n"
    "DISCUSSION QUESTIONS RULES:\n\n"
    "- Exactly 6 questions\n"
    "- Max 8 words per question\n"
    "- Questions must be about real personal situations\n"
    "- Questions must have a real answer worth sharing\n"
    "- Questions must sound natural in real conversation\n"
    "- Use ONLY simple patterns:\n"
    "  - Do you...?\n"
    "  - Have you...?\n"
    "  - Can you...? (simple ability only — NOT meta-instruction; see FORBIDDEN QUESTION PATTERNS)\n"
    "  - Do you like...?\n\n"
    "FORBIDDEN QUESTION PATTERNS (meta-questions — not real speaking questions):\n\n"
    '- "Can you talk about..."\n'
    '- "Can you share about..."\n'
    '- "Can you describe..."\n\n'
    "REPLACEMENT RULE:\n\n"
    "- Replace meta-questions with specific personal questions\n"
    "- Questions must be about:\n"
    "  - personal life\n"
    "  - real situations\n"
    "  - simple experience\n\n"
    "EXAMPLES:\n\n"
    'Wrong: "Can you talk about hair loss?"\n\n'
    'Right: "Do you worry about losing hair?"\n'
    'Right: "Do you know someone who is bald?"\n'
    'Right: "Do you like your hair?"\n\n'
    "FORBIDDEN QUESTIONS:\n\n"
    "- Questions with obvious yes/no answers "
    '(e.g. "Do you have hair?" "Do you have eyes?")\n'
    "- Questions that are trivial or meaningless\n"
    "- Questions that feel unnatural or robotic\n"
    "- Questions that require specialist knowledge or long explanation\n"
    "- why / how question forms\n"
    "- abstract ideas, scientific or medical jargon\n"
    "- complex grammar structures\n\n"
    "GOOD EXAMPLES FOR A1:\n\n"
    '- "Do you like short or long hair?"\n'
    '- "Do you know a bald person?"\n'
    '- "Do you use products for your hair?"\n'
    '- "Have you changed your hairstyle before?"\n\n'
    "SIMPLICITY CONTROL:\n\n"
    "- Use only basic vocabulary\n"
    "- Keep grammar simple\n"
    "- Avoid complex or rare words\n"
    "- If a beginner cannot understand instantly → rewrite\n\n"
    "FORBIDDEN VOCABULARY:\n\n"
    "- treatments, treatment\n"
    "- any medical terms\n\n"
    "REWRITE RULE:\n"
    "- Replace complex or medical words with simple everyday alternatives\n"
    "- If you cannot simplify → remove the idea\n\n"
    "MEANING CONTROL:\n\n"
    "- Question must invite at least a short answer (not just yes/no)\n"
    "- Prefer questions that allow:\n"
    "  - a small explanation\n"
    "  - an example\n"
    "  - personal detail\n\n"
    "SPEAKING TASK SPECIFICITY:\n\n"
    "- Exactly 1 speaking_task\n"
    "- Must be specific to the topic from the source — NOT generic\n"
    "- Must connect directly to the source topic\n"
    "- Must refer to a real situation from the filtered source\n\n"
    "FORBIDDEN (speaking_task):\n\n"
    '- Generic tasks that fit almost any topic, e.g. "Talk about your hair." "Describe your day."\n\n'
    "ALLOWED:\n\n"
    "- Task must connect directly to the source topic\n"
    "- Must refer to a real situation\n\n"
    "GOOD EXAMPLES (topic-dependent; adapt to source):\n\n"
    '- "Talk about someone you know who is bald."\n'
    '- "Talk about a time you changed your hairstyle."\n'
    '- "Describe a person with short hair you know."\n\n'
    "ADDITIONAL SAFEGUARD (speaking_task):\n\n"
    "- Use ONLY CEFR A1 vocabulary\n"
    "- Must be easy to understand instantly\n"
    "- Must not require explanation or complex thinking\n"
    "- Must be doable with basic grammar (no heavy past structures)\n\n"
    "ADDITIONAL SAFEGUARD (A1 LEVEL):\n\n"
    "- Avoid adding complex ideas when rewriting\n"
    '- Do NOT introduce "why" questions that require explanation\n'
    "- Keep vocabulary simple and familiar\n"
    "- Keep sentence structure basic\n\n"
    "FINAL A1 TOPIC + VOCAB CONSISTENCY RULE:\n\n"
    "QUESTION CONSISTENCY:\n"
    "- Every question must stay connected to the topic\n"
    "- Do NOT introduce unrelated ideas (e.g. stress)\n\n"
    "VOCABULARY LIMIT:\n"
    "- Use ONLY basic everyday words\n"
    "- Avoid:\n"
    "  - growth\n"
    "  - loss\n"
    "  - stress (if not essential and simple)\n\n"
    "REWRITE RULE:\n"
    "- If a word is above A1 → replace or remove\n"
    "- If a question is not directly about the topic → remove or rewrite\n\n"
    "GLOBAL RULES:\n\n"
    "- Everything must be simple, clear, and real-life\n"
    "- No academic or abstract language\n"
    "- No complex vocabulary\n"
    "- No multi-step thinking\n\n"
    "CRITICAL CHECK:\n\n"
    "- Does every question match the topic?\n"
    "- Does every word belong to A1 vocabulary?\n"
    "- Can a beginner understand instantly?\n"
    "If NO to any → rewrite\n\n"
    "- If a question is longer than 8 words → shorten\n"
    '- If the answer would be only "yes" or "no" → rewrite\n'
    "- If a question sounds unnatural → rewrite\n"
    "- If a question is too obvious → replace\n"
    "- If it sounds like a textbook → simplify\n"
    "- If a question sounds like an instruction (meta: talk/share/describe about...) → rewrite\n"
    "- If a question is not about real life → rewrite\n"
    "- If a question requires thinking → simplify\n"
    "- If speaking_task could apply to ANY topic → rewrite (must be topic-specific)\n"
    "- If speaking_task does not clearly connect to the topic → rewrite\n"
    "- If vocabulary is above A1 or uses forbidden words → simplify\n\n"
    "GOAL:\n"
    "Fully consistent, simple, topic-based speaking with zero confusion for the A1 student\n\n"
    'Return ONLY these keys:\n'
    '- "topic": one short line (A1 words only; prefer 1–3 words; max 4 English words; daily-life only; '
    "e.g. simplify hair growth → hair; avoid or simplify hair loss)\n"
    '- "discussion_questions": exactly 6 strings (on-topic only; allowed patterns only; max 8 words each; '
    "meaningful, not trivial; no treatment/medical wording; avoid growth, loss, stress unless essential; "
    "no meta-questions: Can you talk/share/describe about...)\n"
    '- "speaking_task": exactly 1 string (A1 vocabulary; specific to this topic; not generic)\n'
    "Do not include key_ideas, warmup_questions, vocabulary_items, grammar_patterns, choices, or support_words."
)

_PREVIEW_SYSTEM_SPEAKING_A2 = (
    "You are a helpful teacher. Output ONE JSON object only, no markdown.\n"
    "Use the SPEAKING FILTERED SOURCE in the user message ONLY as the source (no invented facts beyond it). "
    "It is a filtered topic and key situations — not a raw transcript; do not use or assume any other text.\n"
    "This preview is CEFR A2 speaking only. Apply ONLY these rules.\n\n"
    + _SPEAKING_GLOBAL_FORBIDDEN_RULES
    + "STRUCTURE:\n\n"
    "- Topic (simple, natural, max 5 words, from source)\n"
    "- Discussion questions: exactly 6\n"
    "- Speaking task: exactly 1\n\n"
    "A2 = EXTENSION OF A1:\n\n"
    "- Keep ALL CEFR A1 speaking rules as the base (on-topic questions; no meta-questions; "
    "no trivial or obvious meaningless questions; no treatment/medical jargon where A1 forbids it; "
    "topic consistency; no unrelated ideas; etc.)\n"
    "- Add for A2:\n"
    "  - simple reasons (because)\n"
    "  - simple experiences (past)\n"
    "  - short explanations (student may answer in 1–3 sentences)\n\n"
    "QUESTIONS RULES:\n\n"
    "- Allowed forms:\n"
    "  - Do you...?\n"
    "  - Have you...?\n"
    "  - Can you...? (NOT meta-instruction — see FORBIDDEN)\n"
    "  - What do you...?\n"
    "  - Why do you...?\n"
    "  - How do you feel about...?\n"
    "  - What do you usually do when...?\n\n"
    "- Max 10 words per question\n"
    "- Must be personal, real-life, concrete\n"
    "- Must require more than yes/no\n"
    "- Must allow a 1–3 sentence answer\n\n"
    "FORBIDDEN QUESTION PATTERNS FOR A2 (these push into B1 territory):\n\n"
    '- "What do you know about..."\n'
    '- "What do you think about people..."\n'
    '- "Why do you think people..."\n\n'
    "A2 questions must be ONLY about:\n\n"
    "- the student themselves\n"
    "- their personal experience\n"
    "- their direct observations\n\n"
    "EXAMPLES:\n\n"
    'Wrong: "What do you know about hair growth?"\n'
    'Right: "Have you noticed your hair changing?"\n\n'
    'Wrong: "What causes you stress in life?"\n'
    'Right: "Does hair loss worry you?"\n\n'
    "SEMANTIC DRIFT RULE:\n\n"
    "- Every question must stay connected to the source topic.\n"
    "- Do NOT introduce general life topics unless directly connected to the source.\n\n"
    "CRITICAL CONTROL:\n\n"
    "- Question must be answerable from:\n"
    "  - personal experience OR\n"
    "  - daily habits\n"
    "- If the student needs specialist or outside knowledge → reject that question (rewrite)\n"
    "- If the student can answer from life → accept\n\n"
    "FORBIDDEN IN QUESTIONS:\n\n"
    "- ALL CEFR A1 forbidden rules (including meta-questions: "
    '"Can you talk about...", "Can you describe...", "Can you share about...")\n'
    "- abstract reasoning\n"
    "- societal or general debates (not personal)\n"
    "- expert or outside knowledge\n"
    "- trivial, obvious, or meaningless questions\n\n"
    "TOPIC RULES:\n\n"
    "- Max 5 words\n"
    "- Must be simple, natural, based on source, connected to daily life\n"
    "- Forbidden: academic framing, complex wording\n\n"
    "SPEAKING TASK RULES:\n\n"
    "- Must be specific and connected to the topic\n"
    "- Must describe a real situation OR personal experience from the source topic\n"
    "- Must allow an answer of about 3–5 sentences\n\n"
    "FORBIDDEN TASKS:\n\n"
    '- Generic lines like "Talk about this topic"\n'
    "- Unclear or generic tasks\n"
    "- Tasks not connected to the topic\n\n"
    "CORE PRINCIPLE:\n"
    "A2 = A1 + simple reasons + simple experiences\n\n"
    'Return ONLY these keys:\n'
    '- "topic": one short line (max 5 English words; simple, natural; from source; daily-life)\n'
    '- "discussion_questions": exactly 6 strings (allowed forms; max 10 words each; ONLY about the student / '
    "their experience / direct observations; on source topic — no semantic drift; no A2-forbidden patterns)\n"
    '- "speaking_task": exactly 1 string (specific; on-topic; real situation or personal experience; ~3–5 sentence answer)\n'
    "Do not include key_ideas, warmup_questions, vocabulary_items, grammar_patterns, choices, or support_words."
)

_PREVIEW_SYSTEM_SPEAKING_B1 = (
    "You are a helpful teacher. Output ONE JSON object only, no markdown.\n"
    "Use the SPEAKING FILTERED SOURCE in the user message ONLY as the source (no invented facts beyond it). "
    "It is a filtered topic and key situations — not a raw transcript; do not use or assume any other text.\n"
    "This preview is CEFR B1 speaking only. Apply ONLY these rules.\n\n"
    + _SPEAKING_GLOBAL_FORBIDDEN_RULES
    + "STRUCTURE:\n\n"
    "- Topic (max 5 words, natural, from source)\n"
    "- Discussion questions: exactly 6\n"
    "- Speaking task: exactly 1\n\n"
    "B1 = EXTENSION OF A2:\n\n"
    "- Keep ALL CEFR A2 speaking rules as the base\n"
    "- Add:\n"
    "  - opinions\n"
    "  - reasons (because, so)\n"
    "  - simple arguments\n"
    "  - comparison\n\n"
    "B1 STUDENT CAN:\n\n"
    "- express opinions with reasons\n"
    "- talk about experiences in detail\n"
    "- compare situations\n"
    "- describe feelings and reactions\n"
    "- speak 3–5 sentences\n\n"
    "QUESTIONS RULES:\n\n"
    "- Allowed forms:\n"
    '  - "Why do you think...?"\n'
    '  - "How do you feel about...?"\n'
    '  - "Do you agree that...?"\n'
    '  - "What would you do if...?"\n'
    '  - "Which do you prefer... and why?"\n'
    '  - "Do you think...?"\n\n'
    "- Max 12 words per question\n"
    "- Must be:\n"
    "  - personal OR relatable\n"
    "  - connected to real-life situations\n"
    "  - based on source meaning\n\n"
    "- Must require:\n"
    "  - opinion\n"
    "  - reason (explicit or implied)\n\n"
    "CRITICAL CONTROL (IMPORTANT):\n\n"
    "- Questions MAY include general ideas BUT must stay within everyday understanding\n"
    "- Question must be answerable by:\n"
    "  - opinion\n"
    "  - personal experience\n"
    "  - simple reasoning\n"
    "- If question requires specialist knowledge → reject (rewrite)\n"
    '- If question requires explanation like a teacher → reject (rewrite)\n'
    '- If student can answer using "I think / I feel / because" → accept\n\n'
    "FORBIDDEN IN QUESTIONS:\n\n"
    "- ALL A1 + A2 speaking forbidden rules\n"
    '- knowledge-style: "What do you know about..."\n'
    "- expert or scientific explanations\n"
    "- abstract theory or philosophy\n"
    "- questions that sound like essays\n"
    "- questions not connected to topic\n\n"
    "FORBIDDEN QUESTION PATTERNS (B1):\n\n"
    '- "How do people..."\n'
    '- "Why do people..."\n'
    "- any questions about society or groups in general\n"
    '- "Do you agree that [scientific fact]...?"\n'
    "- questions that require confirming scientific claims\n\n"
    "EXAMPLES (scientific-agree pattern):\n\n"
    'Wrong: "Do you agree that stress can cause hair loss?"\n'
    'Right: "Do you think stress affects how you feel about your hair?"\n\n'
    "B1 questions must focus on:\n\n"
    "- the student personally\n"
    "- people they know\n"
    "- real situations they have seen\n\n"
    "NOT general world explanations.\n\n"
    "TOPIC RULES:\n\n"
    "- Max 5 words\n"
    "- Must be clear, natural, from source\n"
    "- Topic must NOT use academic framing.\n"
    'Wrong: "Social perceptions of baldness"\n'
    'Right: "How people feel about baldness"\n'
    "- Forbidden: academic framing, complex terminology\n\n"
    "SPEAKING TASK RULES:\n\n"
    "- speaking_task must be an INSTRUCTION for the student — what they must DO, not a sample answer or example text\n"
    "- Speaking task = the task to perform in speech; do NOT write model dialogue or sentences for the student to read\n"
    "- Must be specific and connected to topic\n"
    "- Must require:\n"
    "  - opinion + explanation OR\n"
    "  - experience + reflection\n"
    "- Student output: about 4–6 sentences when they speak (the task tells them what to do, not what to say word-for-word)\n\n"
    "Speaking task must be specific, not generic:\n\n"
    'Wrong: "Share your personal experience or opinion about..."\n'
    "Right must include a specific situation or angle (instructions only — not a model answer), e.g.:\n"
    '"Talk about someone you know who lost hair. How did they feel about it? '
    'Do you think it changed how they saw themselves?"\n\n'
    "FORBIDDEN TASKS:\n\n"
    "- generic tasks\n"
    "- abstract or theoretical tasks\n"
    "- tasks requiring expert knowledge\n"
    '- vague openers like "Share your personal experience or opinion about..." without a concrete angle\n\n'
    "CORE PRINCIPLE:\n"
    'B1 = A2 + opinion + "because"\n\n'
    'Return ONLY these keys:\n'
    '- "topic": one short line (max 5 English words; natural; from source; no academic framing — e.g. not like a lecture title)\n'
    '- "discussion_questions": exactly 6 strings (allowed forms; max 12 words each; opinion + reason; on-topic; source-based; '
    "student/people-they-know/real situations only — not How do people / Why do people / society in general; "
    "not Do you agree that [scientific fact] / confirming scientific claims)\n"
    '- "speaking_task": exactly 1 string (imperative instruction: specific situation or angle — not generic; '
    "what the student must DO; NOT a sample answer; on-topic; expects ~4–6 sentences of student speech)\n"
    "Do not include key_ideas, warmup_questions, vocabulary_items, grammar_patterns, choices, or support_words."
)


def _preview_system_speaking(level: Optional[str]) -> str:
    if _is_lesson_cefr_a1(level):
        return _PREVIEW_SYSTEM_SPEAKING_A1
    if _is_lesson_cefr_a2(level):
        return _PREVIEW_SYSTEM_SPEAKING_A2
    if _is_lesson_cefr_b1(level):
        return _PREVIEW_SYSTEM_SPEAKING_B1
    if _is_lesson_cefr_b2(level):
        level_block = (
            "LEVEL ADAPTATION (B2):\n"
            "- Add deeper thinking and arguments.\n"
            "- Use:\n"
            '  - "To what extent do you agree...?"\n'
            '  - "What would happen if...?"\n'
            '  - "Do you think this is a good idea? Why?"\n\n'
        )
    else:
        level_block = (
            "LEVEL ADAPTATION (fallback — not B2):\n"
            "- Simple structure; personal and concrete.\n"
            "- Must still avoid pure yes/no.\n\n"
        )
    return (
        "You are a helpful teacher. Output ONE JSON object only, no markdown.\n"
        "Use the SPEAKING FILTERED SOURCE in the user message ONLY as the source (no invented facts beyond it). "
        "It is a filtered topic and key situations — not a raw transcript; do not use or assume any other text.\n"
        "This preview is for speaking practice only. Apply ONLY these rules.\n\n"
        + _SPEAKING_GLOBAL_FORBIDDEN_RULES
        + "TOPIC:\n"
        "- 1 short, clear topic from source\n"
        "- Must sound natural and usable in conversation\n"
        "- No academic framing\n\n"
        "DISCUSSION QUESTIONS: exactly 6 questions\n"
        "- Must be open-ended (NO yes/no questions)\n"
        "- Must require 1–2 minutes of speaking\n"
        "- Focus on:\n"
        "  - personal experience\n"
        "  - opinions\n"
        "  - feelings\n"
        "  - real-life situations\n\n"
        + level_block
        + "SPEAKING TASK: exactly 1\n\n"
        "- Must be a real speaking scenario\n"
        "- Must require 30–60 seconds of continuous speech\n"
        "- Use formats like:\n"
        '  - "Talk about a time when..."\n'
        '  - "Describe a situation where..."\n'
        '  - "Imagine you are..."\n\n'
        "- Must be directly connected to topic\n"
        "- Must NOT require special knowledge\n\n"
        "GLOBAL RULES:\n\n"
        "- Every element must stay connected to source meaning\n"
        "- Do NOT introduce unrelated ideas\n"
        "- Do NOT generate academic or scientific language\n"
        "- Do NOT add vocabulary or grammar sections\n"
        "- Keep language natural and conversational\n"
        "- Avoid generic filler questions\n"
        "- Questions must help teacher start speaking immediately\n\n"
        "CRITICAL CHECK BEFORE OUTPUT:\n\n"
        '- If question can be answered with "yes/no" → rewrite it\n'
        "- If question sounds like textbook → simplify it\n"
        "- If question does not push student to speak → replace it\n\n"
        "Goal:\n"
        "Teacher opens → student starts speaking immediately\n\n"
        'Return ONLY these keys:\n'
        '- "topic": one short line (teacher-friendly)\n'
        '- "discussion_questions": exactly 6 strings\n'
        '- "speaking_task": exactly 1 string (the speaking scenario)\n'
        "Do not include key_ideas, warmup_questions, vocabulary_items, grammar_patterns, choices, or support_words."
    )


_PREVIEW_SYSTEM_VOCABULARY = (
    "You are a helpful teacher. Output ONE JSON object only, no markdown.\n"
    "Use the transcript below ONLY as the source (no invented facts).\n"
    'Return ONLY these keys for a vocabulary preview:\n'
    '- "topic": one short line (teacher-friendly)\n'
    '- "vocabulary_items": 8 to 10 items. Each item is an object with:\n'
    '  "english": word or short chunk from the source, and "note": short Ukrainian or English meaning/gloss\n'
    "Do not include key_ideas, warmup_questions, discussion_questions, or grammar_patterns."
)

_PREVIEW_SYSTEM_PHRASES = (
    "You are a helpful teacher. Output ONE JSON object only, no markdown.\n"
    "Use the transcript below ONLY as the source (no invented facts).\n"
    'Return ONLY these keys for a grammar / phrases preview:\n'
    '- "topic": one short line (teacher-friendly)\n'
    '- "grammar_patterns": 2 to 3 objects, each with:\n'
    '  "structure": short name of the pattern, and "formula": short formula (e.g. "used to + verb")\n'
    "Do not include key_ideas, words, vocabulary_items, or discussion_questions."
)

_PREVIEW_INSTR_EASY = (
    "Make the material easier. Simplify vocabulary and ideas. "
    "Keep source meaning unchanged."
)

_PREVIEW_INSTR_DEEP = (
    "Deepen the material. Give stronger ideas and richer vocabulary. "
    "Stay within source content only. Do not invent new topics."
)

_PREVIEW_PATCH_SYSTEM = (
    "You are a helpful teacher. Output ONE JSON object only, no markdown.\n"
    "The user message includes Original transcript + Current preview (complete JSON). "
    "You MUST PATCH that JSON: start from it, do NOT regenerate the preview from the transcript alone. "
    "Output the SAME keys and structure as the current preview (format-specific). "
    "Do not add keys from other formats. Ground new facts only in the transcript. "
    "Preserve every block unless the teacher explicitly asks to change it; default = minimal edit. "
    "For [Простіше]/[Глибше] the output MUST differ measurably from the current preview JSON "
    "(simpler or richer wording), not a verbatim copy."
)

_MEMORY_PATCH_RULES = """MEMORY RULES (never violate):
1. Do not change topic unless explicitly asked
2. Do not change level unless explicitly asked
3. Do not change format unless explicitly asked
4. Do not delete existing questions, words, or patterns
5. If teacher asks to ADD, only add new content, never replace old content
6. Always work on top of current preview
7. Return full result = existing content + additions
8. Do not rewrite, paraphrase, simplify, reorder, or modify existing items unless explicitly asked
9. Never add duplicates or near-duplicates of existing questions, words, or patterns
10. If teacher asks to simplify, keep the same topic and structure, but make the language easier
11. If teacher asks to deepen, keep the same topic and structure, but expand with more depth and new content
"""


def _frozen_base_snapshot_for_patch(
    patch_kind: str, pd: dict[str, Any]
) -> tuple[Any, Any, Any, Any]:
    topic = pd.get("topic", "")
    if patch_kind == "lesson":
        questions: Any = {
            "warmup_questions": pd.get("warmup_questions", []),
            "core_questions": pd.get("core_questions", []),
            "choices": pd.get("choices", []),
        }
        words = pd.get("support_words", [])
        patterns: Any = []
    elif patch_kind == "questions":
        questions = pd.get("discussion_questions", [])
        words = []
        patterns = []
    elif patch_kind == "speaking":
        questions = {
            "discussion_questions": pd.get("discussion_questions", []),
            "speaking_task": pd.get("speaking_task", ""),
        }
        words = []
        patterns = []
    elif patch_kind == "vocabulary":
        questions = []
        words = pd.get("vocabulary_items", [])
        patterns = []
    elif patch_kind == "phrases":
        questions = []
        words = []
        patterns = pd.get("grammar_patterns", [])
    else:
        q_raw = pd.get("questions")
        if isinstance(q_raw, list) and q_raw:
            questions = q_raw
        else:
            questions = pd.get("key_ideas", [])
        words = pd.get("words", [])
        patterns = pd.get("exercises", [])
    return topic, questions, words, patterns


def _memory_frozen_teacher_section(
    patch_kind: str,
    preview_data: dict[str, Any],
    instruction: str,
) -> str:
    topic, questions, words, patterns = _frozen_base_snapshot_for_patch(
        patch_kind, preview_data
    )
    return (
        f"{_MEMORY_PATCH_RULES}\n"
        "Current preview (FROZEN BASE):\n"
        f"topic: {topic}\n"
        f"questions: {json.dumps(questions, ensure_ascii=False, default=str)}\n"
        f"words: {json.dumps(words, ensure_ascii=False, default=str)}\n"
        f"patterns: {json.dumps(patterns, ensure_ascii=False, default=str)}\n\n"
        f"Teacher instruction: {instruction}\n"
    )


_INTENT_BIAS_BY_KIND: dict[str, str] = {
    "lesson": (
        "VELIORA_ONBOARDING_INTENT_HINT: урок lesson warm up lead in розігрів розпочати навчання"
    ),
    "speaking": (
        "VELIORA_ONBOARDING_INTENT_HINT: питання discussion обговорення запитання speaking діалог"
    ),
    "questions": (
        "VELIORA_ONBOARDING_INTENT_HINT: питання discussion обговорення запитання speaking діалог"
    ),
    "vocabulary": (
        "VELIORA_ONBOARDING_INTENT_HINT: слова vocabulary лексика переклад словниковий vocab лексичний"
    ),
    "phrases": (
        "VELIORA_ONBOARDING_INTENT_HINT: граматика grammar фрази phrases sentence pattern структура морфологія"
    ),
}


def _patch_hard_constraints_block(
    kind: str, level: Optional[str] = None
) -> str:
    if kind == "vocabulary":
        return (
            "HARD CONSTRAINTS (this format):\n"
            "- vocabulary_items: MUST contain 8–10 items (never fewer than 8). Each item: english + note (meaning).\n"
            '- Command "додай більше слів" / "більше слів": ADD new grounded items so the list reaches 9–10 entries; '
            "keep existing pairs; do not replace the whole list unless asked.\n"
        )
    if kind == "questions":
        return (
            "HARD CONSTRAINTS (this format):\n"
            "- discussion_questions: MUST contain 3–5 items.\n"
            '- Command "додай більше питань" / "більше питань": ADD questions until there are 4–5 total; '
            "keep existing questions unless asked to remove.\n"
        )
    if kind == "speaking":
        if _is_lesson_cefr_a1(level):
            return (
                "HARD CONSTRAINTS (CEFR A1 speaking):\n"
                + _SPEAKING_GLOBAL_HARD_CONSTRAINTS_LINE
                + "- topic: A1 vocabulary only; prefer 1–3 words; max 4 English words; daily-life; "
                "e.g. hair growth → hair; hair loss → avoid or simplify; no abstract phrasing.\n"
                "- forbidden vocabulary everywhere: treatment, treatments, any medical terms; "
                "prefer avoiding growth, loss, stress unless essential and simple; "
                "replace with simple everyday words or drop the idea.\n"
                "- discussion_questions: exactly 6; every question must match the topic — no unrelated ideas; "
                "ONLY patterns: Do you...? / Have you...? / Can you...? (not meta) / Do you like...?; "
                "FORBIDDEN: \"Can you talk about...\", \"Can you share about...\", \"Can you describe...\"; "
                "max 8 words per question; simple but meaningful (not trivial or obvious); real personal situations; "
                "natural conversation; no obvious yes/no-only questions; no why/how; no robotic or specialist-knowledge questions.\n"
                "- speaking_task: exactly 1; CEFR A1 vocabulary only; topic-specific to the source — NOT generic; "
                "must clearly fit the filtered topic (not reusable for random topics); real situation from source; "
                "instantly understandable; no complex thinking.\n"
                '- Command "додай більше питань": refine or replace entries; keep exactly 6; all A1 rules.\n'
            )
        if _is_lesson_cefr_a2(level):
            return (
                "HARD CONSTRAINTS (CEFR A2 speaking):\n"
                + _SPEAKING_GLOBAL_HARD_CONSTRAINTS_LINE
                + "- topic: max 5 English words; simple, natural; from source; daily-life; no academic framing.\n"
                "- discussion_questions: exactly 6; keep ALL A1 speaking bases (on-topic; no meta "
                '"Can you talk/share/describe about..."; no trivial/obvious meaningless); '
                "A2-only: forbid \"What do you know about...\", \"What do you think about people...\", "
                '"Why do you think people..."; questions ONLY about the student, their experience, direct observations; '
                "no semantic drift — stay on source topic; "
                "allowed forms include What/Why/How do you feel/What do you usually do when; "
                "max 10 words each; personal experience or daily habits only — reject specialist knowledge; "
                "more than yes/no; 1–3 sentence answers.\n"
                "- FORBIDDEN in questions: abstract reasoning, societal/general (non-personal) debates, expert knowledge.\n"
                "- speaking_task: exactly 1; specific; on-topic; real situation OR personal experience; "
                "answer ~3–5 sentences; NOT generic (e.g. not \"Talk about this topic\").\n"
                "- CORE: A2 = A1 + simple reasons + simple experiences.\n"
                '- Command "додай більше питань": refine or replace; keep exactly 6; all A2 rules.\n'
            )
        if _is_lesson_cefr_b1(level):
            return (
                "HARD CONSTRAINTS (CEFR B1 speaking):\n"
                + _SPEAKING_GLOBAL_HARD_CONSTRAINTS_LINE
                + "- topic: max 5 English words; clear, natural; from source; no academic framing or complex terminology "
                '(e.g. wrong: "Social perceptions of baldness"; right: "How people feel about baldness").\n'
                "- discussion_questions: exactly 6; keep ALL A1+A2 speaking bases; max 12 words each; "
                "allowed forms include Why/How do you feel/Do you agree/What would you do if/Which do you prefer and why/Do you think; "
                "must require opinion + reason (explicit or implied); focus on student personally, people they know, situations they saw — "
                "NOT general world/society; FORBIDDEN: How do people...; Why do people...; society/groups in general; "
                'Do you agree that [scientific fact]...; questions forcing confirmation of scientific claims; '
                "What do you know about...; expert/scientific; abstract theory; essay-like; off-topic.\n"
                "- speaking_task: exactly 1; must be an instruction for what the student must DO (not sample answer or model text); "
                "specific situation or angle — NOT generic (e.g. not \"Share your personal experience or opinion about...\"); "
                "on-topic; opinion+explanation OR experience+reflection; expects ~4–6 sentences of student speech; "
                "not generic, abstract, or expert-level.\n"
                "- CORE: B1 = A2 + opinion + because.\n"
                '- Command "додай більше питань": refine or replace; keep exactly 6; all B1 speaking rules.\n'
            )
        return (
            "HARD CONSTRAINTS (this format — B2 speaking):\n"
            + _SPEAKING_GLOBAL_HARD_CONSTRAINTS_LINE
            + "- discussion_questions: MUST contain exactly 6 items (open-ended; not yes/no).\n"
            "- speaking_task: MUST be exactly 1 non-empty string (real speaking scenario, 30–60 seconds).\n"
            '- Command "додай більше питань": ADD or adjust questions while keeping exactly 6 total unless teacher asks otherwise.\n'
        )
    if kind == "phrases":
        return (
            "HARD CONSTRAINTS (this format):\n"
            "- grammar_patterns: MUST contain 2–3 objects (structure + formula each).\n"
        )
    if kind == "lesson":
        if _is_lesson_cefr_a1(level):
            return (
                "HARD CONSTRAINTS (CEFR A1 lesson):\n"
                "- GENERAL: CEFR A1 vocabulary only; Present Simple only; max 6 words per sentence; "
                "one topic; no \"—\" or empty fields; do not regenerate existing blocks—only extend when asked; "
                "exact counts only.\n"
                "- warmup_questions: exactly 5 (each max 6 words; Do you / Is it / Can you; no Why).\n"
                "- core_questions: exactly 4 (each max 7 words; What/Where/When do you; no hypotheticals).\n"
                '- choices: exactly 4, format only "X or Y?" (concrete; X/Y one word or max 2-word phrase).\n'
                '- support_words: exactly 6, each "English — Ukrainian", nouns/verbs only, topic-related.\n'
            )
        if _is_lesson_cefr_a2(level):
            return (
                "HARD CONSTRAINTS (CEFR A2 lesson):\n"
                "- GENERAL: concrete, simple English; no \"—\" or empty fields; "
                "do not regenerate existing blocks—only extend when asked; exact counts only; "
                "every element tied to main situation and core meaning; no unrelated or generic filler.\n"
                "- warmup_questions: exactly 5; patterns only Do you / Is it / Can you; "
                "no \"Have you ever\"; personal and concrete.\n"
                "- core_questions: exactly 4; Why/How allowed if concrete and tied to source; "
                "no abstract reasoning or theory; simple cause-effect only.\n"
                '- choices: exactly 4, format only "X or Y?".\n'
                '- support_words: exactly 6, each "English — Ukrainian", topic-related, practical A2 level.\n'
            )
        if _is_lesson_cefr_b1(level):
            return (
                "HARD CONSTRAINTS (CEFR B1 lesson):\n"
                "- GENERAL: clear English; no \"—\" or empty fields; "
                "do not regenerate existing blocks—only extend when asked; exact counts only; "
                "main situation and core meaning; no unrelated filler; not an essay or debate club.\n"
                "- warmup_questions: exactly 5; Do you / Is it / Can you / Have you; personal, relatable.\n"
                "- core_questions: exactly 4; opinions, reasons, simple arguments; "
                "no academic/scientific/theoretical explanations; no abstract debate.\n"
                '- choices: exactly 4, format only "X or Y?", topic-relevant.\n'
                "- support_words: exactly 6, everyday practical English — Ukrainian, "
                "all English words lowercase; no scientific/academic/technical terms.\n"
            )
        if _is_lesson_cefr_b2(level):
            return (
                "HARD CONSTRAINTS (CEFR B2 lesson):\n"
                "- GENERAL: natural English; no \"—\" or empty fields; "
                "do not regenerate existing blocks—only extend when asked; exact counts only; "
                "main situation and core meaning; deeper meaning OK but not an essay or lecture.\n"
                "- warmup_questions: exactly 5; Do you / Is it / Can you / Have you; relatable; "
                "no academic or theoretical phrasing.\n"
                "- core_questions: exactly 4; opinions, reasons, simple arguments; defend a point of view; "
                "comparison allowed; no abstract theory, academic analysis, lecture style, or specialized knowledge.\n"
                '- choices: exactly 4, format only "X or Y?"; realistic and relatable; '
                "no academic, medical, or expert-only concepts.\n"
                "- support_words: exactly 6, everyday practical English — Ukrainian, topic-related; "
                "all English words lowercase; no scientific/academic/technical terms.\n"
            )
        return (
            "HARD CONSTRAINTS (this format):\n"
            "- warmup_questions: exactly 5 real strings (never \"—\" or empty).\n"
            "- choices: 3–4 real \"this or that\" strings (never \"—\" or empty).\n"
            "- support_words: at least 5 real strings from source or topic.\n"
        )
    return (
        "HARD CONSTRAINTS (default format):\n"
        "- key_ideas: exactly 3 strings; words: 3–15 strings as appropriate.\n"
    )


def _preview_patch_rules_easy(kind: str, level: Optional[str] = None) -> str:
    if _is_lesson_cefr_a1(level):
        lesson_easy = (
            "Apply: зроби простіше — simplify wording of warmup_questions, core_questions, choices, "
            "and support_words only; keep exactly 5 warm-ups, 4 core questions, 4 choices, 6 support lines; "
            "same topic, simpler A1 English and Ukrainian glosses.\n"
        )
    elif _is_lesson_cefr_a2(level):
        lesson_easy = (
            "Apply: зроби простіше — simplify wording of warmup_questions, core_questions, choices, "
            "and support_words only; keep exactly 5 warm-ups, 4 core questions, 4 choices, 6 support lines; "
            "same topic and source situation, simpler A2 English and Ukrainian glosses; "
            "keep warmup patterns Do you / Is it / Can you only; core Why/How only if still concrete and tied to source.\n"
        )
    elif _is_lesson_cefr_b1(level):
        lesson_easy = (
            "Apply: зроби простіше — simplify wording of warmup_questions, core_questions, choices, "
            "and support_words only; keep exactly 5 warm-ups, 4 core questions, 4 choices, 6 support lines; "
            "same topic and core meaning, clearer B1 English and Ukrainian glosses; stay personal and relatable; "
            "no academic or essay-style phrasing.\n"
        )
    elif _is_lesson_cefr_b2(level):
        lesson_easy = (
            "Apply: зроби простіше — simplify wording of warmup_questions, core_questions, choices, "
            "and support_words only; keep exactly 5 warm-ups, 4 core questions, 4 choices, 6 support lines; "
            "same topic and core meaning, clearer B2 English and Ukrainian glosses; stay conversational; "
            "not an essay or lecture.\n"
        )
    else:
        lesson_easy = (
            "Apply: зроби простіше — simplify wording of warmup_questions, choices, and support_words only; "
            "keep exactly 5 warm-up questions, 3–4 choices, and at least 5 support words; same topics, simpler English.\n"
        )
    if _is_lesson_cefr_a1(level):
        speaking_patch_easy = (
            "Apply: зроби простіше — simplify topic (A1 words; prefer 1–3 words; max 4), discussion_questions, and speaking_task; "
            "keep every question on-topic; keep exactly 6 questions using ONLY Do you / Have you / Can you / Do you like (max 8 words each); "
            "no meta-questions (Can you talk/share/describe about...); replace with specific personal questions; "
            "stay meaningful and natural, not trivial; invite a short answer; "
            "no treatment, treatments, medical terms; avoid growth, loss, stress unless essential; "
            "1 speaking_task: topic-specific (not generic), A1 words only; all CEFR A1 speaking STRICT rules.\n"
        )
    elif _is_lesson_cefr_a2(level):
        speaking_patch_easy = (
            "Apply: зроби простіше — simplify topic (max 5 words), discussion_questions, and speaking_task; "
            "keep exactly 6 questions (allowed A2 forms; max 10 words each); on-topic; personal; "
            "no meta-questions; no abstract/societal/expert prompts; "
            "1 speaking_task: specific, on-topic, ~3–5 sentence answer; not generic; all CEFR A2 speaking rules.\n"
        )
    elif _is_lesson_cefr_b1(level):
        speaking_patch_easy = (
            "Apply: зроби простіше — simplify topic (max 5 words), discussion_questions, and speaking_task; "
            "keep exactly 6 questions (allowed B1 forms; max 12 words each); opinion + reason; on-topic; source-based; "
            "not essay-like or expert; 1 speaking_task: specific, on-topic, opinion+explanation or experience+reflection; "
            "4–6 sentences; all CEFR B1 speaking rules.\n"
        )
    else:
        speaking_patch_easy = (
            "Apply: зроби простіше — simplify discussion_questions and speaking_task wording only; "
            "keep exactly 6 questions and 1 speaking_task; stay open-ended and conversational.\n"
        )
    spec = {
        "lesson": lesson_easy,
        "questions": (
            "Apply: зроби простіше — simplify discussion_questions wording only; keep 3–5 questions.\n"
        ),
        "speaking": speaking_patch_easy,
        "vocabulary": (
            "Apply: зроби простіше — simplify `note` (meaning) text only; keep 8–10 vocabulary_items; "
            "same english chunks unless simplification requires tiny edits.\n"
        ),
        "phrases": (
            "Apply: зроби простіше — simplify structure/formula explanations; keep 2–3 patterns.\n"
        ),
        "default": "Apply: зроби простіше — simplify key_ideas and words; keep counts.\n",
    }.get(kind, "Apply: зроби простіше — simplify key_ideas and words.\n")
    return (
        "Rules (button Простіше — PATCH only, NOT full regeneration):\n"
        "- Current preview (complete JSON) is the stable base; copy it forward then edit.\n"
        "- You MUST produce JSON that is not identical to the current preview (measurable simpler text).\n"
        "- Do not rebuild from transcript alone; do not drop unrelated blocks.\n"
        f"- {spec}"
        + _patch_hard_constraints_block(kind, level)
    )


def _preview_patch_rules_deep(kind: str, level: Optional[str] = None) -> str:
    if _is_lesson_cefr_a1(level):
        lesson_deep = (
            "Apply: зроби глибше (CEFR A1 lesson) — warmup_questions and core_questions: do NOT repeat or paraphrase "
            "the existing strings; write NEW questions only, each more specific to daily life, while keeping exactly "
            "5 warm-ups and 4 core questions. Example: \"Do you talk to yourself?\" → "
            "\"Do you talk to yourself at home or outside?\" "
            "For choices and support_words: use NEW concrete daily-life wording where you change them; "
            "keep exactly 4 choices and 6 support lines; stay in source and all A1 rules.\n"
        )
    elif _is_lesson_cefr_a2(level):
        lesson_deep = (
            "Apply: зроби глибше (CEFR A2 lesson) — enrich warmup_questions, core_questions, choices, support_words; "
            "stay in the same source situation and core meaning; keep exactly 5 warm-ups, 4 core questions, "
            "4 choices, 6 support lines; add concrete detail and simple cause-effect where it helps; "
            "do NOT repeat or lightly paraphrase—make wording meaningfully new; no abstract theory; all A2 rules.\n"
        )
    elif _is_lesson_cefr_b1(level):
        lesson_deep = (
            "Apply: зроби глибше (CEFR B1 lesson) — enrich warmup_questions, core_questions, choices, support_words; "
            "stay in the same source situation and core meaning; keep exactly 5 warm-ups, 4 core questions, "
            "4 choices, 6 support lines; add opinion, reasons, and realistic detail without becoming academic; "
            "do NOT repeat or lightly paraphrase—make wording meaningfully new; not an essay or debate; all B1 rules.\n"
        )
    elif _is_lesson_cefr_b2(level):
        lesson_deep = (
            "Apply: зроби глибше (CEFR B2 lesson) — enrich warmup_questions, core_questions, choices, support_words; "
            "stay in the same source situation and core meaning; keep exactly 5 warm-ups, 4 core questions, "
            "4 choices, 6 support lines; allow richer opinion, comparison, and argument without essay or lecture style; "
            "do NOT repeat or lightly paraphrase—make wording meaningfully new; all B2 rules.\n"
        )
    else:
        lesson_deep = (
            "Apply: зроби глибше — enrich warmup_questions, choices, and support_words; stay in source; "
            "keep 5 warm-ups, 3–4 choices, at least 5 support words.\n"
        )
    if _is_lesson_cefr_a1(level):
        speaking_patch_deep = (
            "Apply: зроби глибше (CEFR A1 speaking) — NEW wording only; keep exactly 6 questions and 1 speaking_task; "
            "all questions must stay tied to the same topic; more concrete, personal detail from source; "
            "still ONLY Do you / Have you / Can you / Do you like; never Can you talk/share/describe about...; "
            "max 8 words per question; meaningful not trivial; topic A1 words, prefer 1–3 words, max 4; "
            "no why that needs long explanation; no unrelated ideas (e.g. stress); avoid growth, loss, stress unless essential; "
            "no treatment/medical wording; speaking_task must stay topic-specific (not generic); measurable change; all A1 STRICT rules.\n"
        )
    elif _is_lesson_cefr_a2(level):
        speaking_patch_deep = (
            "Apply: зроби глибше (CEFR A2 speaking) — NEW wording only; keep exactly 6 questions and 1 speaking_task; "
            "richer simple reasons (because), past experience, short explanations; stay on-topic and personal; "
            "allowed question forms only; max 10 words per question; no meta; no abstract/societal/expert; "
            "speaking_task stays specific and on-topic (~3–5 sentence answer); measurable change; all A2 rules.\n"
        )
    elif _is_lesson_cefr_b1(level):
        speaking_patch_deep = (
            "Apply: зроби глибше (CEFR B1 speaking) — NEW wording only; keep exactly 6 questions and 1 speaking_task; "
            "richer opinions, reasons (because/so), simple arguments, comparison; stay on-topic and source-based; "
            "allowed B1 forms only; max 12 words per question; not essay-like or expert; "
            "speaking_task: opinion+explanation or experience+reflection; 4–6 sentences; measurable change; all B1 rules.\n"
        )
    else:
        speaking_patch_deep = (
            "Apply: зроби глибше — richer discussion_questions and speaking_task; stay in source; "
            "keep exactly 6 questions and 1 speaking_task; remain open-ended.\n"
        )
    spec = {
        "lesson": lesson_deep,
        "questions": (
            "Apply: зроби глибше — richer discussion_questions; stay in source; keep 3–5 items.\n"
        ),
        "speaking": speaking_patch_deep,
        "vocabulary": (
            "Apply: зроби глибше — richer `note` (meanings) or slightly more precise english; "
            "keep 8–10 vocabulary_items.\n"
        ),
        "phrases": (
            "Apply: зроби глибше — sharper structure names/formulas; stay in source; keep 2–3 patterns.\n"
        ),
        "default": "Apply: зроби глибше — enrich key_ideas and words; stay in source.\n",
    }.get(kind, "Apply: зроби глибше — enrich key_ideas and words; stay in source.\n")
    return (
        "Rules (button Глибше — PATCH only, NOT full regeneration):\n"
        "- Current preview (complete JSON) is the stable base; copy it forward then edit.\n"
        "- You MUST produce JSON that is not identical to the current preview (measurable richer detail).\n"
        "- Do not invent topics outside the transcript; do not drop unrelated blocks.\n"
        f"- {spec}"
        + _patch_hard_constraints_block(kind, level)
    )


def _preview_patch_rules_custom(kind: str, level: Optional[str] = None) -> str:
    common = (
        "Rules (custom teacher text — PATCH only):\n"
        "- Current preview (complete JSON) is the ONLY stable base; merge changes into it.\n"
        "- Never return identical JSON when the teacher asked for a change.\n"
        "- Preserve all blocks unless the teacher explicitly asks to remove or replace.\n"
        "- Default: change only the minimal block the instruction targets.\n"
        '- "зроби простіше" / simpler: simplify only the targeted block.\n'
        '- "зроби глибше" / deeper: enrich only the targeted block; stay in source.\n'
    )
    if kind == "vocabulary":
        return (
            common
            + _patch_hard_constraints_block(kind, level)
            + (
                '- "додай більше слів": extend vocabulary_items to 9–10 grounded entries (not a rewrite with the same count).\n'
            )
        )
    if kind == "questions":
        return (
            common
            + _patch_hard_constraints_block(kind, level)
            + (
                '- "додай більше питань": extend discussion_questions to 4–5 grounded questions.\n'
            )
        )
    if kind == "speaking":
        return (
            common
            + _patch_hard_constraints_block(kind, level)
            + (
                '- "додай більше питань": refine or replace discussion_questions entries; keep exactly 6 total.\n'
            )
        )
    if kind == "lesson":
        return common + _patch_hard_constraints_block(kind, level)
    if kind == "phrases":
        return common + _patch_hard_constraints_block(kind, level)
    return (
        common
        + _patch_hard_constraints_block(kind, level)
        + (
            '- "більше слів": extend `words` with NEW items.\n'
            '- "більше питань": extend `questions` if present, else adjust key_ideas.\n'
            '- "більше ідей": extend or enrich `key_ideas`.\n'
            '- "більше вправ": extend `exercises` if present.\n'
        )
    )

_PREVIEW_LIMIT_TEXT = "Давай підтвердимо або почнемо з нового 👇"

_MAX_PREVIEW_EDIT_ROUNDS = 5

_ONB_FMT_STEP1_KB = InlineKeyboardMarkup(
    [
        [
            InlineKeyboardButton("📚 Урок", callback_data="onb_fmt_lesson"),
            InlineKeyboardButton("💬 Speaking", callback_data="onb_fmt_speaking"),
        ],
        [
            InlineKeyboardButton("📖 Слова", callback_data="onb_fmt_vocabulary"),
            InlineKeyboardButton("✏️ Граматика", callback_data="onb_fmt_phrases"),
        ],
    ]
)

_ONB_LEVEL_KB = InlineKeyboardMarkup(
    [
        [
            InlineKeyboardButton("A1", callback_data="onb_lvl_A1"),
            InlineKeyboardButton("A2", callback_data="onb_lvl_A2"),
            InlineKeyboardButton("B1", callback_data="onb_lvl_B1"),
            InlineKeyboardButton("B2", callback_data="onb_lvl_B2"),
        ],
    ]
)

# Post-card actions (prefix onb_p_ — matches CallbackQueryHandler ^onb_ in main).
_POST_CARD_KB = InlineKeyboardMarkup(
    [
        [
            InlineKeyboardButton("🔄 Змінити формат", callback_data="onb_p_fmt"),
            InlineKeyboardButton("📊 Змінити рівень", callback_data="onb_p_lvl"),
        ],
    ]
)

_POST_CARD_FMT_KB = InlineKeyboardMarkup(
    [
        [
            InlineKeyboardButton("📚 Урок", callback_data="onb_p_f_lesson"),
            InlineKeyboardButton("💬 Speaking", callback_data="onb_p_f_speaking"),
        ],
        [
            InlineKeyboardButton("📖 Слова", callback_data="onb_p_f_vocabulary"),
            InlineKeyboardButton("✏️ Граматика", callback_data="onb_p_f_phrases"),
        ],
    ]
)

_POST_CARD_LVL_KB = InlineKeyboardMarkup(
    [
        [
            InlineKeyboardButton("A1", callback_data="onb_p_l_A1"),
            InlineKeyboardButton("A2", callback_data="onb_p_l_A2"),
            InlineKeyboardButton("B1", callback_data="onb_p_l_B1"),
            InlineKeyboardButton("B2", callback_data="onb_p_l_B2"),
        ],
    ]
)

_PREVIEW_KB = InlineKeyboardMarkup(
    [
        [
            InlineKeyboardButton("✅ Все ок", callback_data="onb_prv_ok"),
            InlineKeyboardButton("✏️ Уточнити", callback_data="onb_prv_ref"),
        ],
    ]
)

_PREVIEW_REFINE_KB = InlineKeyboardMarkup(
    [
        [
            InlineKeyboardButton("📉 Простіше", callback_data="onb_prv_r_easy"),
            InlineKeyboardButton("📚 Глибше", callback_data="onb_prv_r_deep"),
            InlineKeyboardButton("✍️ Своє", callback_data="onb_prv_r_own"),
        ],
    ]
)

_PREVIEW_LIMIT_KB = InlineKeyboardMarkup(
    [
        [
            InlineKeyboardButton("✅ Все ок", callback_data="onb_prv_ok"),
            InlineKeyboardButton("🔄 Нове джерело", callback_data="onb_prv_new"),
        ],
    ]
)

_FMT_CHANGE_LABELS = {
    "lesson": "📚 Урок",
    "speaking": "💬 Speaking",
    "questions": "💬 Speaking",
    "vocabulary": "📖 Слова",
    "phrases": "✏️ Граматика",
}


def _is_lesson_cefr_a1(level: Optional[str]) -> bool:
    if level is None:
        return False
    return str(level).strip().upper() == "A1"


def _is_lesson_cefr_a2(level: Optional[str]) -> bool:
    if level is None:
        return False
    return str(level).strip().upper() == "A2"


def _is_lesson_cefr_b1(level: Optional[str]) -> bool:
    if level is None:
        return False
    return str(level).strip().upper() == "B1"


def _is_lesson_cefr_b2(level: Optional[str]) -> bool:
    if level is None:
        return False
    return str(level).strip().upper() == "B2"


def _preview_format_kind(fmt: Optional[str]) -> str:
    if not fmt:
        return "default"
    f = str(fmt).strip().lower()
    if f == "lesson":
        return "lesson"
    if f == "speaking":
        return "speaking"
    if f == "questions":
        return "questions"
    if f in ("vocabulary", "words"):
        return "vocabulary"
    if f in ("grammar", "phrases"):
        return "phrases"
    return "default"


def _preview_system_for_initial(kind: str, level: Optional[str] = None) -> str:
    if kind == "lesson" and _is_lesson_cefr_a1(level):
        return _PREVIEW_SYSTEM_LESSON_A1
    if kind == "lesson" and _is_lesson_cefr_a2(level):
        return _PREVIEW_SYSTEM_LESSON_A2
    if kind == "lesson" and _is_lesson_cefr_b1(level):
        return _PREVIEW_SYSTEM_LESSON_B1
    if kind == "lesson" and _is_lesson_cefr_b2(level):
        return _PREVIEW_SYSTEM_LESSON_B2
    if kind == "speaking":
        return _preview_system_speaking(level)
    return {
        "lesson": _PREVIEW_SYSTEM_LESSON,
        "questions": _PREVIEW_SYSTEM_QUESTIONS,
        "vocabulary": _PREVIEW_SYSTEM_VOCABULARY,
        "phrases": _PREVIEW_SYSTEM_PHRASES,
        "default": _PREVIEW_SYSTEM_DEFAULT,
    }.get(kind, _PREVIEW_SYSTEM_DEFAULT)


def _preview_merge_list_keys(
    kind: str, level: Optional[str] = None
) -> tuple[str, ...]:
    if kind == "lesson" and (
        _is_lesson_cefr_a1(level)
        or _is_lesson_cefr_a2(level)
        or _is_lesson_cefr_b1(level)
        or _is_lesson_cefr_b2(level)
    ):
        return ("warmup_questions", "core_questions", "support_words", "choices")
    if kind == "lesson":
        return ("warmup_questions", "support_words", "choices")
    return {
        "questions": ("discussion_questions",),
        "speaking": ("discussion_questions",),
        "vocabulary": ("vocabulary_items",),
        "phrases": ("grammar_patterns",),
        "default": ("questions", "exercises"),
    }.get(kind, ("questions", "exercises"))


def _coerce_vocabulary_items(raw: Any) -> list[str]:
    out: list[str] = []
    if not isinstance(raw, list):
        return []
    for x in raw:
        if isinstance(x, dict):
            en = str(x.get("english") or x.get("en") or "").strip()
            note = str(
                x.get("note") or x.get("ua") or x.get("meaning") or x.get("gloss") or ""
            ).strip()
            if en:
                out.append(f"{en} — {note}" if note else en)
        else:
            s = str(x).strip()
            if s:
                out.append(s)
    return out[:10]


def _coerce_grammar_patterns(raw: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if not isinstance(raw, list):
        return []
    for x in raw:
        if isinstance(x, dict):
            st = str(
                x.get("structure") or x.get("pattern") or x.get("name") or ""
            ).strip()
            fm = str(x.get("formula") or "").strip()
            if st or fm:
                out.append({"structure": st or "—", "formula": fm})
        else:
            s = str(x).strip()
            if s:
                out.append({"structure": s, "formula": ""})
    return out[:3]


def _lesson_nonempty_strings(items: Any, max_n: int) -> list[str]:
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for x in items:
        s = str(x).strip()
        if s and s != "—":
            out.append(s)
    return out[:max_n]


_A1_FILTER_FALLBACK_TOPIC = "everyday life at home and school"
_A1_FILTER_FALLBACK_SCENES: tuple[str, ...] = (
    "you wake up in the morning",
    "you eat at home",
    "you go to school or work",
    "you talk with family",
    "you sleep at night",
)


def _coerce_a1_filter_output(data: dict[str, Any]) -> tuple[str, list[str]]:
    topic = str(data.get("topic") or "").strip()
    raw_scenes = data.get("scenes")
    scenes: list[str] = []
    if isinstance(raw_scenes, list):
        for x in raw_scenes:
            s = str(x).strip()
            if s:
                scenes.append(s)
    return topic, scenes[:5]


def _a1_filtered_source_user_block(topic: str, scenes: list[str]) -> str:
    lines = [
        "A1 FILTERED SOURCE — use ONLY this text as your source for the lesson preview.",
        "Do not use or rely on any raw transcript.",
        "",
        f"Topic: {topic}",
        "",
        "Scenes:",
    ]
    for s in scenes:
        lines.append(f"- {s}")
    return "\n".join(lines)


def _a1_resolved_filtered_block(topic: str, scenes: list[str]) -> str:
    if not topic:
        return _a1_filtered_source_user_block(
            _A1_FILTER_FALLBACK_TOPIC, list(_A1_FILTER_FALLBACK_SCENES)
        )
    if len(scenes) < 3:
        seen = {s.lower() for s in scenes}
        for s in _A1_FILTER_FALLBACK_SCENES:
            if len(scenes) >= 3:
                break
            if s.lower() not in seen:
                scenes.append(s)
                seen.add(s.lower())
        if len(scenes) < 3:
            return _a1_filtered_source_user_block(
                _A1_FILTER_FALLBACK_TOPIC, list(_A1_FILTER_FALLBACK_SCENES)
            )
    return _a1_filtered_source_user_block(topic, scenes[:5])


_A2_FILTER_FALLBACK_TOPIC = _A1_FILTER_FALLBACK_TOPIC
_A2_FILTER_FALLBACK_SCENES = _A1_FILTER_FALLBACK_SCENES

_coerce_a2_filter_output = _coerce_a1_filter_output


def _a2_filtered_source_user_block(topic: str, scenes: list[str]) -> str:
    lines = [
        "A2 FILTERED SOURCE — use ONLY this text as your source for the lesson preview.",
        "Do not use or rely on any raw transcript.",
        "",
        f"Topic: {topic}",
        "",
        "Scenes:",
    ]
    for s in scenes:
        lines.append(f"- {s}")
    return "\n".join(lines)


def _a2_resolved_filtered_block(topic: str, scenes: list[str]) -> str:
    if not topic:
        return _a2_filtered_source_user_block(
            _A2_FILTER_FALLBACK_TOPIC, list(_A2_FILTER_FALLBACK_SCENES)
        )
    if len(scenes) < 3:
        seen = {s.lower() for s in scenes}
        for s in _A2_FILTER_FALLBACK_SCENES:
            if len(scenes) >= 3:
                break
            if s.lower() not in seen:
                scenes.append(s)
                seen.add(s.lower())
        if len(scenes) < 3:
            return _a2_filtered_source_user_block(
                _A2_FILTER_FALLBACK_TOPIC, list(_A2_FILTER_FALLBACK_SCENES)
            )
    return _a2_filtered_source_user_block(topic, scenes[:5])


_B1_FILTER_FALLBACK_TOPIC = _A1_FILTER_FALLBACK_TOPIC
_B1_FILTER_FALLBACK_SCENES = _A1_FILTER_FALLBACK_SCENES

_coerce_b1_filter_output = _coerce_a1_filter_output


def _b1_filtered_source_user_block(topic: str, scenes: list[str]) -> str:
    lines = [
        "B1 FILTERED SOURCE — use ONLY this text as your source for the lesson preview.",
        "Do not use or rely on any raw transcript.",
        "",
        f"Topic: {topic}",
        "",
        "Scenes:",
    ]
    for s in scenes:
        lines.append(f"- {s}")
    return "\n".join(lines)


def _b1_resolved_filtered_block(topic: str, scenes: list[str]) -> str:
    if not topic:
        return _b1_filtered_source_user_block(
            _B1_FILTER_FALLBACK_TOPIC, list(_B1_FILTER_FALLBACK_SCENES)
        )
    if len(scenes) < 3:
        seen = {s.lower() for s in scenes}
        for s in _B1_FILTER_FALLBACK_SCENES:
            if len(scenes) >= 3:
                break
            if s.lower() not in seen:
                scenes.append(s)
                seen.add(s.lower())
        if len(scenes) < 3:
            return _b1_filtered_source_user_block(
                _B1_FILTER_FALLBACK_TOPIC, list(_B1_FILTER_FALLBACK_SCENES)
            )
    return _b1_filtered_source_user_block(topic, scenes[:5])


_B2_FILTER_FALLBACK_TOPIC = _A1_FILTER_FALLBACK_TOPIC
_B2_FILTER_FALLBACK_SCENES = _A1_FILTER_FALLBACK_SCENES

_coerce_b2_filter_output = _coerce_a1_filter_output


def _b2_filtered_source_user_block(topic: str, scenes: list[str]) -> str:
    lines = [
        "B2 FILTERED SOURCE — use ONLY this text as your source for the lesson preview.",
        "Do not use or rely on any raw transcript.",
        "",
        f"Topic: {topic}",
        "",
        "Scenes:",
    ]
    for s in scenes:
        lines.append(f"- {s}")
    return "\n".join(lines)


def _b2_resolved_filtered_block(topic: str, scenes: list[str]) -> str:
    if not topic:
        return _b2_filtered_source_user_block(
            _B2_FILTER_FALLBACK_TOPIC, list(_B2_FILTER_FALLBACK_SCENES)
        )
    if len(scenes) < 3:
        seen = {s.lower() for s in scenes}
        for s in _B2_FILTER_FALLBACK_SCENES:
            if len(scenes) >= 3:
                break
            if s.lower() not in seen:
                scenes.append(s)
                seen.add(s.lower())
        if len(scenes) < 3:
            return _b2_filtered_source_user_block(
                _B2_FILTER_FALLBACK_TOPIC, list(_B2_FILTER_FALLBACK_SCENES)
            )
    return _b2_filtered_source_user_block(topic, scenes[:5])


_SPEAKING_FILTER_FALLBACK_TOPIC = _B2_FILTER_FALLBACK_TOPIC
_SPEAKING_FILTER_FALLBACK_SCENES = _B2_FILTER_FALLBACK_SCENES

_coerce_speaking_filter_output = _coerce_a1_filter_output


def _speaking_filtered_source_user_block(topic: str, scenes: list[str]) -> str:
    lines = [
        "SPEAKING FILTERED SOURCE — use ONLY this text as your source for the speaking preview.",
        "Do not use or rely on any raw transcript.",
        "",
        f"Topic: {topic}",
        "",
        "Key situations / angles:",
    ]
    for s in scenes:
        lines.append(f"- {s}")
    return "\n".join(lines)


def _speaking_resolved_filtered_block(topic: str, scenes: list[str]) -> str:
    if not topic:
        return _speaking_filtered_source_user_block(
            _SPEAKING_FILTER_FALLBACK_TOPIC, list(_SPEAKING_FILTER_FALLBACK_SCENES)
        )
    if len(scenes) < 3:
        seen = {s.lower() for s in scenes}
        for s in _SPEAKING_FILTER_FALLBACK_SCENES:
            if len(scenes) >= 3:
                break
            if s.lower() not in seen:
                scenes.append(s)
                seen.add(s.lower())
        if len(scenes) < 3:
            return _speaking_filtered_source_user_block(
                _SPEAKING_FILTER_FALLBACK_TOPIC, list(_SPEAKING_FILTER_FALLBACK_SCENES)
            )
    return _speaking_filtered_source_user_block(topic, scenes[:5])


class _OnboardingEnrichedMessage:
    """Proxy so pipeline sees enriched text/caption without mutating the real Message."""

    __slots__ = ("_base", "_enriched")

    def __init__(self, base: Message, enriched: str) -> None:
        self._base = base
        self._enriched = enriched

    @property
    def text(self) -> str:
        return self._enriched

    @property
    def caption(self) -> Optional[str]:
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


def _normalize_preview_output(
    data: dict[str, Any], kind: str, level: Optional[str] = None
) -> dict[str, Any]:
    topic = str(data.get("topic", "") or "").strip() or "—"

    if kind == "lesson":
        if (
            _is_lesson_cefr_a1(level)
            or _is_lesson_cefr_a2(level)
            or _is_lesson_cefr_b1(level)
            or _is_lesson_cefr_b2(level)
        ):
            wq = _lesson_nonempty_strings(data.get("warmup_questions"), 5)
            cq = _lesson_nonempty_strings(data.get("core_questions"), 4)
            ch = _lesson_nonempty_strings(data.get("choices"), 4)
            sw = _lesson_nonempty_strings(data.get("support_words"), 6)
            return {
                "topic": topic,
                "warmup_questions": wq,
                "core_questions": cq,
                "choices": ch,
                "support_words": sw,
            }
        wq = _lesson_nonempty_strings(data.get("warmup_questions"), 5)
        ch = _lesson_nonempty_strings(data.get("choices"), 4)
        sw = _lesson_nonempty_strings(data.get("support_words"), 15)
        return {
            "topic": topic,
            "warmup_questions": wq,
            "choices": ch,
            "support_words": sw,
        }

    if kind == "questions":
        dq = data.get("discussion_questions")
        if not isinstance(dq, list):
            dq = []
        dq = [str(x).strip() for x in dq if str(x).strip()][:5]
        while len(dq) < 3:
            dq.append("—")
        return {"topic": topic, "discussion_questions": dq}

    if kind == "speaking":
        dq = data.get("discussion_questions")
        if not isinstance(dq, list):
            dq = []
        dq = [str(x).strip() for x in dq if str(x).strip()][:6]
        while len(dq) < 6:
            dq.append("—")
        task = str(data.get("speaking_task") or "").strip() or "—"
        return {
            "topic": topic,
            "discussion_questions": dq,
            "speaking_task": task,
        }

    if kind == "vocabulary":
        items = _coerce_vocabulary_items(data.get("vocabulary_items"))
        while len(items) < 8:
            items.append("—")
        return {"topic": topic, "vocabulary_items": items[:10]}

    if kind == "phrases":
        gp = _coerce_grammar_patterns(data.get("grammar_patterns"))
        while len(gp) < 2:
            gp.append({"structure": "—", "formula": ""})
        return {"topic": topic, "grammar_patterns": gp[:3]}

    key_ideas = data.get("key_ideas")
    words = data.get("words")
    if not isinstance(key_ideas, list):
        key_ideas = []
    if not isinstance(words, list):
        words = []
    ki = [str(x).strip() for x in key_ideas if str(x).strip()][:6]
    while len(ki) < 3:
        ki.append("—")
    wd = [str(x).strip() for x in words if str(x).strip()][:15]
    out: dict[str, Any] = {
        "topic": topic,
        "key_ideas": ki,
        "words": wd,
    }
    for extra_key in ("questions", "exercises"):
        extra_val = data.get(extra_key)
        if isinstance(extra_val, list) and extra_val:
            out[extra_key] = [
                str(x).strip() for x in extra_val if str(x).strip()
            ][:25]
    return out


def _normalize_gpt_preview_dict(data: dict[str, Any]) -> dict[str, Any]:
    return _normalize_preview_output(data, "default")


def _enriched_onboarding_transcript_block(
    fmt: Optional[str],
    lvl: Optional[str],
    transcript: str,
) -> str:
    kind = _preview_format_kind(fmt)
    bias = _INTENT_BIAS_BY_KIND.get(kind, "")
    parts = [f"[FORMAT={fmt}]", f"[LEVEL={lvl}]"]
    if bias:
        parts.append(bias)
    parts.extend(["", f"USER CONTENT:\n{transcript}"])
    return "\n".join(parts)


def _preview_blocks_for_prompt(preview_data: dict[str, Any]) -> tuple[str, str, str]:
    topic = str(preview_data.get("topic") or "—").strip() or "—"
    ideas = preview_data.get("key_ideas")
    if not isinstance(ideas, list):
        ideas = []
    ideas = [str(x).strip() for x in ideas if str(x).strip()]
    while len(ideas) < 3:
        ideas.append("—")
    ideas = ideas[:6]
    ideas_str = " | ".join(ideas)
    words = preview_data.get("words")
    if not isinstance(words, list):
        words = []
    words = [str(x).strip() for x in words if str(x).strip()]
    words_str = ", ".join(words) if words else "—"
    return topic, ideas_str, words_str


def _build_preview_patch_user_content(
    transcript_snippet: str,
    preview_data: dict[str, Any],
    teacher_text: str,
    rules_block: str,
    patch_kind: str,
) -> str:
    pd = preview_data if isinstance(preview_data, dict) else {}
    topic, ideas_str, words_str = _preview_blocks_for_prompt(pd)
    preview_json = json.dumps(pd, ensure_ascii=False, default=str)
    memory_frozen = _memory_frozen_teacher_section(
        patch_kind, pd, teacher_text.strip()
    )
    return (
        f"PATCH_FORMAT_KIND: {patch_kind}\n"
        "You are editing the existing preview JSON below — do NOT rebuild preview from transcript only.\n\n"
        f"Original transcript (source of truth for new facts):\n{transcript_snippet}\n\n"
        "Current preview (human summary — JSON below is authoritative):\n"
        f"TOPIC: {topic}\n"
        f"IDEAS: {ideas_str}\n"
        f"WORDS: {words_str}\n\n"
        f"Current preview (complete JSON, all fields — stable base):\n{preview_json}\n\n"
        f"{memory_frozen}\n"
        f"{rules_block}"
    )


def _format_preview_message(
    preview_data: dict[str, Any],
    format_key: Optional[str] = None,
) -> str:
    kind = _preview_format_kind(format_key)
    topic = str(preview_data.get("topic") or "—").strip() or "—"
    header = "📋 Ось що знайшов:\n\n" + f"📌 Тема: {topic}\n\n"

    if kind == "lesson":
        wq = _lesson_nonempty_strings(preview_data.get("warmup_questions"), 10)
        cq = _lesson_nonempty_strings(preview_data.get("core_questions"), 10)
        ch = _lesson_nonempty_strings(preview_data.get("choices"), 10)
        sw = _lesson_nonempty_strings(preview_data.get("support_words"), 20)
        body = "🔥 Розминка:\n" + "\n".join(f"• {x}" for x in wq) + "\n\n"
        if cq:
            body += "❓ Основні питання:\n" + "\n".join(f"• {x}" for x in cq) + "\n\n"
        if ch:
            body += "⚖️ This or that:\n" + "\n".join(f"• {x}" for x in ch) + "\n\n"
        body += "📚 Слова: " + (", ".join(sw) if sw else "—")
        return header + body

    if kind == "questions":
        dq = preview_data.get("discussion_questions")
        if not isinstance(dq, list):
            dq = []
        dq = [str(x).strip() for x in dq if str(x).strip()]
        body = "💬 Питання для обговорення:\n" + "\n".join(f"• {x}" for x in dq)
        return header + body

    if kind == "speaking":
        dq = preview_data.get("discussion_questions")
        if not isinstance(dq, list):
            dq = []
        dq = [str(x).strip() for x in dq if str(x).strip()]
        task = str(preview_data.get("speaking_task") or "").strip() or "—"
        body = (
            "💬 Питання для обговорення:\n"
            + "\n".join(f"• {x}" for x in dq)
            + "\n\n🎤 Завдання для мовлення:\n"
            + f"• {task}"
        )
        return header + body

    if kind == "vocabulary":
        items = preview_data.get("vocabulary_items")
        if not isinstance(items, list):
            items = []
        lines = [str(x).strip() for x in items if str(x).strip()]
        body = "📖 Ключові слова:\n" + "\n".join(f"• {x}" for x in lines)
        return header + body

    if kind == "phrases":
        gp = preview_data.get("grammar_patterns")
        if not isinstance(gp, list):
            gp = []
        parts: list[str] = []
        for x in gp:
            if isinstance(x, dict):
                st = str(x.get("structure") or "—").strip()
                fm = str(x.get("formula") or "").strip()
                parts.append(f"• {st}" + (f": {fm}" if fm else ""))
            else:
                s = str(x).strip()
                if s:
                    parts.append(f"• {s}")
        body = "✏️ Граматика / структури:\n" + "\n".join(parts)
        return header + body

    ideas = preview_data.get("key_ideas")
    if not isinstance(ideas, list):
        ideas = []
    ideas = [str(x).strip() for x in ideas if str(x).strip()]
    while len(ideas) < 3:
        ideas.append("—")
    ideas = ideas[:6]
    words = preview_data.get("words")
    if not isinstance(words, list):
        words = []
    words = [str(x).strip() for x in words if str(x).strip()][:15]
    words_str = ", ".join(words) if words else "—"
    lines = [
        header,
        "💡 Ідеї:\n",
        "\n".join(f"• {x}" for x in ideas) + "\n\n",
        f"📚 Слова:\n{words_str}",
    ]
    qn = preview_data.get("questions")
    if isinstance(qn, list) and qn:
        lines.append(
            "\n\n❓ Питання:\n"
            + "\n".join(f"• {str(x).strip()}" for x in qn if str(x).strip())
        )
    ex = preview_data.get("exercises")
    if isinstance(ex, list) and ex:
        lines.append(
            "\n\n🏋 Вправи:\n"
            + "\n".join(f"• {str(x).strip()}" for x in ex if str(x).strip())
        )
    return "".join(lines)


class MessageHandlerService:
    def __init__(
        self,
        pipeline: ContentPipelineService,
        deduplicator: MessageDeduplicator,
        youtube_service: YouTubeTranscriptService,
        openai_client: AsyncOpenAI,
        openai_model: str,
    ) -> None:
        self._pipeline = pipeline
        self._deduplicator = deduplicator
        self._youtube_service = youtube_service
        self._openai_client = openai_client
        self._openai_model = openai_model

    async def _call_a1_filter_gpt(self, transcript_snippet: str) -> str:
        user_content = (
            f"{_PREVIEW_A1_FILTER_USER}\n\n---\n\nTranscript to filter:\n{transcript_snippet}"
        )
        response = await self._openai_client.chat.completions.create(
            model=self._openai_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _PREVIEW_A1_FILTER_SYSTEM},
                {"role": "user", "content": user_content},
            ],
        )
        raw = response.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        topic, scenes = _coerce_a1_filter_output(data)
        return _a1_resolved_filtered_block(topic, scenes)

    async def _call_a2_filter_gpt(self, transcript_snippet: str) -> str:
        user_content = (
            f"{_PREVIEW_A2_FILTER_USER}\n\n---\n\nTranscript to filter:\n{transcript_snippet}"
        )
        response = await self._openai_client.chat.completions.create(
            model=self._openai_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _PREVIEW_A2_FILTER_SYSTEM},
                {"role": "user", "content": user_content},
            ],
        )
        raw = response.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        topic, scenes = _coerce_a2_filter_output(data)
        return _a2_resolved_filtered_block(topic, scenes)

    async def _call_b1_filter_gpt(self, transcript_snippet: str) -> str:
        user_content = (
            f"{_PREVIEW_B1_FILTER_USER}\n\n---\n\nTranscript to filter:\n{transcript_snippet}"
        )
        response = await self._openai_client.chat.completions.create(
            model=self._openai_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _PREVIEW_B1_FILTER_SYSTEM},
                {"role": "user", "content": user_content},
            ],
        )
        raw = response.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        topic, scenes = _coerce_b1_filter_output(data)
        return _b1_resolved_filtered_block(topic, scenes)

    async def _call_b2_filter_gpt(self, transcript_snippet: str) -> str:
        user_content = (
            f"{_PREVIEW_B2_FILTER_USER}\n\n---\n\nTranscript to filter:\n{transcript_snippet}"
        )
        response = await self._openai_client.chat.completions.create(
            model=self._openai_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _PREVIEW_B2_FILTER_SYSTEM},
                {"role": "user", "content": user_content},
            ],
        )
        raw = response.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        topic, scenes = _coerce_b2_filter_output(data)
        return _b2_resolved_filtered_block(topic, scenes)

    async def _call_speaking_filter_gpt(self, transcript_snippet: str) -> str:
        user_content = (
            f"{_PREVIEW_SPEAKING_FILTER_USER}\n\n---\n\nTranscript to filter:\n{transcript_snippet}"
        )
        response = await self._openai_client.chat.completions.create(
            model=self._openai_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _PREVIEW_SPEAKING_FILTER_SYSTEM},
                {"role": "user", "content": user_content},
            ],
        )
        raw = response.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        topic, scenes = _coerce_speaking_filter_output(data)
        return _speaking_resolved_filtered_block(topic, scenes)

    async def _call_preview_gpt(
        self,
        transcript: str,
        format_key: Optional[str] = None,
        extra_instruction: Optional[str] = None,
        level: Optional[str] = None,
    ) -> dict[str, Any]:
        kind = _preview_format_kind(format_key)
        system = _preview_system_for_initial(kind, level)
        snippet = transcript.strip()
        if len(snippet) > _PREVIEW_TRANSCRIPT_MAX:
            snippet = snippet[:_PREVIEW_TRANSCRIPT_MAX]
        if kind == "lesson" and _is_lesson_cefr_a1(level):
            user_block = await self._call_a1_filter_gpt(snippet)
        elif kind == "lesson" and _is_lesson_cefr_a2(level):
            user_block = await self._call_a2_filter_gpt(snippet)
        elif kind == "lesson" and _is_lesson_cefr_b1(level):
            user_block = await self._call_b1_filter_gpt(snippet)
        elif kind == "lesson" and _is_lesson_cefr_b2(level):
            user_block = await self._call_b2_filter_gpt(snippet)
        elif kind == "speaking":
            user_block = await self._call_speaking_filter_gpt(snippet)
        else:
            user_block = f"Transcript:\n{snippet}"
        if extra_instruction and extra_instruction.strip():
            user_block += f"\n\nAdditional instruction:\n{extra_instruction.strip()}"
        response = await self._openai_client.chat.completions.create(
            model=self._openai_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_block},
            ],
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        if not isinstance(data, dict):
            data = {}
        return _normalize_preview_output(data, kind, level)

    async def _call_preview_patch_gpt(
        self,
        transcript: str,
        preview_data: dict[str, Any],
        teacher_text: str,
        *,
        refine_mode: str = "easy",
        custom_correction: bool = False,
        preview_format: Optional[str] = None,
        preview_level: Optional[str] = None,
    ) -> dict[str, Any]:
        snippet = transcript.strip()
        if len(snippet) > _PREVIEW_TRANSCRIPT_MAX:
            snippet = snippet[:_PREVIEW_TRANSCRIPT_MAX]
        pd_in = preview_data if isinstance(preview_data, dict) else {}
        patch_kind = _preview_format_kind(preview_format)
        if custom_correction:
            rules_block = _preview_patch_rules_custom(patch_kind, preview_level)
        elif refine_mode == "deep":
            rules_block = _preview_patch_rules_deep(patch_kind, preview_level)
        else:
            rules_block = _preview_patch_rules_easy(patch_kind, preview_level)
        LOGGER.info(
            "preview_patch_gpt kind=%s refine_mode=%s custom=%s transcript_len=%s "
            "preview_data=%s teacher_instruction=%s",
            patch_kind,
            "custom" if custom_correction else refine_mode,
            custom_correction,
            len(snippet),
            json.dumps(pd_in, ensure_ascii=False, default=str),
            teacher_text.strip()[:2000],
        )
        user_content = _build_preview_patch_user_content(
            snippet,
            pd_in,
            teacher_text.strip(),
            rules_block,
            patch_kind,
        )
        response = await self._openai_client.chat.completions.create(
            model=self._openai_model,
            response_format={"type": "json_object"},
            temperature=0.75,
            messages=[
                {"role": "system", "content": _PREVIEW_PATCH_SYSTEM},
                {"role": "user", "content": user_content},
            ],
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        if not isinstance(data, dict):
            data = {}
        normalized = _normalize_preview_output(data, patch_kind, preview_level)
        if patch_kind == "speaking":
            st = str(normalized.get("speaking_task") or "").strip()
            if not st or st == "—":
                prev_t = pd_in.get("speaking_task")
                if isinstance(prev_t, str) and prev_t.strip() and prev_t.strip() != "—":
                    normalized["speaking_task"] = prev_t.strip()
        if custom_correction:
            for k in _preview_merge_list_keys(patch_kind, preview_level):
                if k in normalized:
                    continue
                prev = pd_in.get(k)
                if isinstance(prev, list) and prev:
                    if k == "grammar_patterns":
                        normalized[k] = _coerce_grammar_patterns(prev)
                    elif k == "vocabulary_items":
                        normalized[k] = _coerce_vocabulary_items(prev)
                    else:
                        normalized[k] = [
                            str(x).strip()
                            for x in prev
                            if str(x).strip() and str(x).strip() != "—"
                        ][:25]
        return normalized

    def _guided_ready(self, chat_id: int) -> bool:
        st = user_state.get(chat_id)
        return bool(st and st.get("format") and st.get("level"))

    def _preview_state_bootstrap(self) -> dict[str, Any]:
        return {
            "transcript": None,
            "format": None,
            "level": None,
            "preview_data": {},
            "generating": False,
            "preview_message_id": None,
            "awaiting_edit": False,
            "edit_rounds": 0,
            "limit_reached": False,
            "confirmed": False,
        }

    async def _edit_or_reply_preview(
        self,
        bot: Any,
        chat_id: int,
        prv: dict[str, Any],
        anchor_message: Message,
        text: str,
        reply_markup: InlineKeyboardMarkup,
    ) -> None:
        mid = prv.get("preview_message_id")
        try:
            if mid:
                await bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=int(mid),
                    text=text,
                    reply_markup=reply_markup,
                )
                return
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Preview edit failed, sending new message: %s", exc)
        sent = await anchor_message.reply_text(text, reply_markup=reply_markup)
        prv["preview_message_id"] = sent.message_id

    async def _after_refine_increment(
        self,
        bot: Any,
        chat_id: int,
        prv: dict[str, Any],
        anchor_message: Message,
        preview_data: dict[str, Any],
    ) -> None:
        prv["preview_data"] = preview_data
        prv["edit_rounds"] = int(prv.get("edit_rounds") or 0) + 1
        prv["awaiting_edit"] = False
        prv["limit_reached"] = False
        if prv["edit_rounds"] >= _MAX_PREVIEW_EDIT_ROUNDS:
            prv["limit_reached"] = True
            await self._edit_or_reply_preview(
                bot,
                chat_id,
                prv,
                anchor_message,
                _PREVIEW_LIMIT_TEXT,
                _PREVIEW_LIMIT_KB,
            )
        else:
            await self._edit_or_reply_preview(
                bot,
                chat_id,
                prv,
                anchor_message,
                _format_preview_message(preview_data, prv.get("format")),
                _PREVIEW_KB,
            )

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return
        chat_id = update.message.chat_id
        user_state.pop(chat_id, None)
        preview_state.pop(chat_id, None)
        await update.message.reply_text(
            "👋 Привіт! Я Veliora 🎓\n\n"
            "Допоможу швидко підготувати матеріал для уроку англійської.\n\n"
            "Що хочеш отримати?",
            reply_markup=_ONB_FMT_STEP1_KB,
        )

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if not query or not query.message:
            return
        await query.answer()
        chat_id = query.message.chat_id
        data = (query.data or "").strip()

        if data == "onb_prv_ok":
            prv = preview_state.get(chat_id)
            if not prv or not prv.get("transcript"):
                await query.message.reply_text(
                    "Не знайшов збережене джерело. Надішли посилання на YouTube ще раз."
                )
                return
            if prv.get("generating"):
                return
            prv["generating"] = True
            prv["confirmed"] = True
            st = user_state.get(chat_id)
            fmt = (st or {}).get("format") or prv.get("format")
            lvl = (st or {}).get("level") or prv.get("level")
            transcript = str(prv.get("transcript") or "").strip()
            preview_data = prv.get("preview_data") or {}
            kind = _preview_format_kind(fmt)

            if kind == "lesson":
                cq = preview_data.get("core_questions")
                approved_block = (
                    f"APPROVED PREVIEW:\n"
                    f"TOPIC: {preview_data.get('topic', '')}\n"
                    f"WARMUP QUESTIONS: {preview_data.get('warmup_questions', [])}\n"
                )
                if isinstance(cq, list) and any(
                    str(x).strip() and str(x).strip() != "—" for x in cq
                ):
                    approved_block += f"CORE QUESTIONS: {cq}\n"
                approved_block += (
                    f"CHOICES: {preview_data.get('choices', [])}\n"
                    f"SUPPORT WORDS: {preview_data.get('support_words', [])}\n"
                )
            elif kind == "vocabulary":
                approved_block = (
                    f"APPROVED PREVIEW:\n"
                    f"TOPIC: {preview_data.get('topic', '')}\n"
                    f"VOCABULARY: {preview_data.get('vocabulary_items', [])}\n"
                )
            elif kind == "phrases":
                approved_block = (
                    f"APPROVED PREVIEW:\n"
                    f"TOPIC: {preview_data.get('topic', '')}\n"
                    f"PATTERNS: {preview_data.get('grammar_patterns', [])}\n"
                )
            elif kind == "questions":
                approved_block = (
                    f"APPROVED PREVIEW:\n"
                    f"TOPIC: {preview_data.get('topic', '')}\n"
                    f"DISCUSSION QUESTIONS: {preview_data.get('discussion_questions', [])}\n"
                )
            elif kind == "speaking":
                approved_block = (
                    f"APPROVED PREVIEW:\n"
                    f"TOPIC: {preview_data.get('topic', '')}\n"
                    f"DISCUSSION QUESTIONS: {preview_data.get('discussion_questions', [])}\n"
                    f"SPEAKING TASK: {preview_data.get('speaking_task', '')}\n"
                )
            else:
                approved_block = (
                    f"APPROVED PREVIEW:\n"
                    f"TOPIC: {preview_data.get('topic', '')}\n"
                    f"IDEAS: {preview_data.get('key_ideas', [])}\n"
                    f"WORDS: {preview_data.get('words', [])}\n"
                )

            structured = (
                f"[FORMAT={fmt}]\n[LEVEL={lvl}]\n\n"
                f"{approved_block}\n"
                f"SOURCE TRANSCRIPT (reference only):\n"
                f"{transcript}"
            )
            proxy = _OnboardingEnrichedMessage(query.message, structured)
            try:
                prepare = await self._pipeline.prepare(context.bot, proxy, chat_id)
                if prepare.preface:
                    await query.message.reply_text("Вже готую твій матеріал ✨")
                elif prepare.status_line:
                    await query.message.reply_text(prepare.status_line)
                result = await self._pipeline.execute(prepare)
            except TranscriptUnavailableError as err:
                prv["generating"] = False
                await query.message.reply_text(err.user_message)
                return
            except GenerationFailedError as err:
                prv["generating"] = False
                await query.message.reply_text(err.user_message)
                return
            except Exception as error:  # noqa: BLE001
                LOGGER.exception("Guided confirm pipeline failed: %s", error)
                prv["generating"] = False
                await query.message.reply_text("Не вдалося згенерувати картку. Спробуй ще раз.")
                return
            preview_state.pop(chat_id, None)
            await self._send_pipeline_result(query.message, result)
            return

        if data == "onb_prv_new":
            preview_state.pop(chat_id, None)
            await query.message.reply_text("Надішли посилання на YouTube ще раз 👇")
            try:
                await query.edit_message_reply_markup(reply_markup=None)
            except Exception:  # noqa: BLE001
                pass
            return

        if data == "onb_prv_ref":
            prv = preview_state.get(chat_id)
            if not prv or not prv.get("transcript"):
                await query.message.reply_text(
                    "Не знайшов збережене джерело. Надішли посилання на YouTube ще раз."
                )
                return
            if prv.get("limit_reached") or int(prv.get("edit_rounds") or 0) >= _MAX_PREVIEW_EDIT_ROUNDS:
                return
            prv["awaiting_edit"] = False
            try:
                await query.edit_message_text(
                    "Як адаптувати матеріал?",
                    reply_markup=_PREVIEW_REFINE_KB,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Refine menu edit failed: %s", exc)
                sent = await query.message.reply_text(
                    "Як адаптувати матеріал?",
                    reply_markup=_PREVIEW_REFINE_KB,
                )
                prv["preview_message_id"] = sent.message_id
            return

        if data == "onb_prv_r_easy":
            prv = preview_state.get(chat_id)
            if not prv or not prv.get("transcript"):
                await query.message.reply_text(
                    "Не знайшов збережене джерело. Надішли посилання на YouTube ще раз."
                )
                return
            if int(prv.get("edit_rounds") or 0) >= _MAX_PREVIEW_EDIT_ROUNDS:
                return
            try:
                pd = await self._call_preview_patch_gpt(
                    str(prv["transcript"]),
                    prv.get("preview_data") or {},
                    _PREVIEW_INSTR_EASY,
                    refine_mode="easy",
                    preview_format=prv.get("format"),
                    preview_level=prv.get("level"),
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Preview GPT failed: %s", exc)
                await query.message.reply_text(
                    "Не вдалося оновити перегляд. Надішли джерело знову або спробуй пізніше."
                )
                return
            await self._after_refine_increment(
                context.bot, chat_id, prv, query.message, pd
            )
            return

        if data == "onb_prv_r_deep":
            prv = preview_state.get(chat_id)
            if not prv or not prv.get("transcript"):
                await query.message.reply_text(
                    "Не знайшов збережене джерело. Надішли посилання на YouTube ще раз."
                )
                return
            if int(prv.get("edit_rounds") or 0) >= _MAX_PREVIEW_EDIT_ROUNDS:
                return
            try:
                pd = await self._call_preview_patch_gpt(
                    str(prv["transcript"]),
                    prv.get("preview_data") or {},
                    _PREVIEW_INSTR_DEEP,
                    refine_mode="deep",
                    preview_format=prv.get("format"),
                    preview_level=prv.get("level"),
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Preview GPT failed: %s", exc)
                await query.message.reply_text(
                    "Не вдалося оновити перегляд. Надішли джерело знову або спробуй пізніше."
                )
                return
            await self._after_refine_increment(
                context.bot, chat_id, prv, query.message, pd
            )
            return

        if data == "onb_prv_r_own":
            prv = preview_state.get(chat_id)
            if not prv or not prv.get("transcript"):
                await query.message.reply_text(
                    "Не знайшов збережене джерело. Надішли посилання на YouTube ще раз."
                )
                return
            if int(prv.get("edit_rounds") or 0) >= _MAX_PREVIEW_EDIT_ROUNDS:
                return
            prv["awaiting_edit"] = True
            await query.message.reply_text(
                "Напиши одним реченням що змінити 👇\n"
                "Наприклад: 'фокус на Present Simple' або 'для бізнес теми'"
            )
            return

        if data == "onb_p_fmt":
            await query.edit_message_reply_markup(reply_markup=_POST_CARD_FMT_KB)
            return
        if data == "onb_p_lvl":
            await query.edit_message_reply_markup(reply_markup=_POST_CARD_LVL_KB)
            return
        if data.startswith("onb_p_f_"):
            fmt = data.removeprefix("onb_p_f_")
            label = _FMT_CHANGE_LABELS.get(fmt, fmt)
            st = user_state.setdefault(chat_id, {})
            st["format"] = fmt
            await query.message.reply_text(
                f"Формат змінено на: {label}\n"
                "Скинь YouTube-відео ще раз або напиши тему 👇"
            )
            await query.edit_message_reply_markup(reply_markup=_POST_CARD_KB)
            return
        if data.startswith("onb_p_l_"):
            level = data.removeprefix("onb_p_l_")
            st = user_state.setdefault(chat_id, {})
            st["level"] = level
            await query.message.reply_text(
                f"Рівень змінено на: {level}\n"
                "Скинь YouTube-відео ще раз або напиши тему 👇"
            )
            await query.edit_message_reply_markup(reply_markup=_POST_CARD_KB)
            return

        if data.startswith("onb_fmt_"):
            fmt = data.removeprefix("onb_fmt_")
            user_state[chat_id] = {"format": fmt, "level": None}
            preview_state.pop(chat_id, None)
            await query.edit_message_text(
                "Для якого рівня?",
                reply_markup=_ONB_LEVEL_KB,
            )
            return

        if data.startswith("onb_lvl_"):
            level = data.removeprefix("onb_lvl_")
            st = user_state.get(chat_id)
            if not st or not st.get("format"):
                await query.edit_message_text("Натисни /start, щоб почати спочатку.")
                return
            st["level"] = level
            preview_state.pop(chat_id, None)
            await query.edit_message_text("Супер. Скинь YouTube-відео або напиши тему 👇")
            return

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return

        message = update.message
        chat_id = message.chat_id
        message_id = message.message_id

        if await self._deduplicator.is_duplicate(chat_id, message_id):
            LOGGER.info("Skipping duplicate message chat_id=%s message_id=%s", chat_id, message_id)
            return

        st = user_state.get(chat_id)
        original_content = (message.text or message.caption or "").strip()

        prv_early = preview_state.get(chat_id)
        if prv_early and prv_early.get("awaiting_edit"):
            if message.voice:
                await message.reply_text(
                    "Напиши одним реченням текстом, що змінити 👇"
                )
                return
            if not original_content:
                return
            video_id_early = extract_video_id(original_content)
            if video_id_early:
                prv_early["awaiting_edit"] = False
            else:
                if not prv_early.get("transcript"):
                    preview_state.pop(chat_id, None)
                    await message.reply_text(
                        "Не знайшов збережене джерело. Надішли посилання на YouTube ще раз."
                    )
                    return
                raw_pd = prv_early.get("preview_data")
                pd_for_patch = dict(raw_pd) if isinstance(raw_pd, dict) else {}
                tr = str(prv_early["transcript"])
                LOGGER.info(
                    "handler=guided_preview.awaiting_edit chat_id=%s transcript_len=%s "
                    "preview_data=%s teacher_instruction=%s",
                    chat_id,
                    len(tr),
                    json.dumps(pd_for_patch, ensure_ascii=False, default=str),
                    original_content[:2000],
                )
                try:
                    pd = await self._call_preview_patch_gpt(
                        tr,
                        pd_for_patch,
                        original_content,
                        custom_correction=True,
                        preview_format=prv_early.get("format"),
                        preview_level=prv_early.get("level"),
                    )
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("Preview GPT failed (awaiting edit): %s", exc)
                    prv_early["awaiting_edit"] = False
                    await message.reply_text(
                        "Не вдалося оновити перегляд. Надішли джерело знову або спробуй пізніше."
                    )
                    return
                await self._after_refine_increment(
                    context.bot, chat_id, prv_early, message, pd
                )
                return

        if (
            self._guided_ready(chat_id)
            and not message.voice
            and original_content
        ):
            video_id = extract_video_id(original_content)
            if video_id:
                base = self._preview_state_bootstrap()
                base["format"] = st.get("format")
                base["level"] = st.get("level")
                preview_state[chat_id] = base
                try:
                    transcript = await self._youtube_service.fetch_transcript(video_id)
                except TranscriptUnavailableError as err:
                    preview_state.pop(chat_id, None)
                    await message.reply_text(err.user_message)
                    return
                preview_state[chat_id]["transcript"] = transcript
                try:
                    preview_data = await self._call_preview_gpt(
                        transcript,
                        format_key=st.get("format"),
                        level=st.get("level"),
                    )
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("Preview GPT failed: %s", exc)
                    preview_state.pop(chat_id, None)
                    await message.reply_text(
                        "Не вдалося зробити попередній перегляд. Спробуй ще раз."
                    )
                    return
                preview_state[chat_id]["preview_data"] = preview_data
                sent = await message.reply_text(
                    _format_preview_message(preview_data, st.get("format")),
                    reply_markup=_PREVIEW_KB,
                )
                preview_state[chat_id]["preview_message_id"] = sent.message_id
                return

        pipeline_message: Union[Message, _OnboardingEnrichedMessage] = message
        if (
            st
            and st.get("format")
            and st.get("level")
            and not message.voice
        ):
            if original_content:
                enriched = _enriched_onboarding_transcript_block(
                    st.get("format"),
                    st.get("level"),
                    original_content,
                )
                pipeline_message = _OnboardingEnrichedMessage(message, enriched)

        try:
            prepare = await self._pipeline.prepare(context.bot, pipeline_message, chat_id)
        except NeedActiveSourceError:
            await message.reply_text(
                "Спочатку надішли матеріал: посилання YouTube, текст або голосове повідомлення. "
                "Потім можна написати, наприклад: «зроби картку»."
            )
            return
        except TranscriptUnavailableError as err:
            await message.reply_text(err.user_message)
            return
        except GenerationFailedError as err:
            await message.reply_text(err.user_message)
            return
        except ValueError as err:
            LOGGER.warning("Invalid user input message_id=%s: %s", message_id, err)
            await message.reply_text(
                str(err) or "Надішли текст, голос або посилання YouTube."
            )
            return
        except Exception as error:  # noqa: BLE001
            LOGGER.exception("Prepare failed message_id=%s: %s", message_id, error)
            await message.reply_text("Щось пішло не так. Спробуй ще раз.")
            return

        if prepare.preface:
            await message.reply_text("Вже готую твій матеріал ✨")
        elif prepare.status_line:
            await message.reply_text(prepare.status_line)

        try:
            result = await self._pipeline.execute(prepare)
        except GenerationFailedError as err:
            await message.reply_text(err.user_message)
            return
        except Exception as error:  # noqa: BLE001
            LOGGER.exception("Execute failed message_id=%s: %s", message_id, error)
            await message.reply_text("Не вдалося завершити картку. Спробуй ще раз за хвилину.")
            return

        await self._send_pipeline_result(message, result)

    async def _send_pipeline_result(self, message, result) -> None:
        if result.image_bytes:
            image_file = InputFile(result.image_bytes, filename="educard.png")
            await message.reply_photo(
                photo=image_file,
                caption=f"Картка · {result.template_used} · {result.source_type}",
                reply_markup=_POST_CARD_KB,
            )
        elif result.text_fallback:
            await message.reply_text(result.text_fallback)
        else:
            LOGGER.error("Pipeline returned neither image nor text")
            await message.reply_text(
                "Не вдалося показати прев’ю. Спробуй ще раз."
            )
