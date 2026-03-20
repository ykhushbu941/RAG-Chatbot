"""
Wattmonk RAG Chatbot — FastAPI Backend (Groq)
Handles: RAG pipeline, chunk retrieval, intent classification, Groq API calls
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from groq import Groq
import os
import re
from dotenv import load_dotenv

load_dotenv()

# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Wattmonk RAG Chatbot API",
    description="RAG-based chatbot for Wattmonk + NEC 2017 knowledge base (Groq)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Knowledge Base ───────────────────────────────────────────────────────────

KNOWLEDGE_BASE = {
    "wattmonk": """
=== WATTMONK COMPANY INFORMATION ===

OVERVIEW:
Wattmonk is a modern solar engineering and technology company established in 2019, dedicated to driving the global shift toward clean, renewable energy. Headquartered in Delaware, USA, with offices in Gurugram, India and Singapore. Founded by Ankit Sheoran who has been in the solar industry for over a decade. Wattmonk describes itself as "Your Everyday Solar Store".

KEY FACTS:
- Established: 2019
- Headquarters: Delaware, USA
- Global Presence: Services across all 50 US states, teams in India and Singapore
- Employees: 201-500 (380+ skilled professionals)
- Monthly output: 5,000+ permits/month, 60 MW solar projects/month
- Portfolio: 300 MW in 2022 in the US alone
- Growth: 300% year-over-year
- Serves 5 out of 10 solar installers in the US
- Database of 6,500+ AHJs and 500+ utility providers
- Provides design support for 20,000+ homes monthly
- Cost savings: Up to 30% on solar project installation costs

SERVICES:
1. Sales Proposal: 2-hour turnaround for solar proposals and shade analysis.
2. Site Survey App: Comprehensive solar site survey tool. Data auto-feeds into plan sets.
3. Permit Plan Sets: Delivered in 6 hours. Includes single and three-line diagrams, roof plans, attachment details, spec sheets. Database includes 6,500+ AHJs.
4. Engineering Review / PE Stamping: AHJ and Utility-compliant reviews within 24 hours.
5. PTO Application / Interconnection: Dedicated team handles interconnection and Permission To Operate applications.
6. Permitting Support: Full guidance from application submission to final approval.

TECHNOLOGY:
- Zippy: Machine-learning-powered semi-automated plan set tool. Automates single-line and three-line diagram creation and electrical/structural calculations.
- Real-Time Project Management: Track projects and tasks in real time. Master Details dashboard for project monitoring.
- All-in-one platform: Information auto-feeds between services in a single click.

WHO WATTMONK SERVES:
- Solar professionals, EPCs, developers, dealers, and installers
- Residential and commercial projects: rooftops, ground mounts, carports

MISSION AND VALUES:
- Mission: Increase solar adoption worldwide by supercharging solar installers
- Vision: Revolutionize how the world harnesses clean energy
- Values: Sustainability, efficiency, technology leadership, customer-centric service, diversity and inclusion
- Culture: Solar Warriors encouraged to innovate and challenge norms

MARKET STATS:
- Delivers 3,500+ permit plans/month
- 1,500+ PE Structural and Electrical Reviews/month
- 5,000+ Sales Proposals/month
- 600+ PTO and Interconnection Applications/month
- Successfully worked on 40,000+ residential projects in 2022
- Fuels 60 MW of solar projects every month
""",

    "nec": """
=== NEC 2017 (NFPA 70) SOLAR PV RELEVANT SECTIONS ===

ARTICLE 690: SOLAR PHOTOVOLTAIC (PV) SYSTEMS

690.1 SCOPE:
Applies to solar PV systems including array circuits, inverters, and controllers. Systems may be interactive with other electrical power sources or stand-alone, and may or may not be connected to energy storage systems. PV systems may have ac or dc output.

690.2 KEY DEFINITIONS:
- PV System: Total components and subsystems that convert solar energy into electric energy.
- Array: Mechanically integrated assembly of modules or panels with support structure and tracker.
- Module: Complete environmentally protected unit of solar cells designed to generate dc power.
- Panel: Collection of modules mechanically fastened together as a field-installable unit.
- Solar Cell: Basic PV device that generates electricity when exposed to light.
- Interactive System: A PV system that operates in parallel with and may deliver power to an electrical production and distribution network.
- Stand-Alone System: A solar PV system that supplies power independently of an electrical production and distribution network.
- Inverter: Equipment used to change voltage level or waveform; commonly converts dc input to ac output.
- DC-to-DC Converter: Device providing output dc voltage and current at a higher or lower value than input.
- Generating Capacity: Sum of parallel-connected inverter maximum continuous output power at 40 degrees C in kilowatts.

690.4 GENERAL REQUIREMENTS:
- (A) PV systems permitted to supply a building in addition to any other electrical supply systems.
- (B) Equipment: Inverters, PV modules, PV panels, dc combiners, dc-to-dc converters, and charge controllers shall be listed or field labeled for PV application.
- (C) Qualified Personnel: Installation shall be performed only by qualified persons.
- (D) Multiple PV Systems permitted in/on a single building with directory per 705.10.
- (E) Locations Not Permitted: PV system equipment shall NOT be installed in bathrooms.

690.7 MAXIMUM VOLTAGE:
- One- and two-family dwellings: maximum voltage of 600 volts or less.
- Other building types: maximum voltage of 1000 volts or less.
- Not located on/in buildings: listed dc PV equipment rated at 1500 volts or less.
- Temperature correction factors apply for crystalline/multicrystalline silicon modules per Table 690.7(A).
- For PV systems 100 kW or larger: stamped design by a licensed professional electrical engineer is permitted.

690.8 CIRCUIT SIZING AND CURRENT:
- PV Source Circuit: Sum of parallel module short-circuit currents multiplied by 125 percent.
- PV Output Circuit: Sum of parallel source circuit maximum currents.
- Inverter Output Circuit: Inverter continuous output current rating.
- Conductor Ampacity: Must be at least 125 percent of maximum calculated currents.
- Combined 125% factors result in 156 percent multiplication factor.

690.9 OVERCURRENT PROTECTION:
- PV system dc circuit and inverter output conductors shall be protected against overcurrent.
- Circuits connected to current-limited supplies and higher current sources shall be protected at the higher current source.

ARTICLE 250 GROUNDING AND BONDING:
Covers system grounding, grounding electrode system, bonding, and equipment grounding conductors relevant to PV systems.

ARTICLE 705 INTERCONNECTED ELECTRIC POWER PRODUCTION SOURCES:
Directory requirements when multiple PV systems are on same structure.

ARTICLE 691 LARGE-SCALE PV ELECTRIC POWER PRODUCTION FACILITY:
Covers large-scale PV stations excluded from Article 690.

NEC ARTICLE 110 GENERAL REQUIREMENTS:
- 110.3(B): Listed or labeled equipment shall be installed and used per instructions in the listing or labeling.
- Qualified Person: One who has skills and knowledge related to electrical equipment and has received safety training.
""",

    "general": """
=== GENERAL SOLAR INDUSTRY KNOWLEDGE ===

SOLAR INSTALLATION PROCESS:
1. Site Assessment and Sales Proposal
2. Engineering Design and Permit Plans
3. Permit Submission and AHJ Approval
4. Installation
5. Electrical Inspection
6. Interconnection Application
7. Utility Inspection
8. Permission to Operate (PTO)
9. System Monitoring

COMMON SOLAR ACRONYMS:
- AHJ: Authority Having Jurisdiction
- PTO: Permission to Operate
- PV: Photovoltaic
- EPC: Engineering, Procurement and Construction
- NEC: National Electrical Code (NFPA 70)
- PE: Professional Engineer
- SLD: Single-Line Diagram
- 3LD: Three-Line Diagram
- AC: Alternating Current
- DC: Direct Current
- MPPT: Maximum Power Point Tracker
- NFPA: National Fire Protection Association
- OCPD: Overcurrent Protective Device
- GEC: Grounding Electrode Conductor
- EGC: Equipment Grounding Conductor

TYPES OF SOLAR PROJECTS:
- Residential Rooftop: 5-15 kW systems on homes; 600V max per NEC 690.7
- Commercial Rooftop: Larger systems on commercial buildings; up to 1000V
- Ground Mount: Systems installed on the ground
- Carport: Solar panels installed over parking areas
- Utility-Scale: MW-range systems per NEC Article 691; up to 1500V DC

KEY SOLAR COMPONENTS:
- PV Modules: Solar panels that convert sunlight to DC electricity
- Inverter: Converts DC to AC electricity
- Racking/Mounting: Structural system holding panels
- Disconnect: Safety switch required by NEC to isolate the PV system
- Combiner Box: Combines multiple string circuits into one output
- Monitoring System: Tracks energy production and system health
"""
}

# ─── RAG Pipeline ─────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + chunk_size])
        i += chunk_size - overlap
    return chunks


def score_chunk(query: str, chunk: str) -> float:
    tokens = [t for t in re.split(r"\W+", query.lower()) if len(t) > 2]
    chunk_lower = chunk.lower()
    score = 0.0
    for token in tokens:
        matches = re.findall(re.escape(token), chunk_lower)
        score += len(matches)
    return score


def retrieve_chunks(query: str, top_k: int = 5) -> List[dict]:
    all_chunks = []
    for source, text in KNOWLEDGE_BASE.items():
        for chunk in chunk_text(text):
            all_chunks.append({"source": source, "chunk": chunk})
    scored = [{**c, "score": score_chunk(query, c["chunk"])} for c in all_chunks]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return [c for c in scored[:top_k] if c["score"] > 0]


def classify_intent(query: str) -> str:
    q = query.lower()
    nec_keywords = [
        "nec", "code", "article", "690", "705", "250", "ampacity", "overcurrent",
        "grounding", "bonding", "voltage", "wiring", "conductor", "inverter",
        "circuit", "disconnecting", "nfpa", "electrical code", "listed", "qualified",
        "short-circuit", "maximum current", "600 volt", "1000 volt"
    ]
    watt_keywords = [
        "wattmonk", "company", "service", "proposal", "planset", "plan set",
        "permit", "survey", "pto", "interconnection", "zippy", "ankit",
        "founded", "employees", "office", "turnaround", "pe stamp", "solar store"
    ]
    nec_score = sum(1 for k in nec_keywords if k in q)
    watt_score = sum(1 for k in watt_keywords if k in q)
    if nec_score > watt_score:
        return "nec"
    elif watt_score > nec_score:
        return "wattmonk"
    elif nec_score > 0 or watt_score > 0:
        return "both"
    return "general"


def build_system_prompt(intent: str, chunks: List[dict]) -> str:
    context_block = ""
    if chunks:
        context_parts = [
            f"[Source: {c['source'].upper()} | Score: {c['score']:.0f}]\n{c['chunk']}"
            for c in chunks
        ]
        context_block = "RETRIEVED CONTEXT FROM KNOWLEDGE BASE:\n\n" + "\n\n---\n\n".join(context_parts)

    return f"""You are the Wattmonk AI Assistant, a RAG-powered chatbot with expertise in:
1. Wattmonk Technologies - solar engineering company (founded 2019, Delaware USA)
2. NEC 2017 (NFPA 70) - especially Article 690 on Solar PV Systems
3. General solar industry knowledge

RESPONSE RULES:
- Use the provided context chunks to answer accurately
- Always cite your source: [Wattmonk Info], [NEC 2017 Article 690], or [General Knowledge]
- If context does not contain enough information, say so honestly
- For NEC questions: be precise with article numbers and section letters
- For Wattmonk questions: use specific services, turnaround times, and statistics
- Keep answers clear, structured, and professional

DETECTED QUERY INTENT: {intent.upper()}

{context_block}
"""

# ─── Models ───────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[Message] = []


class ChunkInfo(BaseModel):
    source: str
    score: float
    preview: str


class ChatResponse(BaseModel):
    reply: str
    intent: str
    chunks_used: int
    retrieved_chunks: List[ChunkInfo]
    source_label: str


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "Wattmonk RAG Chatbot API (Groq) is running"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "knowledge_sources": list(KNOWLEDGE_BASE.keys()),
        "model": "llama-3.3-70b-versatile"
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set in environment")

    # RAG pipeline
    intent = classify_intent(request.message)
    retrieved = retrieve_chunks(request.message, top_k=5)
    system_prompt = build_system_prompt(intent, retrieved)

    # Build messages for Groq (OpenAI-compatible format)
    messages = [{"role": "system", "content": system_prompt}]
    for m in request.history[-8:]:
        messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": request.message})

    # Call Groq API
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=1000,
            temperature=0.7,
        )
        reply = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API error: {str(e)}")

    source_labels = {
        "nec": "NEC 2017",
        "wattmonk": "Wattmonk Info",
        "both": "Wattmonk + NEC",
        "general": "General Knowledge",
    }

    return ChatResponse(
        reply=reply,
        intent=intent,
        chunks_used=len(retrieved),
        retrieved_chunks=[
            ChunkInfo(
                source=c["source"],
                score=c["score"],
                preview=c["chunk"][:200] + "..."
            )
            for c in retrieved
        ],
        source_label=source_labels.get(intent, "General"),
    )


@app.post("/retrieve")
def retrieve(query: str, top_k: int = 5):
    chunks = retrieve_chunks(query, top_k=top_k)
    intent = classify_intent(query)
    return {
        "query": query,
        "intent": intent,
        "chunks": [
            {
                "source": c["source"],
                "score": c["score"],
                "preview": c["chunk"][:300]
            }
            for c in chunks
        ],
    }
