# 🧠 Text World Agent — World Model Architecture

> **HackTronix 2.0 — Track B: Text World Agent**
>
> An AI agent that navigates text adventure games using a **persistent, structured world model**
> instead of conversation memory. The agent receives only an **objective + current world slice**,
> never full history.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    ACTION LOOP                          │
│                                                         │
│   ┌──────────┐    ┌───────────┐    ┌──────────────┐    │
│   │  Text     │───▸│ Extractor │───▸│ World Model  │    │
│   │  Env      │    │ (JSON)    │    │ (SQLite +    │    │
│   │           │    │           │    │  NetworkX)   │    │
│   └──────────┘    └───────────┘    └──────┬───────┘    │
│        ▲                                   │            │
│        │                                   ▼            │
│   ┌──────────┐    ┌───────────┐    ┌──────────────┐    │
│   │  Action   │◂──│ LLM Agent │◂──│ Query Layer  │    │
│   │  Execute  │    │ (Gemma2)  │    │ (World Slice)│    │
│   └──────────┘    └───────────┘    └──────────────┘    │
│        │                                   ▲            │
│        │          ┌───────────┐             │            │
│        └─────────▸│  Updater  │────────────┘            │
│                   │ (Reconcile)│                         │
│                   └───────────┘                         │
└─────────────────────────────────────────────────────────┘
```

### Data Flow (per step)

1. **Environment** → emits a text observation (room description, action result)
2. **Extractor** → parses text into structured JSON (`ExtractionResult`)
3. **World Model** → updates SQLite via the **Reconciler** (upserts, deduplication, fact superseding)
4. **Query Engine** → builds a minimal **World Slice** (~250 tokens) from the DB
5. **LLM Agent** → receives `objective + world_slice`, outputs exactly one action
6. **Action Parser** → fuzzy-matches LLM output to valid actions
7. **Updater** → updates inventory, room state, facts after action execution
8. **Loop** → repeat from step 1

---

## 🗄️ Database Schema (ER Diagram)

```
┌─────────────────┐       ┌───────────────────┐       ┌─────────────────┐
│     rooms        │       │   connections      │       │    objects       │
├─────────────────┤       ├───────────────────┤       ├─────────────────┤
│ id (PK)         │◂──┐   │ id (PK)           │   ┌──▸│ id (PK)         │
│ name (UNIQUE)   │   ├──▸│ from_room_id (FK) │   │   │ name            │
│ description     │   │   │ to_room_id (FK)───┼──▸│   │ description     │
│ visited         │   │   │ direction         │   │   │ room_id (FK)────┼──▸rooms
│ visit_count     │   │   │ locked            │   │   │ portable        │
│ first_seen      │   │   │ lock_key          │   │   │ state           │
│ last_seen       │   │   │ UNIQUE(from,dir)  │   │   │ properties (JSON│
└─────────────────┘   │   └───────────────────┘   │   │ UNIQUE(name,room│
                      │                           │   └─────────────────┘
                      │   ┌───────────────────┐   │
                      │   │   inventory        │   │   ┌─────────────────┐
                      │   ├───────────────────┤   │   │     npcs         │
                      │   │ id (PK)           │   │   ├─────────────────┤
                      │   │ object_id (FK)────┼───┘   │ id (PK)         │
                      │   │ acquired_step     │       │ name (UNIQUE)   │
                      │   │ UNIQUE(object_id) │       │ description     │
                      │   └───────────────────┘       │ room_id (FK)────┼──▸rooms
                      │                               │ dialogue        │
                      │   ┌───────────────────┐       │ state           │
                      ├──▸│   room_states      │       └─────────────────┘
                      │   ├───────────────────┤
                      │   │ id (PK)           │       ┌─────────────────┐
                      │   │ room_id (FK)      │       │ observed_facts  │
                      │   │ state_key         │       ├─────────────────┤
                      │   │ state_value       │       │ id (PK)         │
                      │   │ updated_step      │       │ subject         │
                      │   │ UNIQUE(room,key)  │       │ predicate       │
                      │   └───────────────────┘       │ object          │
                      │                               │ confidence      │
                      │                               │ source_step     │
                      │                               │ superseded_by   │
                      │                               │ active          │
                      │                               │ UNIQUE(subj,pred│
                      │                               └─────────────────┘
                      │
                      │   ┌───────────────────┐
                      │   │  agent_memory      │
                      │   ├───────────────────┤
                      │   │ id (PK)           │
                      │   │ memory_type       │
                      │   │ content           │
                      │   │ relevance         │
                      │   │ created_step      │
                      │   │ active            │
                      │   └───────────────────┘
```

**8 normalized tables** with UPSERT-friendly unique constraints. Contradicting facts **update** existing rows via the superseding chain — never duplicate.

---

## 🔑 Key Design Decisions

### Why World Model over Conversation Memory?

| Aspect | Conversation Memory | World Model (ours) |
|--------|--------------------|--------------------|
| Context Growth | Linear (grows every turn) | Bounded (fixed-size DB) |
| Contradictions | Accumulate silently | Resolved via superseding |
| Queryable | No (dump everything) | Yes (SQL + relevance scoring) |
| LLM Input Size | Grows until context overflow | Fixed ~250 tokens |
| Persistence | Lost on restart | SQLite file survives |

### Why Rule-Based Extraction (not LLM)?

On CPU with Gemma2:2b, each LLM call takes 5-10 seconds. The rule-based extractor handles 90%+ of cases in <1ms, keeping the game loop at ~2s/step instead of ~15s.

### Why 250-Token World Slices?

Gemma2:2b has limited reasoning capacity. By surgically constructing a minimal context with only the relevant room, objects, inventory, and facts, we maximize the model's chance of choosing the correct action.

---

## 📁 Project Structure

```
Hackatronics/
├── main.py                           # CLI entry point
├── requirements.txt                  # Dependencies
├── app/
│   ├── config.py                     # Centralized configuration
│   ├── models/
│   │   └── schemas.py                # All Pydantic domain models
│   ├── database/
│   │   ├── schema.sql                # SQLite DDL (8 tables)
│   │   ├── connection.py             # DB connection manager
│   │   └── repository.py            # Repository pattern (CRUD + UPSERT)
│   ├── environment/
│   │   ├── base.py                   # GameEnvironment protocol
│   │   ├── custom_env.py             # Custom text adventure engine
│   │   └── worlds/
│   │       └── sample_world.json     # 8-room puzzle manor
│   ├── extractor/
│   │   └── extractor.py              # Rule-based observation parser
│   ├── world_model/
│   │   ├── world_model.py            # Central orchestrator
│   │   └── reconciler.py             # State reconciliation engine
│   ├── query_engine/
│   │   └── query_engine.py           # Minimal world slice builder
│   ├── agent/
│   │   ├── agent.py                  # LLM agent + heuristic fallback
│   │   └── action_parser.py          # Fuzzy action matching
│   ├── llm/
│   │   └── llm_client.py             # Ollama wrapper
│   └── utils/
│       └── token_counter.py          # Token budget enforcement
└── tests/
    ├── test_extractor.py
    ├── test_world_model.py
    ├── test_environment.py
    ├── test_query_engine.py
    ├── test_agent.py
    └── test_action_parser.py
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Ollama** installed and running (`ollama serve`)
- **Gemma2:2b** model pulled (`ollama pull gemma2:2b`)

### Installation

```bash
# Clone and install dependencies
cd Hackatronics
pip install -r requirements.txt
```

### Run the Agent

```bash
# Run with default sample world (The Enchanted Manor)
python main.py

# Custom options
python main.py --world app/environment/worlds/sample_world.json --steps 100 --db my_run.db

# Run without Ollama (uses smart heuristic fallback)
python main.py --steps 50
```

### Run Tests

```bash
python -m pytest tests/ -v
```

---

## 🎮 Sample World: The Enchanted Manor

An 8-room puzzle mansion where the agent must:

1. **Explore** → Entrance Hall, Hallway, Kitchen, Library, Cellar, Hidden Chamber, Upper Landing, Study
2. **Find clues** → Talk to the Ghost of Blackwood, read the journal
3. **Collect keys** → Brass key (under welcome mat), Iron key (kitchen)
4. **Unlock doors** → Library cabinet (brass key → silver amulet), Cellar gate (iron key → hidden chamber)
5. **Get the treasure** → Golden chalice from the hidden chamber
6. **Win** → Return to Entrance Hall with chalice + amulet, use the chalice

---

## 🧪 Self-Correcting World Model

The world model automatically handles contradictions:

```
Step 5:  FACT: "iron_gate state_is locked"     (confidence=1.0, active=TRUE)
Step 12: FACT: "iron_gate state_is unlocked"   (confidence=1.0, active=TRUE)
         → Old fact deactivated, superseded_by links to new fact
         → Query engine only sees: "iron_gate state_is unlocked"
```

**Growth control:**
- Max 50 active facts, max 20 active memories
- Memory relevance decays 0.95x per step
- Memories below 0.1 relevance are pruned every 20 steps
- Superseded facts kept for audit trail but excluded from queries

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| LLM | Gemma2:2b via Ollama |
| Database | SQLite (WAL mode) |
| Graph | NetworkX (room connectivity) |
| Models | Pydantic v2 |
| Console | Rich (colored panels) |
| Testing | pytest |
| API (optional) | FastAPI + Uvicorn |

---

## 📐 SOLID Principles Applied

- **S**ingle Responsibility: Each module has one job (Extractor extracts, Reconciler reconciles)
- **O**pen/Closed: `GameEnvironment` protocol allows new environments without modifying agent code
- **L**iskov Substitution: Any `GameEnvironment` implementation works with the agent
- **I**nterface Segregation: Small, focused protocols instead of monolithic interfaces
- **D**ependency Inversion: Agent depends on abstractions (protocols), not concrete implementations

---

## 📄 License

Built for HackTronix 2.0 — Track B: Text World Agent
