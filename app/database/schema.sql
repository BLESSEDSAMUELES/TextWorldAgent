-- =============================================================================
-- Text World Agent — SQLite Schema
-- =============================================================================
-- 8 normalized tables with UPSERT-friendly unique constraints.
-- Contradicting observations UPDATE existing rows instead of duplicating.
-- =============================================================================

PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- Rooms: one row per unique room
CREATE TABLE IF NOT EXISTS rooms (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    description TEXT DEFAULT '',
    visited     BOOLEAN DEFAULT 0,
    visit_count INTEGER DEFAULT 0,
    first_seen  INTEGER DEFAULT 0,
    last_seen   INTEGER DEFAULT 0
);

-- Connections: directed edges between rooms (one exit per direction per room)
CREATE TABLE IF NOT EXISTS connections (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    from_room_id  INTEGER NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
    to_room_id    INTEGER REFERENCES rooms(id) ON DELETE SET NULL,
    direction     TEXT NOT NULL,
    locked        BOOLEAN DEFAULT 0,
    lock_key      TEXT,
    UNIQUE(from_room_id, direction)
);

-- Objects: anything interactable in the world
CREATE TABLE IF NOT EXISTS objects (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    description TEXT DEFAULT '',
    room_id     INTEGER REFERENCES rooms(id) ON DELETE SET NULL,
    portable    BOOLEAN DEFAULT 1,
    state       TEXT DEFAULT 'default',
    properties  TEXT DEFAULT '{}'
);

-- Inventory: what the player is currently carrying
CREATE TABLE IF NOT EXISTS inventory (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    object_id     INTEGER NOT NULL REFERENCES objects(id) ON DELETE CASCADE UNIQUE,
    acquired_step INTEGER NOT NULL
);

-- NPCs: characters in the world
CREATE TABLE IF NOT EXISTS npcs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    description TEXT DEFAULT '',
    room_id     INTEGER REFERENCES rooms(id) ON DELETE SET NULL,
    dialogue    TEXT,
    state       TEXT DEFAULT 'idle'
);

-- Room States: key-value pairs per room with UPSERT semantics
CREATE TABLE IF NOT EXISTS room_states (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    room_id      INTEGER NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
    state_key    TEXT NOT NULL,
    state_value  TEXT NOT NULL,
    updated_step INTEGER DEFAULT 0,
    UNIQUE(room_id, state_key)
);

-- Observed Facts: RDF-style triples with conflict resolution
CREATE TABLE IF NOT EXISTS observed_facts (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    subject        TEXT NOT NULL,
    predicate      TEXT NOT NULL,
    object         TEXT NOT NULL,
    confidence     REAL DEFAULT 1.0,
    source_step    INTEGER NOT NULL,
    superseded_by  INTEGER REFERENCES observed_facts(id) ON DELETE SET NULL,
    active         BOOLEAN DEFAULT 1
);

-- Agent Memory: high-level strategic memories with relevance decay
CREATE TABLE IF NOT EXISTS agent_memory (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_type  TEXT NOT NULL CHECK(memory_type IN ('goal','plan','observation','failure','success')),
    content      TEXT NOT NULL,
    relevance    REAL DEFAULT 1.0,
    created_step INTEGER NOT NULL,
    active       BOOLEAN DEFAULT 1
);

-- =============================================================================
-- Indexes for query performance
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_connections_from ON connections(from_room_id);
CREATE INDEX IF NOT EXISTS idx_objects_room ON objects(room_id);
CREATE INDEX IF NOT EXISTS idx_npcs_room ON npcs(room_id);
CREATE INDEX IF NOT EXISTS idx_room_states_room ON room_states(room_id);
CREATE INDEX IF NOT EXISTS idx_facts_active ON observed_facts(active);
CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_subject_pred ON observed_facts(subject, predicate) WHERE active = 1;
CREATE INDEX IF NOT EXISTS idx_facts_subject ON observed_facts(subject) WHERE active = 1;
CREATE INDEX IF NOT EXISTS idx_memory_active ON agent_memory(active);
CREATE INDEX IF NOT EXISTS idx_memory_relevance ON agent_memory(relevance) WHERE active = 1;
