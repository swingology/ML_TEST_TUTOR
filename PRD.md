# ML Test Tutor — Product Requirements Document (PRD)

**Version:** 1.0
**Date:** 2026-03-08
**Status:** Draft

---

## 1. Overview

ML Test Tutor is a full-stack adaptive exam preparation platform modeled after the digital SAT interface. It uses a multimodal LLM (Anthropic Claude) to ingest, classify, and generate exam questions across any subject domain. The system personalizes study sessions through spaced repetition, tracks user performance by topic, and surfaces actionable reports.

### Goals
- Replicate the clean, focused UX of the College Board's digital SAT interface
- Enable flexible question ingestion from PDFs, images, text, or LLM generation
- Classify questions by topic using semantic embeddings
- Schedule review questions using a spaced repetition algorithm (FSRS)
- Track every user response and surface performance analytics
- Support any subject domain generically (not locked to SAT or ML)

---

## 2. User Roles

| Role | Description |
|---|---|
| **Student** | Takes exams, reviews sessions, views progress reports |
| **Admin** | Uploads source materials, manages question bank, configures topics |
| **System (LLM)** | Ingests documents, classifies questions, generates new questions |

---

## 3. Core Feature Areas

### 3.1 Exam Interface (Digital SAT-Style)

The frontend exam experience mirrors the College Board's digital SAT UI.

**Requirements:**
- Split-panel layout: reading passage / stimulus on left, question on right
- Multiple choice (4 options, single correct), free response (numeric input), and multi-select question types
- Timer display (per section or full exam)
- Question navigator panel: shows status (unanswered, answered, flagged)
- Flag for review toggle per question
- Passage annotation / highlight tool for reading comprehension questions
- LaTeX math rendering (KaTeX) for mathematical expressions
- Image support in question stems and answer choices (multimodal)
- Keyboard navigation support
- No partial saves mid-question — full answer commit on navigation

### 3.2 Practice Mode

A single-question drill mode without exam pressure.

**Requirements:**
- Single question display with optional hint reveal
- Immediate feedback on submission (correct/incorrect + explanation)
- LLM-generated explanation for each answer (why correct, why distractors are wrong)
- Option to add question to spaced repetition queue
- Topic tag display per question

### 3.3 Question Bank & Ingestion Pipeline

Questions enter the system through multiple channels, all managed by the backend.

**Ingestion Sources:**
- PDF / image upload (admin UI) → LLM extracts and structures questions
- Plain text or markdown paste → LLM parses into question schema
- LLM auto-generation from topic prompt and difficulty level
- JSON/CSV bulk import (structured question bank import)

**Question Schema:**
```
Question:
  id: uuid
  stem: text (supports markdown + LaTeX)
  stimulus: text | null          # reading passage, data table, image ref
  stimulus_type: text | image | table | none
  stimulus_url: url | null       # S3/MinIO reference for image stimuli
  answer_choices: [AnswerChoice]  # empty for free response
  correct_answer: str            # choice id or numeric value
  explanation: text              # LLM-generated or human-authored
  subject: str                   # e.g., "Mathematics", "Biology"
  topics: [str]                  # fine-grained tags e.g., ["Linear Equations", "Slope"]
  difficulty: 1-5
  source_document_id: uuid | null
  embedding: vector(1536)        # for semantic classification
  created_at: timestamp
  updated_at: timestamp
```

**Ingestion Pipeline (async):**
1. Admin uploads file → stored in MinIO/local object storage
2. Ingestion job queued (Celery)
3. Worker sends file to Claude API (multimodal) with structured extraction prompt
4. LLM returns array of question objects in JSON schema
5. Each question embedded via embedding model → stored with pgvector
6. Questions stored in PostgreSQL, tagged with source document reference
7. Admin reviews and approves extracted questions (optional review queue)

### 3.4 Question Classification & Retrieval

The system retrieves questions intelligently for sessions, practice, and SRS.

**Classification:**
- Each question has a topic embedding stored in pgvector
- Semantic similarity search enables retrieval by topic even without exact tag match
- Topic taxonomy is flexible — topics defined per subject, no fixed hierarchy

**Retrieval Modes:**
- By topic (exact tag or semantic similarity)
- By difficulty range
- By user error history (questions answered incorrectly, weighted by recency)
- By spaced repetition schedule (due cards for today)
- Random sampling within subject/topic filters
- Exclude recently seen questions (configurable recency window)

### 3.5 Session Management

A session represents one continuous exam or practice sitting.

**Session Types:**
- **Timed Exam:** Full exam with section timers, strict linear or module navigation
- **Practice Set:** Configurable question count, topic filter, untimed
- **Spaced Repetition Session:** Auto-generated from SRS due queue
- **Review Session:** Replay of a past session to review answers and explanations

**Session Lifecycle:**
1. Session created → configuration locked (type, topics, question count, time limit)
2. Questions selected and ordered by backend
3. User progresses through questions; each response saved immediately
4. Session ends (timer expires or user submits) → results calculated
5. SRS cards updated based on response correctness
6. Session summary generated

**Session Schema:**
```
Session:
  id: uuid
  user_id: uuid
  session_type: exam | practice | srs | review
  status: active | completed | abandoned
  config: {
    subject: str
    topics: [str]
    question_count: int
    time_limit_seconds: int | null
    difficulty_range: [int, int]
  }
  started_at: timestamp
  completed_at: timestamp | null
  score: float | null            # percentage correct

SessionQuestion:
  id: uuid
  session_id: uuid
  question_id: uuid
  order: int
  user_answer: str | null
  is_correct: bool | null
  time_spent_seconds: int
  flagged: bool
  answered_at: timestamp | null
```

### 3.6 Spaced Repetition Mode

Uses the **FSRS (Free Spaced Repetition Scheduler)** algorithm — a modern, open-source alternative to SM-2 with higher retention accuracy.

**Requirements:**
- Every question answered by a user has a corresponding SRS card
- FSRS computes next review date based on: difficulty, stability, retrievability, response grade
- Response grades: Again (0), Hard (1), Good (2), Easy (3)
- Daily SRS queue shows questions due for review
- New cards introduced at configurable rate (e.g., 10 new cards/day max)
- SRS deck filterable by subject/topic

**SRS Card Schema:**
```
SRSCard:
  id: uuid
  user_id: uuid
  question_id: uuid
  stability: float               # FSRS parameter
  difficulty: float              # FSRS parameter
  due_date: date
  last_review: timestamp | null
  review_count: int
  lapses: int                    # times rated "Again"
  state: new | learning | review | relearning
  created_at: timestamp
```

### 3.7 User Performance Reports

**Available Reports:**
- **Session Summary:** Score, time, per-question breakdown, correct/incorrect
- **Topic Mastery Map:** Heatmap of accuracy % per topic over time
- **Error Analysis:** Most missed questions/topics, common wrong answer patterns
- **Progress Over Time:** Score trends across sessions by subject
- **SRS Health:** Card maturity distribution, retention rate, daily review count
- **Streak / Activity Calendar:** Days studied, consistency tracking

**Report Access:**
- Available after each session (immediate)
- Historical dashboard view (all time)
- Exportable as PDF (nice-to-have, v2)

---

## 4. Technical Architecture

### 4.1 System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        BROWSER (Client)                         │
│               Next.js 14 + TypeScript + Tailwind                │
│         shadcn/ui components | KaTeX | TanStack Query           │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTPS REST + WebSocket
┌──────────────────────────▼──────────────────────────────────────┐
│                    BACKEND (FastAPI — Python)                    │
│                                                                 │
│  /api/auth          JWT validation (Clerk tokens)               │
│  /api/questions     CRUD, retrieval, classification             │
│  /api/sessions      Create/manage exam & practice sessions      │
│  /api/srs           SRS queue, card updates (FSRS)              │
│  /api/ingest        Document upload, LLM extraction jobs        │
│  /api/reports       Performance analytics aggregation           │
│                                                                 │
│  ┌─────────────────┐   ┌──────────────────┐                    │
│  │  LLM Service    │   │  SRS Engine      │                    │
│  │  (Claude API)   │   │  (FSRS Python)   │                    │
│  │  - extraction   │   │  - scheduling    │                    │
│  │  - generation   │   │  - grading       │                    │
│  │  - explanation  │   └──────────────────┘                    │
│  └─────────────────┘                                            │
│                                                                 │
│  ┌─────────────────┐   ┌──────────────────┐                    │
│  │  Celery Worker  │   │  BackgroundTasks  │                   │
│  │  (async jobs)   │   │  (lightweight)   │                    │
│  └────────┬────────┘   └──────────────────┘                    │
└───────────┼──────────────────────────────────────────────────── ┘
            │
┌───────────▼──────────────────────────────────────────────────── ┐
│                         DATA LAYER                              │
│                                                                 │
│  PostgreSQL 16 + pgvector    Redis              MinIO           │
│  - Users                     - Session cache    - PDFs          │
│  - Questions + embeddings    - SRS queues       - Images        │
│  - Sessions + responses      - Rate limiting    - Audio         │
│  - SRS cards                 - Celery broker    (local S3)      │
│  - Reports (materialized)                                       │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Frontend Stack

| Concern | Technology |
|---|---|
| Framework | Next.js 14 (App Router) |
| Language | TypeScript |
| Styling | Tailwind CSS |
| UI Components | shadcn/ui |
| State (exam session) | Zustand |
| Server state / caching | TanStack Query (React Query) |
| Math rendering | KaTeX + react-markdown |
| Charts | Recharts |
| Authentication | Clerk |
| Forms | React Hook Form + Zod |

### 4.3 Backend Stack

| Concern | Technology |
|---|---|
| Framework | FastAPI (Python 3.12+) |
| ORM | SQLAlchemy 2.0 (async) |
| Migrations | Alembic |
| LLM Client | `anthropic` Python SDK |
| LLM Orchestration | LangChain (ingestion chains) |
| SRS Algorithm | `fsrs` Python library |
| Background Jobs | Celery + Redis broker |
| Auth Validation | python-jose (JWT) |
| Validation | Pydantic v2 |
| Testing | pytest + httpx |

### 4.4 Data Layer

| Concern | Technology |
|---|---|
| Primary Database | PostgreSQL 16 |
| Vector Search | pgvector extension |
| Cache / Broker | Redis 7 |
| Object Storage | MinIO (local S3-compatible) |

### 4.5 Local Development

All services run via **Docker Compose**:

```yaml
services:
  frontend:   # Next.js dev server (port 3000)
  backend:    # FastAPI with uvicorn (port 8000)
  worker:     # Celery worker
  db:         # PostgreSQL 16 + pgvector
  redis:      # Redis 7
  minio:      # MinIO object storage (port 9000 / 9001 console)
```

A single `docker compose up` starts the full stack locally.

---

## 5. Data Models (Conceptual)

```
User
  id, email, name, created_at, clerk_user_id

Question
  id, stem, stimulus, stimulus_type, stimulus_url,
  answer_choices (JSONB), correct_answer, explanation,
  subject, topics (array), difficulty, source_document_id,
  embedding (vector), created_at, updated_at

SourceDocument
  id, filename, storage_url, ingestion_status, admin_id, created_at

Session
  id, user_id, session_type, status, config (JSONB),
  started_at, completed_at, score

SessionQuestion
  id, session_id, question_id, order, user_answer,
  is_correct, time_spent_seconds, flagged, answered_at

SRSCard
  id, user_id, question_id, stability, difficulty,
  due_date, last_review, review_count, lapses, state

Topic
  id, name, subject, description, parent_topic_id (nullable)

PerformanceSummary (materialized view or table)
  user_id, subject, topic, accuracy_pct, sessions_count,
  last_session_at, window_days
```

---

## 6. API Surface (Key Endpoints)

```
POST   /api/auth/validate              Validate Clerk token → issue session

GET    /api/questions                  List/filter questions (topic, difficulty, subject)
POST   /api/questions                  Create question (admin)
GET    /api/questions/{id}             Get single question
PUT    /api/questions/{id}             Update question (admin)

POST   /api/ingest/upload              Upload document → queue ingestion job
GET    /api/ingest/jobs/{id}           Check ingestion job status
GET    /api/ingest/review              List extracted questions pending admin review
POST   /api/ingest/review/{id}/approve Approve extracted question

POST   /api/sessions                   Create session
GET    /api/sessions/{id}              Get session state
POST   /api/sessions/{id}/answer       Submit answer for a question in session
POST   /api/sessions/{id}/complete     End session, trigger report generation
GET    /api/sessions/{id}/review       Get full session review with explanations

GET    /api/srs/queue                  Get today's SRS due cards for user
POST   /api/srs/grade                  Submit grade for SRS card (Again/Hard/Good/Easy)
GET    /api/srs/stats                  SRS deck statistics

GET    /api/reports/summary            Overall user performance summary
GET    /api/reports/topics             Topic mastery breakdown
GET    /api/reports/sessions           Session history list
GET    /api/reports/sessions/{id}      Detailed single session report
GET    /api/reports/errors             Most frequently missed questions/topics
```

---

## 7. LLM Integration Details

### 7.1 Ingestion Prompt Strategy

When a document is uploaded, the system sends it to Claude with a structured extraction prompt:

```
You are an expert exam question parser. Given the following document content,
extract all exam questions and return them as a JSON array matching this schema:
[{ stem, stimulus, answer_choices, correct_answer, explanation, subject, topics, difficulty }]

Rules:
- Preserve LaTeX math notation using $...$ inline and $$...$$ block
- If a question references an image, note it as [IMAGE_REF_n]
- Infer subject and fine-grained topic tags
- Rate difficulty 1 (easiest) to 5 (hardest)
- Generate a clear explanation for the correct answer

Document content:
{document_text}
```

### 7.2 Question Generation

Admin can generate questions by topic + difficulty:

```
Generate {n} multiple-choice questions on the topic "{topic}"
at difficulty level {difficulty}/5 for subject "{subject}".
Return as JSON array following the question schema.
Include LaTeX for any mathematical notation.
```

### 7.3 Explanation Generation

For questions without existing explanations, Claude generates on first request and caches:

```
Question: {stem}
Correct answer: {correct_answer}
Distractors: {other_choices}

Write a clear, concise explanation (3-5 sentences) of:
1. Why the correct answer is right
2. The key concept being tested
3. Why each distractor is wrong (briefly)
```

---

## 8. Spaced Repetition — FSRS Details

FSRS (Free Spaced Repetition Scheduler) is implemented via the `fsrs` Python library.

**Grading Scale (shown to user as buttons):**
- **Again** — Completely forgot / wrong answer
- **Hard** — Correct but very difficult
- **Good** — Correct with some effort
- **Easy** — Correct, very easy

**System behavior:**
- New questions begin as "new" cards
- After first correct answer, enters "learning" phase with short intervals
- Graduates to "review" phase with exponentially growing intervals
- Lapses (Again) reset to "relearning"
- FSRS parameters can be optimized per-user from review history (future v2)

---

## 9. Phased Roadmap

### Phase 1 — Foundation (Local Dev MVP)
- [ ] Docker Compose environment (all services)
- [ ] PostgreSQL schema + Alembic migrations
- [ ] FastAPI skeleton with auth middleware
- [ ] Question CRUD API
- [ ] Basic Next.js frontend shell with exam interface layout
- [ ] Manual question entry (admin)

### Phase 2 — Core Exam Experience
- [ ] Full exam session flow (create, answer, complete)
- [ ] Session review with explanations
- [ ] Practice mode with immediate feedback
- [ ] Basic performance report (session summary)

### Phase 3 — LLM Integration
- [ ] Document ingestion pipeline (PDF → Claude → questions)
- [ ] Question generation from topic prompts
- [ ] On-demand explanation generation
- [ ] Topic classification via embeddings + pgvector

### Phase 4 — Spaced Repetition
- [ ] FSRS card creation on question answer
- [ ] Daily SRS queue endpoint
- [ ] SRS session type in frontend
- [ ] SRS statistics and deck health view

### Phase 5 — Analytics & Polish
- [ ] Topic mastery heatmap report
- [ ] Error analysis report
- [ ] Progress over time charts
- [ ] Admin question review workflow
- [ ] Performance tuning, caching

---

## 10. Out of Scope (v1)

- Mobile native app (responsive web only)
- Real-time collaborative study sessions
- Payment / subscription system
- PDF export of reports
- Multi-tenant (single-user local deployment initially)
- Custom FSRS parameter optimization per user
- Audio/video question stimuli (image only in v1)

---

## 11. Open Questions

1. **Auth provider:** Clerk is recommended for speed; can swap to Auth.js for full self-hosting
2. **Embedding model:** Use Claude's embeddings or a dedicated model (e.g., `text-embedding-3-small`)? Dedicated is cheaper for high volume.
3. **Admin review queue:** Is human review of LLM-extracted questions required, or auto-approve?
4. **FSRS vs SM-2:** FSRS chosen; can revisit if library maturity is a concern
5. **Question ordering in exams:** Randomized, difficulty-adaptive, or fixed? Configurable per session config.
