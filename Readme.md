# LiveTrafficAnalytics

A real-time traffic intelligence system that ingests live street camera feeds, detects and tracks vehicles using computer vision, and stores structured data across a multi-database architecture — enabling historical analytics, live streaming, and natural language querying through a multimodal RAG pipeline.

Built as a university project at **SRH Heidelberg University of Applied Sciences** for the Advanced Databases module.

---

## Overview

The system operates as a continuous pipeline across three layers.

**Perception** — A Python script reads a live YouTube traffic stream via VidGear, runs YOLOv8 object detection on every 4th frame, tracks vehicles across frames using a custom ByteTrack configuration, and computes entry/exit angles, sides, and dwell times. When a vehicle exits the frame, the event JSON and the cropped JPEG are pushed into Redis as separate keys.

**Storage & Enrichment** — A Node.js backend drains Redis every 5 seconds. Each event is enriched with real-time weather data from Open-Meteo, persisted to MongoDB, and written as a JPEG to local disk. A natural language sentence is generated from the event fields, embedded by `all-MiniLM-L6-v2` (384-dim), and combined with the CLIP image vector (512-dim) into a single unified Qdrant point using named vectors. Aggregates are updated across four time granularities: minute, hourly, daily, and weekly.

**Intelligence** — A multimodal RAG endpoint accepts text, image, or both. A keyword router decides whether to query Qdrant (semantic + visual retrieval) or MongoDB (aggregated statistics). Retrieved context is passed to Groq's `llama-3.3-70b-versatile` for natural language answers. A live MJPEG stream endpoint serves the annotated camera frame from Redis at 10fps to the React frontend.

---

## Architecture

```
YouTube Live Stream
        │
        ▼
Python Detector (YOLOv8 + ByteTrack)
  ├── vehicle crossing event  → Redis  traffic:events       (JSON list)
  ├── vehicle crop JPEG       → Redis  traffic:frame:*      (binary, 1h TTL)
  └── annotated live frame    → Redis  traffic:frame:live   (binary, overwritten)
        │
        ▼
Node.js Backend (every 5 seconds)
  ├── rPop event JSON from traffic:events
  ├── get() JPEG buffer from traffic:frame:* (binary via withTypeMapping)
  ├── Fetch weather → Open-Meteo API (10-min in-memory cache)
  ├── Save event    → MongoDB (vehicleevents)
  ├── Save JPEG     → Local disk (assets/img/vehicle_<id>_<ts>.jpg)
  ├── Build sentence → Embed text (all-MiniLM-L6-v2, 384-dim)
  ├── Embed image (CLIP ViT-B/32, 512-dim)
  └── Upsert unified point → Qdrant vehicle_events
        │                    (named vectors: text + image, shared payload)
        ▼
Aggregation (every event)
  └── TrafficStats upsert → MongoDB (minute / hourly / daily / weekly buckets)
        │
        ▼
RAG Chat Endpoint  POST /api/chat
  ├── Keyword router
  │     ├── QDRANT_RAG   → searchMultimodal(imageVector, textVector)
  │     │                   score fusion + deduplication → top 5
  │     └── MONGODB_STATS → TrafficStats.find() last 7 daily buckets
  └── Groq LLM (llama-3.3-70b-versatile) → natural language answer

Live Stream Endpoint  GET /api/traffic/stream
  └── Redis traffic:frame:live → MJPEG multipart stream → React
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Computer vision | Python, OpenCV, YOLOv8 (Ultralytics), VidGear |
| Object tracking | Custom ByteTrack YAML |
| Image format conversion | Jimp (PNG/WebP → JPEG before CLIP) |
| Event queue | Redis (binary image store + event list) |
| Live frame store | Redis (single overwritten key at 10fps) |
| Backend | Node.js, Express |
| Primary database | MongoDB + Mongoose |
| Vector database | Qdrant — unified collection with named vectors |
| Text embeddings | `Xenova/all-MiniLM-L6-v2` — 384-dim, runs locally |
| Image embeddings | `Xenova/clip-vit-base-patch32` — 512-dim, runs locally |
| Weather enrichment | Open-Meteo API (free, no key required) |
| LLM | Groq API (`llama-3.3-70b-versatile`) |
| Frontend | React |
| Infrastructure | Docker Compose |

---

## Project Structure

```
LiveTrafficAnalytics/
├── python/
│   ├── Camera.py               # Detector — YOLO + ByteTrack + Redis push
│   └── custom_tracker.yaml     # ByteTrack configuration
├── server/
│   ├── src/
│   │   ├── controllers/
│   │   │   ├── chatController.js     # POST /api/chat — text / image / both
│   │   │   ├── streamController.js   # GET /api/traffic/stream — MJPEG
│   │   │   └── trafficController.js  # REST stats endpoints
│   │   ├── services/
│   │   │   ├── mongoService.js       # Save event, image, embed, aggregate
│   │   │   ├── redisConsumer.js      # 5s polling loop
│   │   │   ├── qdrantService.js      # Unified upsert + multimodal search
│   │   │   ├── embeddingService.js   # MiniLM text + CLIP vision (lazy load)
│   │   │   ├── weatherService.js     # Open-Meteo + 10-min cache
│   │   │   ├── chatService.js        # RAG router + context assembly
│   │   │   ├── groqService.js        # Groq LLM call
│   │   │   └── trafficService.js     # Density + flow calculations
│   │   ├── models/
│   │   │   ├── VehicleEvent.js       # Mongoose schema (includes image_path)
│   │   │   └── TrafficStats.js       # Minute/hourly/daily/weekly buckets
│   │   ├── routes/
│   │   │   ├── chatRoute.js
│   │   │   └── trafficRoute.js
│   │   └── utils/
│   │       ├── vectorBuilder.js      # toNumericPointId helper
│   │       └── sentenceBuilder.js    # Natural language sentence from event
│   ├── assets/img/                   # Vehicle crop JPEGs (gitignored)
│   └── server.js
├── client/                           # React frontend
├── .postman/                         # Postman collection
└── docker-compose.yml                # MongoDB + Redis + Qdrant

```

---

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Node.js 18+
- Python 3.10+
- A Groq API key — free tier available at [console.groq.com](https://console.groq.com)

### 1. Start the databases

```bash
docker-compose up -d
```

Starts MongoDB on `27017`, Redis on `6379`, and Qdrant on `6333`.

### 2. Configure environment

Create `server/.env`:

```env
PORT=5000
MONGO_URI=mongodb://localhost:27017/traffic_db
REDIS_URL=redis://localhost:6379
QDRANT_URL=http://localhost:6333
GROQ_API_KEY=your_groq_api_key_here
BASE_URL=http://localhost:5000
CAM_LAT=52.5200        # latitude of camera location (used for weather)
CAM_LON=13.4050        # longitude of camera location
```

### 3. Start the Node.js backend

```bash
cd server
npm install
node server.js
```

On first run, Xenova downloads and caches two models locally:
- `all-MiniLM-L6-v2` — ~25 MB
- `clip-vit-base-patch32` — ~150 MB

Both are cached after the first run and load instantly on subsequent starts.

### 4. Start the Python detector

```bash
cd python
pip install ultralytics "vidgear[core]" opencv-python redis numpy
python Camera.py
```

To test with a local video file instead of a live stream, replace the `CamGear` source in `Camera.py` with a file path.

### 5. Start the React frontend

```bash
cd client
npm install
npm run dev
```

---

## Qdrant — Unified Collection Design

Rather than two separate collections, the system uses a **single collection with named vectors**. Each point stores both the text and image embedding under named vector keys, sharing a single payload.

```
Collection: vehicle_events
  vectors:
    text:  384-dim (all-MiniLM-L6-v2)  — semantic sentence search
    image: 512-dim (CLIP ViT-B/32)      — visual similarity search
  payload:
    mongo_id, vehicle_id, vehicle_class
    sentence, image_path
    entry_time, exit_time, entry_angle, exit_angle, entry_side, exit_side
    timestamp, hour_of_day
    weather_condition, weather_code, temperature_c, precipitation
```

This means a single upsert stores everything, and multimodal search queries both vectors independently then fuses the scores.

### Score fusion logic

When both image and text vectors are provided, results from each search are merged by `mongo_id`. If the same vehicle appears in both result sets, its scores are summed and a **+10.0 bonus** is applied — ranking vehicles that match both visually and semantically above those that match only one.

---

## Traffic Statistics API

All endpoints return pre-aggregated data from MongoDB — no full document scans.

| Endpoint | Description |
|---|---|
| `GET /api/traffic/density` | Current vs previous hour — density %, flow rate, vehicle breakdown |
| `GET /api/traffic/flow` | Average travel time for the current hour |
| `GET /api/traffic/stats/minute` | Last 60 per-minute buckets |
| `GET /api/traffic/stats/hourly` | Last 24 hourly buckets |
| `GET /api/traffic/stats/daily` | Last 7 daily buckets |
| `GET /api/traffic/stats/weekly` | Last 4 weekly buckets |
| `GET /api/traffic/stream` | MJPEG live stream from Redis (annotated frame) |

Each bucket tracks: vehicle count by class, total count, total travel time (for average calculation), and violation count.

---

## RAG Chat API

**Endpoint:** `POST /api/chat`  
**Content-Type:** `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `question` | string | Natural language question (optional if image provided) |
| `image` | file | JPEG, PNG, or WebP vehicle image (optional if question provided) |

The service automatically routes the query:

- Questions containing `how many`, `count`, `total`, `average`, `stats`, `yesterday`, `today`, or `week` → **MongoDB stats route** (last 7 daily buckets)
- Everything else → **Qdrant RAG route** (semantic + visual retrieval)

```bash
# Text only — routed to MongoDB stats
curl -X POST http://localhost:5000/api/chat \
  -F 'question=How many vehicles passed through today?'

# Text only — routed to Qdrant RAG
curl -X POST http://localhost:5000/api/chat \
  -F 'question=Were there any trucks making sharp turns at night?'

# Image only — visual similarity search
curl -X POST http://localhost:5000/api/chat \
  -F 'image=@vehicle.jpg'

# Text + Image — multimodal, both vectors used
curl -X POST http://localhost:5000/api/chat \
  -F 'question=When was this type of vehicle last seen?' \
  -F 'image=@vehicle.jpg'
```

**Response:**

```json
{
  "reply": "Three similar vehicles were detected between 01:25 and 01:28...",
  "evidenceImages": [
    {
      "url": "http://localhost:5000/assets/img/vehicle_42_1778272367123.jpg",
      "score": 14.72,
      "vehicle_class": "car",
      "vehicle_id": 42,
      "entry_time": "01:25:19",
      "exit_time": "01:25:24",
      "mongo_id": "683abc..."
    }
  ]
}
```

---

## Example Questions

| Input | Example |
|---|---|
| Text (stats route) | "How many vehicles passed through today?" |
| Text (stats route) | "What was the total count yesterday?" |
| Text (RAG route) | "Were there any trucks making sharp turns?" |
| Text (RAG route) | "Show me vehicles that entered from the south after midnight" |
| Image only | Upload a vehicle crop → find visually similar recorded crossings |
| Image + Text | "When was a vehicle like this last seen and what was the weather?" |
| Image + Text | "Was this type of vehicle behaving unusually?" |

---

## Sentence Embedding Example

Raw event fields are converted to a sentence before embedding:

```
"A car entered from the South at -166.0° at 13:27:13, continued straight 
through, and exited at -163.4° at 13:27:19. It was visible for 6 seconds. 
The weather was partly cloudy at 18°C."
```

This gives the text embedding semantic meaning about direction, timing, and environmental context — enabling queries like "vehicles entering from the south in rainy weather" to retrieve relevant matches even without exact keyword overlap.

---

## .gitignore Recommendations

```gitignore
# Environment
.env
.env.local

# Dependencies
node_modules/

# Generated vehicle images
assets/img/*
!assets/img/.gitkeep

# Xenova model cache (~175MB)
.cache/
node_modules/.cache/

# OS
.DS_Store
Thumbs.db
```

---

## License

GPL-3.0 — see [LICENSE](LICENSE) for details.