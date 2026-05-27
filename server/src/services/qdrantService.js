const { QdrantClient }  = require('@qdrant/js-client-rest');
const { toNumericPointId } = require('../utils/vectorBuilder');

const qdrant = new QdrantClient({ url: process.env.QDRANT_URL });
 
const COLLECTION_NAME = 'vehicle_events';

async function ensureCollections() {
    const { collections } = await qdrant.getCollections();
    const names = collections.map(c => c.name);

    if (!names.includes(COLLECTION_NAME)) {
        await qdrant.createCollection(COLLECTION_NAME, {
            vectors: { 
                text: { size: 384, distance: 'Cosine' },
                image: { size: 512, distance: 'Cosine' }
            },
        });
        for (const [field, type] of [
            ['vehicle_class', 'keyword'],
            ['vehicle_color', 'keyword'],
            ['hour_of_day',   'integer'],
            ['weather_code',  'integer'],
            ['timestamp',     'float'  ],
        ]) {
            await qdrant.createPayloadIndex(COLLECTION_NAME, {
                field_name:   field,
                field_schema: type,
            });
        }
        console.log('[QdrantService] Created Unified Collection:', COLLECTION_NAME);
    }
}

async function upsertVehicle(event, sentence, textVector, imageVector, weather) {
    const numericId = toNumericPointId(event._id);

    const vectors = { text: textVector };
    if (imageVector) {
        vectors.image = imageVector;
    }

    await qdrant.upsert(COLLECTION_NAME, {
        wait: true,
        points: [{
            id: numericId,
            vector: vectors,
            payload: {
                mongo_id:          event._id.toString(),
                vehicle_id:        event.vehicle_id,
                vehicle_class:     event.class,
                vehicle_color:     event.color || "Unknown",
                sentence:          sentence,
                image_path:        event.image_path ?? null, // Image path stored right here!
                entry_time:        event.entry_time,
                exit_time:         event.exit_time,
                timestamp:         event.timestamp,
                hour_of_day:       new Date(event.timestamp * 1000).getUTCHours(),
                weather_condition: weather.condition,
                weather_code:      weather.code,
                temperature_c:     weather.temperature_c,
                precipitation:     weather.precipitation,
            },
        }],
    });

    console.log(`[QdrantService] Unified vehicle upserted for ID=${event.vehicle_id}`);
}

async function searchMultimodal(imageVector = null, textVector = null, limit = 5) {
    let combinedResults = [];
    
    const isHybrid = imageVector && textVector;
    const fetchLimit = isHybrid ? 20 : limit;

    if (imageVector) {
        const visualHits = await qdrant.search(COLLECTION_NAME, {
            vector: { name: 'image', vector: imageVector },
            limit: fetchLimit,
            score_threshold: 0.85, 
            with_payload: true
        });
        combinedResults.push(...visualHits.map(h => ({ 
            ...h.payload, 
            score: h.score, 
            fromImage: true 
        })));
    }

    if (textVector) {
        const textHits = await qdrant.search(COLLECTION_NAME, {
            vector: { name: 'text', vector: textVector }, 
            limit: fetchLimit,
            score_threshold: 0.30,
            with_payload: true
        });
        combinedResults.push(...textHits.map(h => ({ 
            ...h.payload, 
            score: h.score, 
            fromText: true 
        })));
    }

    const uniqueMap = new Map();
    for (const item of combinedResults) {
        if (!uniqueMap.has(item.mongo_id)) {
            uniqueMap.set(item.mongo_id, item);
        } else {
            const existing = uniqueMap.get(item.mongo_id);
            existing.score += item.score; 
            
            if ((existing.fromImage && item.fromText) || (existing.fromText && item.fromImage)) {
                existing.score += 10.0; 
            }
            
            existing.fromImage = existing.fromImage || item.fromImage;
            existing.fromText = existing.fromText || item.fromText;
            
            uniqueMap.set(item.mongo_id, existing);
        }
    }

    return Array.from(uniqueMap.values())
        .sort((a, b) => b.score - a.score)
        .slice(0, limit);
}

module.exports = { ensureCollections, upsertVehicle, searchMultimodal};