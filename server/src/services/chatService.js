const { embedText, embedImageBuffer } = require('./embeddingService');
const { searchMultimodal } = require('./qdrantService');
const { generateResponse } = require('./groqService');
const { Jimp } = require('jimp');
const TrafficStats = require('../models/TrafficStats');

function determineRoute(question) {
    if (!question) 
        return "QDRANT_RAG";
    
    const statKeywords = /how many|count|total|average|stats|statistics|yesterday|today|week/i;
    
    if (statKeywords.test(question)) {
        return "MONGODB_STATS";
    }
    return "QDRANT_RAG";
}

async function processChatRequest(imageBuffer, userQuestion) {
    const baseUrl = process.env.BASE_URL || 'http://localhost:5000';

    const route = determineRoute(userQuestion);

    let context = "";
    let evidenceImages = [];

    if (route === "MONGODB_STATS") {
        console.log("[ChatService] Routing to MONGODB_STATS");
        
        const recentStats = await TrafficStats.find({ type: "daily" })
            .sort({ timebucket: -1 })
            .limit(7);

        if (recentStats.length === 0) {
            return { answer: "I don't have any aggregated traffic statistics saved yet.", evidenceImages: [] };
        }

        context = "### AGGREGATED DAILY TRAFFIC STATISTICS ###\n";
        recentStats.forEach(stat => {
            const dateStr = new Date(stat.timebucket).toDateString();
            
            const countsMap = stat.counts ? Object.fromEntries(stat.counts) : {};
            const breakdown = JSON.stringify(countsMap);
            
            context += `- Date: ${dateStr} | Total Vehicles: ${stat.totalCount || stat.vehicleCount} | Breakdown: ${breakdown}\n`;
        });
        evidenceImages = [];
    }
    else{
        console.log("[ChatService] Routing to QDRANT_RAG");
        let imageVector = null;
        let textVector = null;

        if (imageBuffer) {
        // Auto-convert any format (PNG, WebP, etc.) to JPEG before processing
            let processedImageBuffer = imageBuffer;

            try {
                console.log("Converting PNG to JPEG using Jimp...");
                const image = await Jimp.read(imageBuffer);
                processedImageBuffer = await image.getBuffer('image/jpeg');
            } catch (err) {
                console.error("Image conversion failed:", err.message);
                throw new Error("Failed to process the uploaded image.");
            }

            imageVector = await embedImageBuffer(processedImageBuffer);
        }

        if (userQuestion) {
            textVector = await embedText(userQuestion);
        }
        
        const matches = await searchMultimodal(imageVector, textVector, 5);

        if (matches.length === 0) {
            return { answer: "I couldn't find any relevant data.", evidenceImages: [] };
        }

        context = matches.map(m => `[ID: ${m.mongo_id}] ${m.sentence}`).join('\n');
        

        const evidenceUrl = `${baseUrl}/${matches.imagePath}`;

        evidenceImages = matches.map(m => ({
            url: m.image_path ? `${baseUrl}/${m.image_path}` : null,
            score: m.score,
            vehicle_class: m.vehicle_class,
            vehicle_id: m.vehicle_id,
            entry_time: m.entry_time,
            exit_time: m.exit_time,
            mongo_id: m.mongo_id
        }));


    }
    const llmAnswer = await generateResponse(context, userQuestion || "Summarize these events.");
    return {
            answer: llmAnswer,
            evidenceImages: evidenceImages
    };

}

module.exports = { processChatRequest };