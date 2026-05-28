const { StateGraph, START, END } = require("@langchain/langgraph");
const axios = require("axios");
const { embedText, embedImageBuffer } = require('./embeddingService');
const { searchMultimodal } = require('./qdrantService');
const { generateResponse } = require('./groqService');
const { Jimp } = require('jimp');
const TrafficStats = require('../models/TrafficStats');


const GraphState = {
    question: null,
    imageBuffer: null,
    routeDecision: null,
    contextData: null,
    evidenceImages: { value: (x, y) => y, default: () => [] },
    finalAnswer: null
};

async function routerNode(state) {
    console.log("[LangGraph] Node Executing: Router");
    
    if (state.imageBuffer && !state.question) {
        return { routeDecision: "QDRANT" };
    }

    const prompt = `You are an intent classifier for a traffic analytics RAG system.
    Read the user's question and determine the correct database to query.
    - If they ask for statistics, counts, averages, totals, density, or historical aggregations (e.g., "How many cars passed yesterday?"): Output exactly "MONGODB"
    - If they ask for visual characteristics, specific events, or describe a vehicle (e.g., "Find the red truck", "Did a blue car pass?"): Output exactly "QDRANT"
    
    Question: "${state.question}"
    
    Output ONLY the database name (MONGODB or QDRANT).`;

    try {
        const response = await axios.post('https://api.groq.com/openai/v1/chat/completions', {
            messages: [{ role: "user", content: prompt }],
            model: "llama-3.3-70b-versatile",
            temperature: 0, 
        }, {
            headers: { 'Authorization': `Bearer ${process.env.GROQ_API_KEY}` }
        });
        
        const decision = response.data.choices[0].message.content.trim().toUpperCase();
        console.log(`[LangGraph] LLM Routing Decision: ${decision}`);
        
        return { routeDecision: decision.includes("MONGO") ? "MONGODB" : "QDRANT" };
    } catch (error) {
        console.error("[LangGraph] Router Error, defaulting to QDRANT", error.message);
        return { routeDecision: "QDRANT" };
    }
}

async function mongoNode(state) {
    console.log("[LangGraph] Node Executing: MongoDB Stats");
    
    const recentStats = await TrafficStats.find({ type: "daily" })
        .sort({ timebucket: -1 })
        .limit(7);

    if (recentStats.length === 0) {
        return { contextData: "No aggregated traffic statistics saved yet.", evidenceImages: [] };
    }

    let context = "### AGGREGATED DAILY TRAFFIC STATISTICS ###\n";
    recentStats.forEach(stat => {
        const dateStr = new Date(stat.timebucket).toDateString();
        const countsMap = stat.counts ? Object.fromEntries(stat.counts) : {};
        context += `- Date: ${dateStr} | Total: ${stat.totalCount || stat.vehicleCount} | Breakdown: ${JSON.stringify(countsMap)}\n`;
    });
    
    return { contextData: context, evidenceImages: [] };
}

async function qdrantNode(state) {
    console.log("[LangGraph] Node Executing: Qdrant RAG");
    const baseUrl = process.env.BASE_URL || 'http://localhost:5000';
    let imageVector = null;
    let textVector = null;

    if (state.imageBuffer) {
        try {
            console.log("Converting image format...");
            const image = await Jimp.read(state.imageBuffer);
            const processed = await image.getBuffer('image/jpeg');
            imageVector = await embedImageBuffer(processed);
        } catch (err) {
            throw new Error("Failed to process the uploaded image.");
        }
    }

    if (state.question) {
        textVector = await embedText(state.question);
    }
    
    const matches = await searchMultimodal(imageVector, textVector, 5);

    if (matches.length === 0) {
        return { contextData: "I couldn't find any relevant visual data.", evidenceImages: [] };
    }

    const context = matches.map(m => `[ID: ${m.mongo_id}] ${m.sentence}`).join('\n');
    
    const evidenceImages = matches.map(m => ({
        url: m.image_path ? `${baseUrl}/${m.image_path}` : null,
        score: m.score,
        vehicle_class: m.vehicle_class,
        vehicle_id: m.vehicle_id,
        mongo_id: m.mongo_id
    }));

    return { contextData: context, evidenceImages };
}

async function synthesizerNode(state) {
    console.log("[LangGraph] Node Executing: Final LLM Synthesis");
    const fallbackQuestion = "Summarize these visual events.";
    const llmAnswer = await generateResponse(state.contextData, state.question || fallbackQuestion);
    
    return { finalAnswer: llmAnswer };
}

const workflow = new StateGraph({ channels: GraphState })
    .addNode("router", routerNode)
    .addNode("mongo", mongoNode)
    .addNode("qdrant", qdrantNode)
    .addNode("synthesizer", synthesizerNode)
    
    .addEdge(START, "router")
    
    .addConditionalEdges("router", (state) => state.routeDecision, {
        "MONGODB": "mongo",
        "QDRANT": "qdrant"
    })
    
    .addEdge("mongo", "synthesizer")
    .addEdge("qdrant", "synthesizer")
    
    .addEdge("synthesizer", END);

const chatbotApp = workflow.compile();

async function processChatRequest(imageBuffer, userQuestion) {
    const finalState = await chatbotApp.invoke({
        question: userQuestion,
        imageBuffer: imageBuffer
    });

    return {
        answer: finalState.finalAnswer,
        evidenceImages: finalState.evidenceImages
    };
}

module.exports = { processChatRequest };