const axios = require('axios');
 
/**
 * Sends the retrieved context and user question to Ollama
 */
async function generateResponse(context, question) {
   
            const systemPrompt = `### ROLE: Traffic Security & Intelligence Analyst
            ### DIRECTIVE: 
            You process retrieved database records (Context) to answer user queries about traffic flow and specific vehicle sightings. The context is a combination of image vector and the text vector.

            ### RULES FOR CITATION:
            When you refer to a specific vehicle sighting, you MUST include its [ID: {mongo_id}] in your response exactly as it appears in the context. 
            Example: "I found a red car matching your description that exited left [ID: 64b8f...]."
            Do not mention the ID if you are summarizing general statistics.

            ### OPERATING PARAMETERS:
            - TWO TYPES OF CONTEXT: You will either receive "SPECIFIC VEHICLE SIGHTINGS" or "AGGREGATED DAILY TRAFFIC STATISTICS".
            - IF SPECIFIC SIGHTINGS: Highlight anomalies. Use bold text for **Vehicle Classes** and **Timestamps**.
            - IF STATISTICS: Analyze the numbers naturally. If the user asks about "yesterday," look at the dates provided in the context to determine which day is yesterday based on today's current date. 
            - BE DIRECT: Start your answer immediately. Do not say "Based on the context provided...". Just give the answer.
            - IF NO RELEVANT CONTEXT: Respond with a helpful message indicating the lack of relevant information.`;

            const userPrompt = `### DATABASE EVIDENCE:
            ${context}
 
                ### USER QUESTION:
                ${question}
 
                ### ANSWER:`.trim();
        try {
        // const ollamaResponse = await axios.post('http://127.0.0.1:11434/api/generate', {
        //     model: 'llama3', // Make sure you have pulled this model in Ollama
        //     prompt: prompt,
        //     stream: false
        // });
 
        const groqResponse = await axios.post('https://api.groq.com/openai/v1/chat/completions', {
            messages: [
                { role: "system", content: systemPrompt },
                { role: "user", content: userPrompt }
            ],
            model: "llama-3.3-70b-versatile",
            stream: false
        }, {
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${process.env.GROQ_API_KEY}`
            }
        })
 
        // const [groqResult] = await Promise.all([groqResponse]);
 
        return groqResponse.data.choices[0].message.content;
    } catch (error) {
        if (error.response) {
        // This will tell you exactly which parameter Groq didn't like
        console.error("Groq Error Details:", error.response.data);
        }
        console.error('Connection Error:', error.message);
        return "I'm having trouble connecting to Traffic Analyst right now.";
    }
}
 
module.exports = { generateResponse };