const { timeToSeconds } = require('./vectorBuilder');

function angleToCompass(deg) {
    const dirs = ['North', 'Northeast', 'East', 'Southeast', 'South', 'Southwest', 'West', 'Northwest'];
    return dirs[Math.round((((deg % 360) + 360) % 360) / 45) % 8];
}

function describeMovement(normDelta, exitDir) {
    if (normDelta < 20)       
        return 'continued straight through';
    if (normDelta < 60)       
        return `turned slightly toward the ${exitDir}`;
    if (normDelta < 120)      
        return `made a moderate turn toward the ${exitDir}`;
    if (normDelta < 160)      
        return `made a sharp turn toward the ${exitDir}`;
    return `made a near U-turn toward the ${exitDir}`;
}

function buildSentence(doc, weather) {
    const entryAngle = parseFloat(doc.entry_angle ?? 0);
    const exitAngle  = parseFloat(doc.exit_angle  ?? 0);
    const entryDir   = angleToCompass(entryAngle);
    const exitDir    = angleToCompass(exitAngle);
    const cls        = (doc.class || 'vehicle').toLowerCase();
    const entryTime  = doc.entry_time || '00:00:00';
    const exitTime   = doc.exit_time  || '00:00:00';
    const colorText  = (doc.color && doc.color !== 'Unknown') ? `${doc.color.toLowerCase()} ` : '';

    const dwellSec = Math.max(
        timeToSeconds(exitTime) - timeToSeconds(entryTime),
        0
    );

    let angleDelta = Math.abs(exitAngle - entryAngle) % 360;
    if (angleDelta > 180) angleDelta = 360 - angleDelta;

    const movement = describeMovement(angleDelta, exitDir);

    let weatherClause = '';
    if (weather?.condition && weather.condition !== 'unknown') {
        weatherClause = ` The weather was ${weather.condition}` +
            (weather.temperature_c !== null ? ` at ${weather.temperature_c}°C`              : '') +
            (weather.precipitation > 0     ? ` with ${weather.precipitation}mm precipitation` : '') +
            '.';
    }

    return (
        `A ${colorText}${cls} entered from the ${entryDir} at ${entryAngle}° at ${entryTime}, ` +
        `${movement}, and exited at ${exitAngle}° at ${exitTime}. ` +
        `It was visible for ${dwellSec} seconds.` +
        weatherClause
    );
}

module.exports = { buildSentence, angleToCompass, describeMovement };