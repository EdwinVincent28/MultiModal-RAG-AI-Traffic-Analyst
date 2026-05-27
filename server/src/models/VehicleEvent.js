const mongoose = require('mongoose');

const vehicleEventSchema = new mongoose.Schema({
    vehicle_id: { type: Number, required: true },
    class:      { type: String, required: true },
    color: { 
        type: String, 
        default: "Unknown" 
    },
    entry_side:  { type: String },
    entry_angle: { type: Number },
    entry_time:  { type: String },
    exit_side:   { type: String },
    exit_angle:  { type: Number },
    exit_time:   { type: String },
    image_path:  { type: String, default: null },
    timestamp:   { type: Number },
}, { 
    timestamps: true  
});

module.exports = mongoose.model('VehicleEvent', vehicleEventSchema);